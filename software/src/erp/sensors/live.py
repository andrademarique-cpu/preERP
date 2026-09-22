"""Virtual IMU over a *live* plant, for interactive (teleoperated) runs.

:class:`~erp.sensors.sim.SimSensor` cannot do this job, and the difference is
structural rather than a missing option. ``SimSensor`` is a
:class:`~erp.sensors.replay.ReplaySensor` over a **precomputed** ``(t,
sensordata)`` array whose noise is drawn once at construction; ``rate_hz`` and
``latency_s`` decimate and pace a log that already exists. Teleoperation has no
such log -- the truth is produced as the operator presses keys, one physics step
at a time -- so the sampling, the noise draw and the release all have to happen
online.

Imports no mujoco, for the same reason ``SimSensor`` does not: the caller owns
the ``MjData`` and pushes the current reading in through :meth:`poll`. That
keeps :mod:`erp.sensors.mujoco` the only module in this package that touches the
mujoco API, and it makes every test here a pure-array test that runs in the fast
suite with no model and no display attached.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from erp.core.types import Array, CalibrationResult, IntArray, Measurement
from erp.sensors.base import Sensor, identity_calibration, shared_rows_R, sqrt_psd

__all__ = ["LiveSimSensor"]


class LiveSimSensor(Sensor):
    """Sample a running plant at a fixed rate, add N(0, R), release with latency.

    The loop calls :meth:`poll` every physics step with the plant's current
    ``sensordata``; this class decides when that constitutes a sample, corrupts
    it and queues it. :meth:`drain` then behaves exactly like every other
    sensor's, so the filter loop cannot tell this apart from the Teensy.

    Like ``SimSensor``, the noise is drawn from the **same** ``R`` the filter is
    given, and the plant *is* the model. The filter is therefore handed exactly
    the world it assumes, which makes anything built on this a plumbing check --
    that timestamps, row indices, axis maps and the loop all line up -- and not
    evidence that the estimator is good. On the real arm the same filter gives
    NIS median 29 against 9.6 here.
    """

    def __init__(
        self,
        *,
        rows: npt.ArrayLike,
        R: npt.ArrayLike,
        name: str = "live",
        rate_hz: float = 20.0,
        latency_s: float = 0.0,
        seed: int = 0,
        noise: bool = True,
        clock: Callable[[], float] | None = None,
        keep_truth: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        rows, R:
            Channels to emit (indices into ``data.sensordata``) and their noise
            covariance, in units of the MuJoCo sensors squared. Accelerometer
            channels are m/s^2 and gyro channels rad/s, both in the sensor
            **site** frame, so ``R`` mixes units by block -- build it with
            :func:`erp.sensors.mujoco.make_R` rather than by hand.
        rate_hz:
            Sampling rate, Hz. Samples land on an absolute grid anchored at the
            first :meth:`poll`, ``t0 + k / rate_hz``, not on a running
            ``t + 1/rate_hz``: the latter accumulates the physics step's
            remainder and drifts, which is the same failure ``RateLoop`` exists
            to avoid on the command side.
        latency_s:
            Transport delay, seconds. With ``clock`` set, a sample taken at
            ``t`` is withheld until ``clock() >= t + latency_s``; without one it
            is available as soon as it is taken.
        seed:
            Noise RNG seed. A run is reproducible given the same key presses.
        noise:
            ``False`` emits the clean readings -- useful for telling whether a
            disagreement is noise or a wiring error.
        clock:
            Host clock for paced release, injected so a test can drive it with
            ``VirtualClock``. ``None`` releases everything as it is sampled.
        keep_truth:
            Record the clean reading beside every sample, so the "measured vs
            truth" strip costs nothing. Turn it off for a long unattended run;
            the lists grow without bound.
        """
        if rate_hz <= 0.0:
            raise ValueError(f"rate_hz must be > 0, got {rate_hz}")
        if latency_s < 0.0:
            raise ValueError(f"latency_s must be >= 0, got {latency_s}")
        self.name = name
        self._rows, self._R = shared_rows_R(rows, R)
        self.rate_hz = float(rate_hz)
        self.latency_s = float(latency_s)
        self.clock = clock
        self.noise = bool(noise)
        self._L = sqrt_psd(self._R)
        self._rng = np.random.default_rng(seed)

        self._queue: list[Measurement] = []
        self._ts: list[float] = []
        self._cursor = 0
        self._t0: float | None = None
        self._k = 0

        self.keep_truth = bool(keep_truth)
        self.truth_t: list[float] = []
        """Sample instants, host seconds -- one per emitted sample."""
        self.truth_z: list[Array] = []
        """(k,) noise-free readings at those instants, in the same order."""

    # -- live ingestion -----------------------------------------------------

    def poll(self, t: float, sensordata: npt.ArrayLike) -> Measurement | None:
        """Offer the plant's reading at time ``t``; sample it if one is due.

        Parameters
        ----------
        t:
            Time of this reading, seconds, in the **same base as ``clock``**.
            Pass the host clock's value, not the simulation's step count: the
            filter compares this against its own advance, and two bases that
            differ by a constant put every sample in the past or the future.
        sensordata:
            (nsensordata,) the plant's full current ``data.sensordata``;
            ``rows`` selects from it.

        Returns
        -------
        The queued :class:`~erp.core.types.Measurement`, or ``None`` when this
        reading falls between samples -- which is the common case, since the
        plant steps at 2 ms and this samples at ~20 Hz, roughly 25 steps apart.
        """
        t = float(t)
        if self._t0 is None:
            self._t0 = t
        due = self._t0 + self._k / self.rate_hz
        if t + 1e-12 < due:
            return None
        self._k += 1

        full = np.asarray(sensordata, dtype=np.float64).reshape(-1)
        if self._rows.size and self._rows.max() >= full.size:
            raise ValueError(
                f"rows reach index {self._rows.max()}, sensordata has {full.size} entries"
            )
        clean = full[self._rows]
        z = clean
        if self.noise and clean.size:
            z = clean + self._L @ self._rng.standard_normal(clean.size)

        m = Measurement(np.asarray(z, dtype=np.float64), t, self._rows, self._R, self.name)
        self._queue.append(m)
        self._ts.append(t)
        if self.keep_truth:
            self.truth_t.append(t)
            self.truth_z.append(clean.copy())
        return m

    @property
    def sampled(self) -> int:
        """Samples taken so far, released or not."""
        return len(self._queue)

    @property
    def pending(self) -> int:
        """Samples taken but not yet handed out."""
        return len(self._queue) - self._cursor

    # -- contract -----------------------------------------------------------

    @property
    def rows(self) -> IntArray:
        return self._rows

    @property
    def R(self) -> Array:
        return self._R

    def _release_end(self) -> int:
        if self.clock is None:
            return len(self._queue)
        return bisect_right(self._ts, self.clock() - self.latency_s, lo=self._cursor)

    def read(self) -> Measurement | None:
        if self._cursor >= self._release_end():
            return None
        m = self._queue[self._cursor]
        self._cursor += 1
        return m

    def drain(self) -> list[Measurement]:
        end = self._release_end()
        out = self._queue[self._cursor : end]
        self._cursor = max(self._cursor, end)
        return out

    def calibrate(self) -> CalibrationResult:
        """Identity: a simulated sensor's readings are already calibrated."""
        return identity_calibration(self._rows.size, f"{self.name}: simulated, not calibrated")
