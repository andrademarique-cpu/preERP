"""Timestamps into filter steps. The only module in ``erp`` that does that.

This package is **English**, alongside ``core/``, ``io/``, ``sensors/`` and
``robot/`` rather than alongside ``estimators/``. The split CLAUDE.md describes
is physics-and-filtering against hardware-and-plumbing, and there is no physics
here: :class:`FilterRunner` decides *when* the filter steps and never what it
computes. The arithmetic stays in :mod:`erp.estimators.ekf`, in Spanish.

What this replaces
------------------
``run_imu_ekf``, which lived in a notebook cell and in
``scripts/make_golden_run.py``. ADR-0002 3.4 is the charge sheet against it:
fixed 2 ms predicts, ``int(round(tk / dt))`` for placement, a ``k_now`` cursor
that never rewinds, **no late-measurement branch and no counter**. It assumes
one sorted, on-time stream, and a second sensor with a different latency breaks
it silently -- which is the failure ADR-0001 D6 was written about.

The step arithmetic here is deliberately the *same* arithmetic, because the
golden run has to reproduce at ``rtol=1e-12`` across this move. What is new is
everything that happens when the assumption fails: ordering, dropping and
counting.

The two rates, and why nothing here relates them
------------------------------------------------
Commands leave at 25 Hz because that is what the serial round trip allows; IMU
lines arrive at ~20 Hz because that is what the firmware does. They coincide
only every 200 ms (ADR-0002 4.7), so a loop written as "one measurement per
control tick" -- the obvious way to write it -- drops one sample in five here
and breaks differently at every rate. **The filter's step count is a function
of measurement timestamps alone.** ``advance_to`` is never called with a
command tick, and no method takes a ``u``: the EKF is blind by construction and
this runner is the layer that would have been tempted to un-blind it.

Recording
---------
``record=False`` turns off the per-step state trace, which is the only part
that grows with the run -- (n, nx, nx) covariances at 500 steps per second. The
NIS and its timestamps are always kept: two floats per measurement, and the
diagnostic that decides whether the filter is consistent at all. A flag that
silently disabled *that* would be a trap.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from erp.core.clock import Clock
from erp.core.types import Array, Belief, Measurement, UpdateInfo
from erp.models.base import DiscreteDynamics

__all__ = ["Estimator", "FilterRunner", "History"]


class Estimator(Protocol):
    """What :class:`FilterRunner` needs from a filter. ``EKF`` satisfies it structurally.

    This is **not** the ``StateEstimator`` ABC of ADR-0002 4.4. P4 declined to
    build that one -- an ABC with a single implementation is the abstraction
    4.5 exists to refuse -- and P5 does not revive it. What P5 needs is a
    written record of the surface the runner depends on, which is a structural
    protocol, the same shape as
    :class:`~erp.robot.mypalletizer.ArmTransport` and
    :class:`~erp.models.base.DiscreteDynamics`. Nothing has to inherit from it.

    Keeping it here rather than in ``estimators/`` is what lets ``fusion`` stay
    off ``estimators`` in the import graph: the runner is handed a filter, it
    does not go looking for one.
    """

    dyn: DiscreteDynamics
    x: Array
    P: Array

    def predict(self) -> None:
        """Advance exactly one ``dyn.dt``. No ``u``: blind by construction."""
        ...

    def update(
        self,
        z: npt.ArrayLike,
        rows: npt.ArrayLike,
        R: npt.ArrayLike | None = None,
    ) -> tuple[Array, float]:
        """Correct with channels ``rows`` of the FULL vector ``z``. ``(innovation, NIS)``."""
        ...


@dataclass(frozen=True, eq=False)
class History:
    """The trace of one run: every filter step, and every update's NIS.

    ``eq=False`` for the same reason as the types in :mod:`erp.core.types`: the
    generated ``__eq__`` would compare arrays elementwise and then call
    ``bool()`` on the result, which raises.

    Attributes
    ----------
    t:
        (n,) seconds, absolute host base. **Timestamps repeat at updates.**
        Each predict appends one row, and each update appends another at the
        *same* time, so the trace carries the prior and the posterior of every
        measurement as consecutive rows. That is what makes a covariance band
        show how far it grows between samples and how far an update pulls it
        back; averaging over ``t`` without knowing this double-counts the
        update instants. Inherited from ``run_imu_ekf``, whose recording
        convention the golden fixture froze.
    x:
        (n, nx) state at each row of ``t``. Empty when ``record=False``.
    P:
        (n, nx, nx) covariance at each row of ``t``. Empty when ``record=False``.
    nis:
        (m,) normalised innovation squared, one per **applied** measurement.
        Dropped measurements contribute nothing here and are counted in
        ``discarded`` instead -- a run whose ``nis`` looks healthy because most
        samples never reached the filter is exactly what that counter exists to
        make visible.
    t_update:
        (m,) the filter step each update landed on, so ``nis`` can be plotted
        against time without re-deriving it from ``t``.
    discarded:
        Measurements dropped as late. See :meth:`FilterRunner.ingest`.
    """

    t: Array
    x: Array
    P: Array
    nis: Array
    t_update: Array
    discarded: int


class FilterRunner:
    """Drives an estimator from measurement timestamps. Never from commands.

    Two entry points over one object (ADR-0002 4.3): :meth:`run` for the
    offline path, replaying a log with no hardware attached, and :meth:`ingest`
    for the online one, called from ``run_trajectory``'s existing per-tick
    ``drain()`` hook. They share all of the time logic, which is the point --
    M1 and M2 must not differ by a line of arithmetic, or the timing layer has
    leaked into the estimator.

    Parameters
    ----------
    est:
        The filter. Held, not constructed here.
    t0:
        Absolute host seconds the filter's clock starts at. The offline path
        passes ``0.0`` because the replayed log is already relative to the
        trajectory's start; the online path passes ``run_trajectory``'s ``t0``,
        so that filter time and ``Measurement.timestamp`` share one origin.
    buffer_horizon:
        Seconds to hold a measurement before applying it, so that streams with
        **different** transport delays can be sorted into timestamp order. The
        estimate then trails real time by this much: it buys ordering with
        latency, and there is no third option. ``0.0`` is correct for a single
        sensor -- there is nothing to reorder -- and it earns its keep the
        moment the arm's encoder readback becomes a second stream (ADR-0002
        4.7 rule 3, and open question 3).
    clock:
        Optional time base, consulted **only** when ``buffer_horizon > 0``.
        Without it the release watermark is the newest timestamp ingested,
        which is deterministic and is what the offline path wants; a run is
        then a pure function of its input. With it, held samples are also
        released once wall time has moved past them, so a sensor that goes
        quiet does not strand the ones already buffered. ADR-0002 4.4's
        signature omits this parameter while P5's phase text asks for it; it is
        optional here because only the online path has an answer for it.
    record:
        Keep the per-step state trace. See the module docstring.

    Notes
    -----
    Nothing here reads a wall clock directly -- ADR-0002 4.8's third mechanism,
    and the reason this module passes the grep P7.5 adds. Time enters only
    through ``Measurement.timestamp`` and, optionally, an injected
    :class:`~erp.core.clock.Clock`.
    """

    def __init__(
        self,
        est: Estimator,
        *,
        t0: float = 0.0,
        buffer_horizon: float = 0.0,
        clock: Clock | None = None,
        record: bool = True,
    ) -> None:
        if buffer_horizon < 0.0:
            raise ValueError(f"buffer_horizon must be >= 0, got {buffer_horizon}")
        self.est = est
        self.t0 = float(t0)
        self.buffer_horizon = float(buffer_horizon)
        self.clock = clock
        self.discarded = 0
        self._dt = float(est.dyn.dt)
        self._k = 0
        # Scattered into and reused, never reallocated: one serial line becomes
        # exactly one update over 12 of the 15 channels, and `EKF.update` reads
        # only the rows it is given.
        self._z_full: Array = np.zeros(est.dyn.nz, dtype=np.float64)
        self._pending: list[Measurement] = []
        self._watermark = -np.inf
        self._record = record
        self._t: list[float] = []
        self._x: list[Array] = []
        self._P: list[Array] = []
        self._nis: list[float] = []
        self._t_upd: list[float] = []
        if self._record:
            self._append_state()

    # ------------------------------------------------------------------ time

    @property
    def t_filter(self) -> float:
        """Absolute host seconds the filter has been advanced to.

        Quantised to ``dyn.dt``: the filter exists only at step boundaries, so
        this is ``t0 + k * dt`` and not the timestamp of the last measurement.
        """
        return self.t0 + self._k * self._dt

    @property
    def steps(self) -> int:
        """Predicts taken since construction."""
        return self._k

    @property
    def pending(self) -> int:
        """Measurements held by the buffer, not yet applied."""
        return len(self._pending)

    @property
    def belief(self) -> Belief:
        """The current estimate, copied.

        Copied because ``x`` and ``P`` belong to the estimator, which is free
        to write through them; a caller holding the live arrays would be
        holding something that changes under it on the next predict.
        """
        return Belief(x=self.est.x.copy(), P=self.est.P.copy(), t=self.t_filter)

    def _step_of(self, t: float) -> int:
        """Which filter step ``t`` belongs to. Nearest, not floor.

        Nearest is what ``run_imu_ekf`` did and what the fixture froze, and it
        is also right: at a 2 ms step and 50 ms between IMU samples, placing a
        measurement at the nearest step costs at most 1 ms of misalignment
        against a 50 ms gap, where flooring would cost up to 2 ms and would be
        biased late rather than centred.
        """
        # `round`, not `int(round(...))`: ruff flags the cast as redundant
        # (RUF046) and it is -- `round` on a float already returns an int. P3
        # made the same change in `resample` and pinned it as bit-identical.
        return round((float(t) - self.t0) / self._dt)

    def advance_to(self, t: float) -> int:
        """Predict up to the step containing ``t``. Returns the steps taken.

        Never rewinds: a ``t`` already passed takes zero steps and is not an
        error, because the command loop legitimately calls this with its own
        tick time, which can sit anywhere relative to the last measurement.
        """
        n = self._step_of(t) - self._k
        if n <= 0:
            return 0
        for _ in range(n):
            self.est.predict()
            self._k += 1
            if self._record:
                self._append_state()
        return n

    def advance_to_safe(self, t_now: float) -> int:
        """Advance as far as is safe given the transport latency: ``t_now - buffer_horizon``.

        **This is what an online loop should call, not** :meth:`advance_to`.
        Advancing to *now* looks right and is a trap: a sensor with any
        transport latency stamps the sample instant and hands it over later, so
        a filter already advanced to now receives every sample stamped in its
        past and drops all of them. Measured on the dry-run loop at 25 Hz with
        a 5 ms simulated latency, advancing to ``tick.t_actual`` discards
        samples that advancing to ``tick.t_actual - 0.005`` keeps.

        So ``buffer_horizon`` is one parameter with one meaning -- how far
        behind real time the estimate runs -- governing both when a held
        measurement is released and how far the filter dares step. At ``0.0``
        this is :meth:`advance_to`, which is correct only for a source with no
        latency, such as a replayed log.

        It also refuses to step past anything the buffer is **still holding**.
        Without that clamp, a horizon set without a ``clock`` discards every
        sample it buffers: the release watermark is then the newest timestamp
        ingested, so a sample waits for its successor, while ``t_now`` marches
        on and the filter walks past it in the meantime. The cost of the clamp
        is that the estimate lags by an extra sample period in that
        configuration -- which is the price of not injecting a clock, paid
        visibly instead of as a silent drop.
        """
        limit = float(t_now) - self.buffer_horizon
        if self._pending:
            limit = min(limit, min(m.timestamp for m in self._pending))
        return self.advance_to(limit)

    # -------------------------------------------------------------- ingestion

    def ingest(self, m: Measurement) -> UpdateInfo | None:
        """Take one measurement. Advance to it, then update.

        Returns the update it caused, or ``None`` if none was applied. With
        ``buffer_horizon > 0`` a single call may release several held
        measurements or none at all, and the return value is then the most
        recent update *this call* produced -- the rest are in the history and
        in the counters either way.

        **A measurement that rounds to a step the filter has already passed is
        dropped and counted in ``discarded``.** Not raised: a live sensor
        producing a late sample is a normal event, and a runner that raised
        would take the command loop down with it. Not applied either, because
        a backwards update is not a Kalman update -- the covariance it was
        supposed to correct has already been propagated past it.

        Note the criterion is the *step*, not the timestamp. A sample stamped
        0.9 ms before the filter's current time still rounds to the step the
        filter is on, and is applied with zero predicts. Sub-step jitter is not
        out-of-order, and counting it as such would report drops on a stream
        that is merely irregular.
        """
        if self.buffer_horizon <= 0.0:
            return self._apply(m)
        self._pending.append(m)
        self._watermark = max(self._watermark, m.timestamp)
        watermark = self._watermark
        if self.clock is not None:
            watermark = max(watermark, self.clock.now())
        released = self._release(watermark - self.buffer_horizon)
        return released[-1] if released else None

    def flush(self) -> list[UpdateInfo]:
        """Apply everything the buffer is still holding, in timestamp order.

        Called at the end of :meth:`run`, and by the online loop after the last
        setpoint. Without it a ``buffer_horizon`` silently eats the tail of
        every run.
        """
        return self._release(float("inf"))

    def run(self, ms: Iterable[Measurement]) -> History:
        """Replay a whole log and return the trace. The offline path.

        The measurements do **not** have to arrive sorted -- that is what
        ``buffer_horizon`` is for -- but with the default ``0.0`` they are
        applied in the order given, and anything out of order is dropped and
        counted.
        """
        for m in ms:
            self.ingest(m)
        self.flush()
        return self.history()

    def _release(self, t_limit: float) -> list[UpdateInfo]:
        """Apply every held measurement stamped at or before ``t_limit``."""
        if not self._pending:
            return []
        self._pending.sort(key=lambda m: m.timestamp)
        keep_from = len(self._pending)
        for i, m in enumerate(self._pending):
            if m.timestamp > t_limit:
                keep_from = i
                break
        ready = self._pending[:keep_from]
        del self._pending[:keep_from]
        out = []
        for m in ready:
            info = self._apply(m)
            if info is not None:
                out.append(info)
        return out

    def _apply(self, m: Measurement) -> UpdateInfo | None:
        if self._step_of(m.timestamp) < self._k:
            self.discarded += 1
            return None
        self.advance_to(m.timestamp)
        self._z_full[m.rows] = m.z
        # `m.R` and not the filter's default block. This is the other half of
        # ADR-0002 3.2: until P5 the measurement's R stopped at MeasurementLog,
        # so `SerialIMUSensor.calibrate()` changed nothing about the estimate.
        # Today both paths give the same number -- the sensor's R is built as
        # exactly the block `EKF.update` was already slicing out -- which is
        # why this move leaves the golden run at rtol=1e-12. It starts to
        # matter at P6, when calibrate() first runs on the arm.
        y, nis = self.est.update(self._z_full, m.rows, m.R)
        t = self.t_filter
        if self._record:
            self._append_state()
        self._nis.append(nis)
        self._t_upd.append(t)
        # `m.rows` shared, not copied -- it is the sensor's one read-only array
        # (`shared_rows_R`), and an UpdateInfo is a report, not an owner.
        return UpdateInfo(innovation=y, nis=nis, rows=m.rows, t=t)

    # ---------------------------------------------------------------- history

    def _append_state(self) -> None:
        self._t.append(self.t_filter)
        self._x.append(self.est.x.copy())
        self._P.append(self.est.P.copy())

    def history(self) -> History:
        """Snapshot the trace so far. Cheap to call mid-run."""
        nx = self.est.dyn.nx
        return History(
            t=np.asarray(self._t, dtype=np.float64),
            x=(
                np.asarray(self._x, dtype=np.float64)
                if self._x
                else np.empty((0, nx), dtype=np.float64)
            ),
            P=(
                np.asarray(self._P, dtype=np.float64)
                if self._P
                else np.empty((0, nx, nx), dtype=np.float64)
            ),
            nis=np.asarray(self._nis, dtype=np.float64),
            t_update=np.asarray(self._t_upd, dtype=np.float64),
            discarded=self.discarded,
        )

    def __repr__(self) -> str:
        return (
            f"FilterRunner(t_filter={self.t_filter:.6f}, steps={self._k}, "
            f"updates={len(self._nis)}, discarded={self.discarded}, pending={self.pending})"
        )
