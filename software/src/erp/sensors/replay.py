"""Recorded measurements served through the live :class:`Sensor` interface.

The loop that drains a :class:`ReplaySensor` cannot tell it from the Teensy,
which is the point: offline validation exercises the same code path as the robot.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np
import numpy.typing as npt

from erp.core.types import Array, CalibrationResult, IntArray, Measurement
from erp.sensors.base import Sensor, identity_calibration, shared_rows_R
from erp.sensors.imu_serial import IMUDecoder

__all__ = ["ReplaySensor"]


class ReplaySensor(Sensor):
    """Serve pre-recorded measurements, all at once or paced by a clock.

    Without ``clock``, :meth:`drain` hands over everything remaining (tests,
    batch processing). With ``clock`` (e.g. ``time.perf_counter``), a sample
    becomes available only once ``clock() >= timestamp + latency_s``, so a
    replay inside the trajectory loop arrives at the same pace, and with the
    same transport delay, as the live device would.
    """

    def __init__(
        self,
        measurements: Sequence[Measurement],
        *,
        rows: npt.ArrayLike,
        R: npt.ArrayLike,
        name: str = "replay",
        clock: Callable[[], float] | None = None,
        latency_s: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        measurements:
            Samples, non-decreasing in ``timestamp`` (a record out of order is
            a logging bug and is rejected rather than silently reordered).
        rows, R:
            What this sensor declares; must match the samples' shapes.
        clock:
            Host clock for paced release; ``None`` releases everything.
        latency_s:
            Seconds between a sample's timestamp and its release, >= 0.
        """
        if latency_s < 0.0:
            raise ValueError(f"latency_s must be >= 0, got {latency_s}")
        self.name = name
        self._rows, self._R = shared_rows_R(rows, R)
        ts = [m.timestamp for m in measurements]
        if any(b < a for a, b in pairwise(ts)):
            raise ValueError("replay measurements must be non-decreasing in timestamp")
        k = self._rows.size
        for m in measurements[:1]:
            if m.z.shape != (k,):
                raise ValueError(f"samples have z shape {m.z.shape}, rows has {k} entries")
        self._measurements = list(measurements)
        self._ts = ts
        self._cursor = 0
        self.clock = clock
        self.latency_s = float(latency_s)

    # -- construction helpers ---------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        t: npt.ArrayLike,
        Z: npt.ArrayLike,
        *,
        rows: npt.ArrayLike,
        R: npt.ArrayLike,
        name: str = "replay",
        t_shift: float = 0.0,
        clock: Callable[[], float] | None = None,
        latency_s: float = 0.0,
    ) -> ReplaySensor:
        """Build from (N,) times and (N, k) calibrated readings.

        ``t_shift`` is added to every time: pass ``time.perf_counter()`` taken
        at the start of the run to turn a log's relative seconds into the
        absolute host base every other sensor uses.
        """
        rows_a, R_a = shared_rows_R(rows, R)
        t_a = np.asarray(t, dtype=np.float64).reshape(-1) + t_shift
        Z_a = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        if Z_a.shape != (t_a.size, rows_a.size) and t_a.size:
            raise ValueError(f"Z must be ({t_a.size}, {rows_a.size}), got {Z_a.shape}")
        ms = [
            Measurement(Z_a[i].copy(), float(t_a[i]), rows_a, R_a, name) for i in range(t_a.size)
        ]
        return cls(ms, rows=rows_a, R=R_a, name=name, clock=clock, latency_s=latency_s)

    @classmethod
    def from_log_csv(
        cls,
        path: str | Path,
        *,
        rows: npt.ArrayLike,
        R: npt.ArrayLike,
        name: str = "replay",
        t_shift: float = 0.0,
        clock: Callable[[], float] | None = None,
        latency_s: float = 0.0,
    ) -> ReplaySensor:
        """Load a CSV written by :meth:`erp.io.log.MeasurementLog.save_csv`."""
        data = _load_csv(path)
        return cls.from_arrays(
            data[:, 0], data[:, 1:], rows=rows, R=R, name=name,
            t_shift=t_shift, clock=clock, latency_s=latency_s,
        )

    @classmethod
    def from_legacy_imu_csv(
        cls,
        path: str | Path,
        decoder: IMUDecoder,
        *,
        rows: npt.ArrayLike,
        R: npt.ArrayLike,
        name: str = "imu",
        t_shift: float = 0.0,
        clock: Callable[[], float] | None = None,
        latency_s: float = 0.0,
    ) -> ReplaySensor:
        """Load a raw device log (``timestamp,IMU_0.ax,...``) through ``decoder``.

        Columns are matched to ``decoder.keys`` by header name, not position,
        so a log with reordered columns still decodes correctly.
        """
        path = Path(path)
        with path.open(encoding="utf-8") as fh:
            header = [h.strip() for h in fh.readline().split(",")]
        col = {h: i for i, h in enumerate(header)}
        missing = [k for k in decoder.keys if k not in col]
        if missing:
            raise ValueError(f"{path.name} has no columns for {missing}")
        data = _load_csv(path)
        raw = data[:, [col[k] for k in decoder.keys]]
        return cls.from_arrays(
            data[:, 0], decoder.apply(raw), rows=rows, R=R, name=name,
            t_shift=t_shift, clock=clock, latency_s=latency_s,
        )

    # -- contract -----------------------------------------------------------

    @property
    def rows(self) -> IntArray:
        return self._rows

    @property
    def R(self) -> Array:
        return self._R

    @property
    def remaining(self) -> int:
        """Samples not yet handed out (released or not)."""
        return len(self._measurements) - self._cursor

    def _release_end(self) -> int:
        if self.clock is None:
            return len(self._measurements)
        return bisect_right(self._ts, self.clock() - self.latency_s, lo=self._cursor)

    def read(self) -> Measurement | None:
        if self._cursor >= self._release_end():
            return None
        m = self._measurements[self._cursor]
        self._cursor += 1
        return m

    def drain(self) -> list[Measurement]:
        end = self._release_end()
        out = self._measurements[self._cursor : end]
        self._cursor = max(self._cursor, end)
        return out

    def rewind(self) -> None:
        """Serve the record again from the first sample."""
        self._cursor = 0

    def calibrate(self) -> CalibrationResult:
        """Identity: recorded data is already calibrated."""
        return identity_calibration(self._rows.size, f"{self.name}: recorded data, not calibrated")


def _load_csv(path: str | Path) -> Array:
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return np.asarray(data, dtype=np.float64)
