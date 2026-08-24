"""Hardware-free :class:`~erp.sensors.base.Sensor` implementations.

``ReplaySensor`` is the Adapter that makes offline validation exercise the same
code path as the robot: the fusion engine cannot tell it from a driver.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from erp.core.types import CalibrationResult, Measurement
from erp.models.base import MeasurementModel
from erp.sensors.base import Sensor

__all__ = ["DummySensor", "ReplaySensor"]


def _trivial_calibration(dim: int, note: str) -> CalibrationResult:
    return CalibrationResult(
        bias=np.zeros(dim),
        scale=np.ones(dim),
        R=np.zeros((dim, dim)),
        valid=True,
        note=note,
    )


class ReplaySensor(Sensor):
    """Serve pre-recorded measurements through the live sensor interface.

    Because the whole record is available up front, :meth:`drain` hands over
    everything remaining and lets the fusion engine's buffer horizon decide
    what is releasable. Out-of-order arrival therefore cannot occur in replay:
    it is a property of live transport, not of the data.
    """

    def __init__(self, measurements: Sequence[Measurement], model: MeasurementModel) -> None:
        """
        Parameters
        ----------
        measurements:
            Recorded samples. Must be non-decreasing in ``timestamp``; a record
            that is out of order is a logging bug and is rejected here rather
            than silently reordered.
        model:
            Measurement model for this sensor.
        """
        ts = [m.timestamp for m in measurements]
        if any(b < a for a, b in zip(ts, ts[1:])):
            raise ValueError("replay measurements must be non-decreasing in timestamp")
        self._measurements = list(measurements)
        self._model = model
        self._cursor = 0

    @property
    def remaining(self) -> int:
        """Number of samples not yet handed out."""
        return len(self._measurements) - self._cursor

    def read(self) -> Measurement | None:
        """Return the next unread sample, or ``None`` at end of record."""
        if self._cursor >= len(self._measurements):
            return None
        m = self._measurements[self._cursor]
        self._cursor += 1
        return m

    def drain(self) -> list[Measurement]:
        """Return every remaining sample, ordered by timestamp."""
        out = self._measurements[self._cursor :]
        self._cursor = len(self._measurements)
        return out

    def calibrate(self) -> CalibrationResult:
        """Return an identity calibration: recorded data is already calibrated."""
        dim = self._measurements[0].z.size if self._measurements else 0
        return _trivial_calibration(dim, "replay: recorded data, no calibration performed")

    @property
    def measurement_model(self) -> MeasurementModel:
        """Model mapping state to this sensor's expected reading."""
        return self._model


class DummySensor(Sensor):
    """Constant-output stub emitting one sample per call at a fixed period.

    Exists so that interface-level tests can run before any driver does. A test
    that passes against this and against a real implementation is what proves
    the :class:`~erp.sensors.base.Sensor` contract is sound.
    """

    def __init__(
        self,
        model: MeasurementModel,
        z: np.ndarray,
        period: float,
        *,
        t0: float = 0.0,
        frame_id: str = "dummy",
    ) -> None:
        """
        Parameters
        ----------
        model:
            Measurement model this stub pretends to satisfy.
        z:
            Constant reading returned every time, in the model's units.
        period:
            Seconds between successive samples, strictly positive.
        t0:
            Timestamp of the first sample, seconds in the host time base.
        frame_id:
            Reference frame reported on every measurement.
        """
        if period <= 0.0:
            raise ValueError(f"period must be > 0, got {period}")
        self._model = model
        self._z = np.asarray(z, dtype=np.float64)
        self._period = float(period)
        self._next_t = float(t0)
        self._frame_id = frame_id

    def read(self) -> Measurement | None:
        """Return one sample and advance the internal clock by ``period``."""
        m = Measurement(z=self._z.copy(), timestamp=self._next_t, frame_id=self._frame_id)
        self._next_t += self._period
        return m

    def drain(self) -> list[Measurement]:
        """Return exactly one sample, so the stub cannot run away with the clock."""
        m = self.read()
        return [] if m is None else [m]

    def calibrate(self) -> CalibrationResult:
        """Return an identity calibration."""
        return _trivial_calibration(self._z.size, "dummy: no calibration performed")

    @property
    def measurement_model(self) -> MeasurementModel:
        """Model mapping state to this sensor's expected reading."""
        return self._model
