"""Sensor contract: the only place hardware is allowed to enter the stack.

Adapted from the ABC removed in ``1987f00`` to the MuJoCo-based estimator: a
sensor no longer carries a ``MeasurementModel`` object, it declares which
``sensordata`` rows its readings correspond to (``rows``) and their noise block
(``R``). The model of what those rows *should* read is MuJoCo itself.

Sensors may depend on :mod:`erp.core`. The dependency never runs the other way.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from erp.core.types import Array, CalibrationResult, IntArray, Measurement

__all__ = ["Sensor", "SensorError", "identity_calibration", "shared_rows_R", "sqrt_psd"]


def sqrt_psd(R: npt.ArrayLike) -> Array:
    """A matrix ``L`` with ``L @ L.T == R``; Cholesky when possible, eigen otherwise.

    Used to colour white noise into N(0, R) for the simulated sensors. The
    eigen path covers a PSD ``R`` with zero-variance channels, which Cholesky
    rejects -- an ``R`` built from a config where one sigma is 0 is a
    configuration choice, not an error, and it should produce a noiseless
    channel rather than a ``LinAlgError`` at construction time.
    """
    R_a = np.asarray(R, dtype=np.float64)
    try:
        return np.asarray(np.linalg.cholesky(R_a), dtype=np.float64)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(0.5 * (R_a + R_a.T))
        return np.asarray(V * np.sqrt(np.clip(w, 0.0, None)), dtype=np.float64)


def shared_rows_R(rows: npt.ArrayLike, R: npt.ArrayLike) -> tuple[IntArray, Array]:
    """Validate ``rows``/``R`` and return private read-only copies.

    Every :class:`~erp.core.types.Measurement` of a sensor references these
    same two arrays instead of copying them per sample. Making them read-only
    is what makes that sharing safe: a caller mutating ``m.R`` in place would
    otherwise silently change the noise of every past and future sample.
    """
    rows_a = np.array(rows, dtype=np.intp).reshape(-1)
    R_a = np.array(R, dtype=np.float64)
    k = rows_a.size
    if R_a.shape != (k, k):
        raise ValueError(f"R must be ({k}, {k}) to match rows, got {R_a.shape}")
    if k and rows_a.min() < 0:
        raise ValueError("rows must be non-negative sensordata indices")
    rows_a.flags.writeable = False
    R_a.flags.writeable = False
    return rows_a, R_a


def identity_calibration(dim: int, note: str) -> CalibrationResult:
    """Calibration for sources whose data is already calibrated (replay, sim)."""
    return CalibrationResult(
        bias=np.zeros(dim), scale=np.ones(dim), R=np.zeros((dim, dim)), valid=True, note=note
    )


class SensorError(RuntimeError):
    """A sensor stopped producing data for a reason the caller must see.

    Raised from :meth:`Sensor.read` / :meth:`Sensor.drain` when a background
    reader died (port unplugged, device reset). Returning an empty list instead
    would be indistinguishable from "no new samples yet", which is how a sensor
    quietly stops contributing while every plot still looks fine.
    """


class Sensor(ABC):
    """A measurement source, real, replayed or simulated.

    Both accessors are non-blocking: the loop that calls them also streams the
    trajectory (and later runs the filter), so it cannot wait on a slow
    device. Implementations return what is available and nothing more.

    Implementations put ``Measurement.timestamp`` into the absolute
    ``time.perf_counter()`` host base, converting from a device clock where
    necessary. That conversion must not leak into the caller.
    """

    name: str

    @property
    @abstractmethod
    def rows(self) -> IntArray:
        """(k,) indices into ``model.sensordata`` this sensor measures."""

    @property
    @abstractmethod
    def R(self) -> Array:
        """(k, k) measurement noise covariance, units of ``z`` squared."""

    @abstractmethod
    def read(self) -> Measurement | None:
        """Return the oldest unread sample, or ``None`` if none is available."""

    @abstractmethod
    def drain(self) -> list[Measurement]:
        """Return all available samples, ordered by timestamp; ``[]`` if none.

        This is the accessor the trajectory/estimation loop uses; ``read``
        exists for interactive and diagnostic use.
        """

    @abstractmethod
    def calibrate(self) -> CalibrationResult:
        """Run the sensor's calibration procedure.

        Callers must check :attr:`~erp.core.types.CalibrationResult.valid`
        rather than assuming the returned result is usable.
        """
