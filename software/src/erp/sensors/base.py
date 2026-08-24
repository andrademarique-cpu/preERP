"""Sensor contract: the only place hardware is allowed to enter the stack.

One of the four ABCs defining the system contract; changing a signature here
requires sign-off from every module owner (``CLAUDE.md``). Current signatures
are those adopted in ``docs/adr/0001-multi-rate-fusion.md`` decision D1.

Sensors may depend on :mod:`erp.core` and :mod:`erp.models`. The dependency
never runs the other way: adding a measurement backend means adding a
:class:`Sensor` subclass, never reaching into the estimation core.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from erp.core.types import CalibrationResult, Measurement
from erp.models.base import MeasurementModel

__all__ = ["Sensor"]


class Sensor(ABC):
    """A measurement source, real or replayed.

    Both accessors are non-blocking. A blocking single-sample pull cannot serve
    a 1 kHz IMU and a 100 Hz force sensor from one loop without either aliasing
    the fast channel or stalling on the slow one, so implementations return
    what is available and nothing more.

    Implementations are responsible for putting ``Measurement.timestamp`` into
    the single monotonic host time base, converting from a device clock where
    necessary. That conversion must not leak into :mod:`erp.fusion`.
    """

    @abstractmethod
    def read(self) -> Measurement | None:
        """Return the next unread sample, or ``None`` if none is available."""

    @abstractmethod
    def drain(self) -> list[Measurement]:
        """Return all samples since the previous call, ordered by timestamp.

        Returns an empty list when nothing is available. This is the accessor
        the fusion layer uses; ``read`` exists for interactive and diagnostic
        use.
        """

    @abstractmethod
    def calibrate(self) -> CalibrationResult:
        """Run the sensor's calibration procedure.

        Callers must check :attr:`~erp.core.types.CalibrationResult.valid`
        rather than assuming the returned result is usable.
        """

    @property
    @abstractmethod
    def measurement_model(self) -> MeasurementModel:
        """Model mapping state to this sensor's expected reading."""
