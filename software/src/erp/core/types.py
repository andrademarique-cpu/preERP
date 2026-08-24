"""Core value types: the vocabulary every layer of the stack speaks.

Nothing here may import from :mod:`erp.sensors`, ``firmware`` or any ROS 2
package. These types must stay constructible in a notebook, in CI and against
recorded data with zero hardware attached.

Time convention
---------------
Every timestamp in this module is **seconds in one monotonic host time base**,
shared by all sensors and inputs. Converting a device clock into that base is a
``Sensor`` responsibility and must never appear in :mod:`erp.fusion`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

__all__ = ["Array", "CalibrationResult", "ControlInput", "GaussianState", "Measurement"]


# ``eq=False`` throughout: the dataclass-generated ``__eq__`` would compare
# ndarray fields elementwise and then call bool() on the resulting array, which
# raises "truth value of an array is ambiguous". Identity comparison is the
# sane default for these; use np.allclose explicitly when you mean value
# equality.


@dataclass(frozen=True, eq=False)
class Measurement:
    """One sensor reading, stamped at the instant the sample was taken.

    Attributes
    ----------
    z:
        Measured value. Units and reference frame are fixed by the
        :class:`~erp.models.base.MeasurementModel` that consumes it, e.g.
        linear acceleration in m/s^2 expressed in the sensor frame.
    timestamp:
        Seconds, monotonic host time base. This is the **sample** instant,
        never the arrival instant: transport latency shows up as the gap
        between this value and the moment the measurement reaches the
        :class:`~erp.fusion.engine.FusionEngine`.
    frame_id:
        Name of the reference frame ``z`` is expressed in.
    R:
        Optional per-sample noise covariance override, in units of ``z``
        squared. ``None`` means "use the measurement model's ``R``".
    """

    z: Array
    timestamp: float
    frame_id: str
    R: Array | None = None


@dataclass(frozen=True, eq=False)
class ControlInput:
    """A commanded actuator input, latched at ``timestamp`` and held.

    The value is piecewise constant (zero-order hold) until the next input.
    That is not an approximation of the hardware, it *is* what the hardware
    does, which is why :class:`~erp.core.timeline.InputHistory` can reproduce
    ``u`` exactly at any query time.

    Attributes
    ----------
    u:
        Commanded input. For the finger servos this is the target joint angle
        in radians (not a torque), matching MuJoCo's ``data.ctrl``.
    timestamp:
        Seconds, monotonic host time base, at which the command was latched.
        Actuation transport delay is *not* folded in here; it is applied by
        :class:`~erp.core.timeline.InputHistory`.
    """

    u: Array
    timestamp: float


@dataclass
class GaussianState:
    """A Gaussian belief over the state vector.

    Attributes
    ----------
    x:
        Mean. Units are per-element and defined by the process model, e.g.
        ``[rad, rad, rad/s, rad/s, rad, rad]`` for a two-joint servoed finger.
    P:
        Covariance, units of ``x`` squared. Must be symmetric positive
        semi-definite; see :func:`erp.core.linalg.make_spd`.
    """

    x: Array
    P: Array

    def __post_init__(self) -> None:
        if self.x.ndim != 1:
            raise ValueError(f"x must be 1-D, got shape {self.x.shape}")
        n = self.x.size
        if self.P.shape != (n, n):
            raise ValueError(f"P must be ({n}, {n}) to match x, got {self.P.shape}")

    @property
    def dim(self) -> int:
        """Number of state dimensions."""
        return int(self.x.size)

    def copy(self) -> GaussianState:
        """Return a deep copy, so callers cannot alias a filter's internals."""
        return GaussianState(self.x.copy(), self.P.copy())


@dataclass(frozen=True, eq=False)
class CalibrationResult:
    """Outcome of a :meth:`~erp.sensors.base.Sensor.calibrate` call.

    Attributes
    ----------
    bias:
        Additive offset in the sensor's own units, to be subtracted from raw
        readings.
    scale:
        Multiplicative per-channel scale factor, dimensionless.
    R:
        Noise covariance estimated during calibration, units of the sensor
        reading squared.
    valid:
        ``False`` when the procedure ran but did not converge. Callers must
        check this rather than assuming a returned result is usable.
    note:
        Human-readable detail, e.g. why a calibration was rejected.
    """

    bias: Array
    scale: Array
    R: Array
    valid: bool
    note: str = ""
