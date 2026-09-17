"""Core value types shared by sensors, logging and (later) the estimators.

Nothing here may import from :mod:`erp.sensors`, ``firmware`` or any ROS 2
package. These types must stay constructible in a notebook, in CI and against
recorded data with zero hardware attached.

Time convention
---------------
Every timestamp in this module is **absolute seconds of** ``time.perf_counter()``
on the host, shared by all sensors and by the trajectory loop. No sensor
subtracts its own ``t0``: two sensors that each rebased to their own start
would disagree about "now" by however far apart they were started. Converting a
device clock (e.g. Teensy ``micros()``) into this base is a ``Sensor``
responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

__all__ = ["Array", "CalibrationResult", "IntArray", "Measurement"]


# ``eq=False`` throughout: the generated ``__eq__`` would compare ndarray fields
# elementwise and then call bool() on the result, which raises. Use
# np.allclose explicitly when value equality is meant.


@dataclass(frozen=True, slots=True, eq=False)
class Measurement:
    """One sensor reading, stamped at the instant the sample was taken.

    The channels are addressed by position in MuJoCo's ``data.sensordata``, so
    the MuJoCo EKF can consume it as ``y = z - h(x)[rows]`` with ``R`` as the
    noise block, without knowing which physical device produced it.

    Attributes
    ----------
    z:
        (k,) calibrated reading in the units and frame of the MuJoCo sensors
        named by ``rows``: accelerometer m/s^2 and gyro rad/s, both in the
        sensor *site* frame. A reading that has not been axis-mapped into the
        site frame does not belong in this field.
    timestamp:
        Seconds, absolute ``time.perf_counter()`` host base. This is the
        **sample** instant, never the arrival instant: USB transport latency
        must already be removed by the sensor.
    rows:
        (k,) indices into ``model.sensordata``. One array shared read-only by
        every sample of a sensor -- not copied per sample.
    R:
        (k, k) measurement noise covariance, units of ``z`` squared. Shared and
        read-only like ``rows``, so the filter can use it without an
        ``np.ix_`` gather on every update.
    source:
        Name of the sensor that produced the sample.
    """

    z: Array
    timestamp: float
    rows: IntArray
    R: Array
    source: str


@dataclass(frozen=True, eq=False)
class CalibrationResult:
    """Outcome of a :meth:`~erp.sensors.base.Sensor.calibrate` call.

    Attributes
    ----------
    bias:
        (k,) additive offset in the calibrated units of ``z``, subtracted from
        readings.
    scale:
        (k,) multiplicative per-channel factor, dimensionless.
    R:
        (k, k) noise covariance estimated during calibration, units of ``z``
        squared.
    valid:
        ``False`` when the procedure ran but its data cannot be trusted (e.g.
        the arm moved during a static calibration). Callers must check this
        before applying the result.
    note:
        Human-readable detail, e.g. why a calibration was rejected.
    """

    bias: Array
    scale: Array
    R: Array
    valid: bool
    note: str = ""
