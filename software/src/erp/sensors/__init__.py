"""Sensor interface and implementations.

Importing this package needs numpy only: pyserial is imported when a serial
sensor is started, and the MuJoCo helpers in :mod:`erp.sensors.mujoco`
(``rows_of``, ``make_R``) are deliberately not re-exported here -- import them
from that module explicitly.
"""

from erp.sensors.base import Sensor, SensorError, identity_calibration, shared_rows_R
from erp.sensors.clock import ArrivalClock, ClockSync, HostClock
from erp.sensors.imu_serial import (
    IMUDecoder,
    LineTransport,
    SerialIMUSensor,
    calibration_from_samples,
)
from erp.sensors.replay import ReplaySensor
from erp.sensors.sim import SimSensor
from erp.sensors.stream import StreamSensor

__all__ = [
    "ArrivalClock",
    "ClockSync",
    "HostClock",
    "IMUDecoder",
    "LineTransport",
    "ReplaySensor",
    "Sensor",
    "SensorError",
    "SerialIMUSensor",
    "SimSensor",
    "StreamSensor",
    "calibration_from_samples",
    "identity_calibration",
    "shared_rows_R",
]
