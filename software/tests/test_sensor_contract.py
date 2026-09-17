"""The Sensor contract, exercised identically against every implementation.

A test that passes against replay, sim and the live serial sensor is what shows
the interface -- not one implementation -- is what the loop depends on.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import pytest
from conftest import ROWS, FakePort, kv_line, wait_until

from erp.core.types import Measurement
from erp.sensors import IMUDecoder, ReplaySensor, Sensor, SerialIMUSensor, SimSensor

N = 30


def _replay(decoder: IMUDecoder, R: np.ndarray) -> Iterator[Sensor]:
    t = np.linspace(0.0, 1.0, N)
    yield ReplaySensor.from_arrays(t, np.zeros((N, 12)), rows=ROWS, R=R)


def _sim(decoder: IMUDecoder, R: np.ndarray) -> Iterator[Sensor]:
    t = np.arange(0, 3.0, 0.002)
    yield SimSensor(t, np.zeros((t.size, 15)), rows=ROWS, R=R, rate_hz=10.0)


def _serial(decoder: IMUDecoder, R: np.ndarray) -> Iterator[Sensor]:
    lines = [kv_line(np.full(12, i)) for i in range(N)]
    sensor = SerialIMUSensor(None, decoder, ROWS, R, transport=FakePort(lines))
    with sensor:
        assert wait_until(lambda: sensor.received == N)
        yield sensor


FACTORIES: dict[str, Callable[[IMUDecoder, np.ndarray], Iterator[Sensor]]] = {
    "replay": _replay,
    "sim": _sim,
    "serial": _serial,
}


@pytest.fixture(params=sorted(FACTORIES))
def sensor(
    request: pytest.FixtureRequest, decoder: IMUDecoder, R12: np.ndarray
) -> Iterator[Sensor]:
    yield from FACTORIES[request.param](decoder, R12)


def test_drain_is_ordered_and_then_empty(sensor: Sensor) -> None:
    first = sensor.drain()
    assert len(first) > 0
    ts = [m.timestamp for m in first]
    assert ts == sorted(ts)
    assert sensor.drain() == []
    assert sensor.read() is None


def test_samples_match_declared_rows_and_R(sensor: Sensor) -> None:
    k = sensor.rows.size
    assert sensor.R.shape == (k, k)
    for m in sensor.drain():
        assert isinstance(m, Measurement)
        assert m.z.shape == (k,)
        assert m.source == sensor.name


def test_rows_and_R_are_shared_and_read_only(sensor: Sensor) -> None:
    ms = sensor.drain()
    assert all(m.rows is ms[0].rows and m.R is ms[0].R for m in ms)
    with pytest.raises(ValueError):
        ms[0].R[0, 0] = 1.0
    with pytest.raises(ValueError):
        ms[0].rows[0] = 5


def test_read_returns_samples_one_at_a_time(sensor: Sensor) -> None:
    m = sensor.read()
    assert m is not None
    rest = sensor.drain()
    assert all(r.timestamp >= m.timestamp for r in rest)


def test_calibrate_returns_a_result_of_the_right_size(sensor: Sensor) -> None:
    if isinstance(sensor, SerialIMUSensor):
        pytest.skip("live calibration is time-based; covered in test_imu_serial")
    res = sensor.calibrate()
    assert res.valid
    assert res.bias.shape == (sensor.rows.size,)
