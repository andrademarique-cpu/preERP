"""LiveSimSensor: what the shared Sensor contract does not reach.

`test_sensor_contract.py` asserts what every sensor owes the loop. What is
specific to this one is *ingestion*: when a poll becomes a sample, and when a
sample becomes visible. Both are timing, and both fail quietly -- a drifting
sample grid and a sample released too early produce a run that still plots.

No mujoco and no display: the sensor takes arrays, so the whole file is in the
fast suite.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import ROWS

from erp.core.clock import VirtualClock
from erp.sensors import LiveSimSensor

NSENSORDATA = 15
DT = 0.002          # the model's physics step
RATE = 20.0         # the IMU's rate
STEPS = 1500        # 3 s of polling, the length of the golden trajectory


def _sensor(**kw: object) -> LiveSimSensor:
    R = np.eye(ROWS.size) * 0.05**2
    return LiveSimSensor(rows=ROWS, R=R, rate_hz=RATE, **kw)  # type: ignore[arg-type]


def _poll_a_run(sensor: LiveSimSensor, steps: int = STEPS) -> list[float]:
    """Poll at the physics step for ``steps`` steps; return the sample instants."""
    for i in range(steps):
        sensor.poll(i * DT, np.zeros(NSENSORDATA))
    return list(sensor.truth_t)


def test_samples_land_on_the_absolute_grid() -> None:
    """Sample k lands at k / rate, to within one physics step."""
    t = np.asarray(_poll_a_run(_sensor()))
    assert t.size == pytest.approx(STEPS * DT * RATE, abs=1)
    grid = np.arange(t.size) / RATE
    # Within one step: a sample is taken at the first poll at or after its
    # deadline, so it can only ever be late, and never by more than one tick.
    err = t - grid
    assert err.min() >= 0.0
    assert err.max() < DT + 1e-12


def _accumulator(rate: float, steps: int = STEPS) -> np.ndarray:
    """The rejected implementation: each deadline set relative to the last sample.

    Inlined rather than described, so the comparison below is against code
    rather than against a claim about code.
    """
    out, due = [], 0.0
    for i in range(steps):
        t = i * DT
        if t + 1e-12 >= due:
            out.append(t)
            due = t + 1.0 / rate        # <- the bug: relative to t, not to the grid
    return np.asarray(out)


def test_at_20_hz_the_accumulator_is_identical_and_that_is_the_trap() -> None:
    """At the configured rate the bug is invisible, which is why it is worth a test.

    1 / 20 Hz is exactly 25 physics steps of 2 ms, so there is no remainder to
    carry and the two implementations agree bit for bit. The same holds at 25
    and 50 Hz. Anyone checking the accumulator at the project's own rates would
    conclude it was fine.
    """
    np.testing.assert_array_equal(np.asarray(_poll_a_run(_sensor())), _accumulator(RATE))


@pytest.mark.parametrize(
    ("rate", "lost", "lag_s"),
    [(30.0, 1, 0.058), (33.0, 5, 0.156)],
)
def test_the_accumulator_drifts_when_the_period_is_not_a_whole_step(
    rate: float, lost: int, lag_s: float
) -> None:
    """The falsification, at a rate where the remainder actually exists.

    1 / 30 Hz is 16.667 steps, so each deadline rounds up to 17 and the extra
    0.667 of a step is carried forward. Measured over 3 s: 30 Hz emits 89
    samples where the grid emits 90, and its last common sample sits 58 ms
    late; 33 Hz loses 5 and runs 156 ms late. Stated precisely because the
    honest size of this matters -- it is a sensor that reads slightly slow,
    not one that stops, so downstream it looks like the IMU, not the clock.
    """
    sensor = LiveSimSensor(rows=ROWS, R=np.eye(ROWS.size) * 0.05**2, rate_hz=rate)
    t_grid = np.asarray(_poll_a_run(sensor))
    t_acc = _accumulator(rate)

    assert t_grid.size - t_acc.size == lost
    n = t_acc.size
    assert float(t_acc[n - 1] - t_grid[n - 1]) == pytest.approx(lag_s, abs=0.002)


def test_latency_withholds_a_sample_until_the_clock_passes_it() -> None:
    """A sample taken at t is invisible until clock() >= t + latency_s."""
    clock = VirtualClock()
    sensor = _sensor(latency_s=0.005, clock=clock.now)

    sensor.poll(0.0, np.zeros(NSENSORDATA))
    assert sensor.sampled == 1
    assert sensor.drain() == [], "released before its latency had elapsed"

    clock.sleep_until(0.0049)
    assert sensor.drain() == []
    clock.sleep_until(0.005)
    assert len(sensor.drain()) == 1


def test_without_a_clock_everything_is_released_as_it_is_sampled() -> None:
    sensor = _sensor()
    sensor.poll(0.0, np.zeros(NSENSORDATA))
    sensor.poll(1.0, np.zeros(NSENSORDATA))
    assert len(sensor.drain()) == 2


def test_noise_off_emits_the_plant_reading_unchanged() -> None:
    reading = np.arange(NSENSORDATA, dtype=float)
    sensor = _sensor(noise=False)
    m = sensor.poll(0.0, reading)
    assert m is not None
    np.testing.assert_array_equal(m.z, reading[ROWS])


def test_noise_is_drawn_from_the_declared_R() -> None:
    """The residual's sample covariance matches R -- this is a plumbing check.

    Which is exactly the uncomfortable property the demo has to state out loud:
    the filter is handed noise from the same R it is given.
    """
    sigma = 0.05
    R = np.eye(ROWS.size) * sigma**2
    sensor = LiveSimSensor(rows=ROWS, R=R, rate_hz=RATE, seed=0)
    for i in range(20_000):
        sensor.poll(i / RATE, np.zeros(NSENSORDATA))
    Z = np.asarray([m.z for m in sensor.drain()])
    assert Z.std(axis=0) == pytest.approx(sigma, rel=0.05)


def test_truth_is_recorded_beside_every_sample() -> None:
    reading = np.arange(NSENSORDATA, dtype=float)
    sensor = _sensor()
    sensor.poll(0.0, reading)
    assert len(sensor.truth_z) == 1
    np.testing.assert_array_equal(sensor.truth_z[0], reading[ROWS])
    assert sensor.truth_t == [0.0]


def test_rows_past_the_end_of_sensordata_raise() -> None:
    """A short sensordata is a model/layout mismatch, not a sample to skip."""
    sensor = LiveSimSensor(rows=[0, 1, 99], R=np.eye(3), rate_hz=RATE)
    with pytest.raises(ValueError, match="rows reach index"):
        sensor.poll(0.0, np.zeros(NSENSORDATA))


@pytest.mark.parametrize(("kw", "match"), [({"rate_hz": 0.0}, "rate_hz"),
                                           ({"latency_s": -1.0}, "latency_s")])
def test_invalid_construction_is_refused(kw: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        LiveSimSensor(rows=ROWS, R=np.eye(ROWS.size), **kw)  # type: ignore[arg-type]
