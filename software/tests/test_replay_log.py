"""MeasurementLog, ReplaySensor loaders and pacing, SimSensor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import IMU_KEYS, ROWS

from erp.core.types import Measurement
from erp.io import MeasurementLog
from erp.sensors import IMUDecoder, ReplaySensor, SimSensor


def _ms(t: np.ndarray, Z: np.ndarray, R: np.ndarray, source: str = "imu") -> list[Measurement]:
    rows = np.asarray(ROWS)
    return [Measurement(Z[i], float(t[i]), rows, R, source) for i in range(t.size)]


def test_log_csv_round_trips_through_replay(tmp_path: Path, R12: np.ndarray,
                                            decoder: IMUDecoder) -> None:
    rng = np.random.default_rng(0)
    t0 = 5000.25  # absolute perf_counter-like base
    t = t0 + np.arange(40) * 0.05
    Z = rng.normal(size=(40, 12))
    log = MeasurementLog()
    log.extend(_ms(t, Z, R12))
    path = log.save_csv(tmp_path / "imu.csv", "imu", t_ref=t0, column_names=decoder.column_names())

    header = path.read_text().splitlines()[0]
    assert header.startswith("timestamp,link1_acc_x")
    rep = ReplaySensor.from_log_csv(path, rows=ROWS, R=R12, name="imu", t_shift=t0)
    got = rep.drain()
    np.testing.assert_allclose([m.timestamp for m in got], t, atol=1e-6)
    np.testing.assert_allclose(np.stack([m.z for m in got]), Z, rtol=1e-8)


def test_raw_log_is_re_decoded_with_the_current_configuration(
    tmp_path: Path, R12: np.ndarray
) -> None:
    """Save raw with one config, fix the config, reload: the fix applies to old data."""
    raws = np.random.default_rng(2).normal(size=(20, 12))
    t = 100.0 + np.arange(20) * 0.05
    wrong = IMUDecoder(IMU_KEYS, {
        "link1_acc": ("IMU_1.ax", "IMU_1.ay", "IMU_1.az"),
        "link2_acc": ("IMU_0.ax", "IMU_0.ay", "IMU_0.az"),
        "link1_gyro": ("IMU_1.wx", "IMU_1.wy", "IMU_1.wz"),
        "link2_gyro": ("IMU_0.wx", "IMU_0.wy", "IMU_0.wz"),
    })
    log = MeasurementLog()
    log.extend(_ms(t, wrong.apply(raws), R12))
    path = log.save_csv(tmp_path / "raw.csv", "imu", t_ref=100.0, column_names=IMU_KEYS,
                        transform=wrong.to_raw)

    rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    fixed = IMUDecoder(IMU_KEYS, {
        "link1_acc": ("IMU_0.ax", "IMU_0.ay", "IMU_0.az"),
        "link2_acc": ("IMU_1.ax", "IMU_1.ay", "IMU_1.az"),
        "link1_gyro": ("IMU_0.wx", "IMU_0.wy", "IMU_0.wz"),
        "link2_gyro": ("IMU_1.wx", "IMU_1.wy", "IMU_1.wz"),
    }, axis_maps={"link1_acc": rot, "link1_gyro": rot})
    rep = ReplaySensor.from_legacy_imu_csv(path, fixed, rows=ROWS, R=R12)
    got = np.stack([m.z for m in rep.drain()])
    np.testing.assert_allclose(got, fixed.apply(raws), atol=1e-7)


def test_legacy_csv_decodes_by_column_name(tmp_path: Path, decoder: IMUDecoder,
                                           R12: np.ndarray) -> None:
    raw = np.random.default_rng(1).normal(size=(10, 12))
    t = np.arange(10) * 0.05
    order = list(reversed(range(12)))  # columns stored in a different order
    path = tmp_path / "legacy.csv"
    np.savetxt(path, np.column_stack([t, raw[:, order]]), delimiter=",", comments="",
               header=",".join(["timestamp", *(IMU_KEYS[i] for i in order)]))
    rep = ReplaySensor.from_legacy_imu_csv(path, decoder, rows=ROWS, R=R12)
    got = np.stack([m.z for m in rep.drain()])
    np.testing.assert_allclose(got, decoder.apply(raw), rtol=1e-12)


def test_window_and_ordering_across_drains(R12: np.ndarray) -> None:
    Z = np.zeros((6, 12))
    log = MeasurementLog()
    log.extend(_ms(np.array([3.0, 4.0, 5.0]), Z[:3], R12))
    log.extend(_ms(np.array([0.0, 1.0, 2.0]), Z[3:], R12))  # an earlier batch, drained later
    t, _, rows = log.to_arrays("imu")
    np.testing.assert_array_equal(t, [0, 1, 2, 3, 4, 5])
    w = log.window(1.0, 4.0)
    np.testing.assert_array_equal(w.to_arrays("imu")[0], [1, 2, 3, 4])
    assert rows is log.to_arrays("imu")[2]
    with pytest.raises(KeyError):
        log.to_arrays("nope")


def test_paced_replay_releases_nothing_early(R12: np.ndarray) -> None:
    now = [0.0]
    rep = ReplaySensor.from_arrays(
        [1.0, 2.0, 3.0], np.zeros((3, 12)), rows=ROWS, R=R12,
        clock=lambda: now[0], latency_s=0.5,
    )
    assert rep.drain() == []
    now[0] = 1.49
    assert rep.read() is None
    now[0] = 2.5
    assert [m.timestamp for m in rep.drain()] == [1.0, 2.0]
    now[0] = 100.0
    assert [m.timestamp for m in rep.drain()] == [3.0]
    assert rep.remaining == 0


def test_replay_rejects_out_of_order_record(R12: np.ndarray) -> None:
    ms = _ms(np.array([1.0, 0.5]), np.zeros((2, 12)), R12)
    with pytest.raises(ValueError, match="non-decreasing"):
        ReplaySensor(ms, rows=ROWS, R=R12)


def test_sim_sensor_rate_noise_and_truth() -> None:
    t = np.arange(0.0, 3.0, 0.002)  # 500 Hz physics log
    S = np.tile(np.linspace(0.0, 1.0, 15), (t.size, 1))
    rows = np.array([0, 3, 14])
    R = np.diag([0.01, 0.04, 0.0])  # zero-variance channel must still work

    decimated = SimSensor(t, S, rows=rows, R=R, rate_hz=20.0, noise=False, t_shift=100.0)
    ms = decimated.drain()
    assert len(ms) == 60
    np.testing.assert_allclose(np.diff([m.timestamp for m in ms]), 0.05, atol=2e-3)
    assert ms[0].timestamp == pytest.approx(100.0)
    np.testing.assert_array_equal(np.stack([m.z for m in ms]), S[:60][:, rows])

    noisy = SimSensor(t, S, rows=rows, R=R, seed=3)
    Z = np.stack([m.z for m in noisy.drain()])
    std = (Z - noisy.truth_z).std(axis=0)
    np.testing.assert_allclose(std[:2], [0.1, 0.2], rtol=0.06)
    assert std[2] == 0.0
