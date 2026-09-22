"""IMUDecoder and SerialIMUSensor: decoding, wiring, loss accounting, calibration."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import IMU_KEYS, LAYOUT, ROWS, FakePort, csv_line, kv_line, wait_until

from erp.sensors import (
    ArrivalClock,
    ClockSync,
    IMUDecoder,
    SensorError,
    SerialIMUSensor,
    calibration_from_samples,
)

RAW = np.arange(1.0, 13.0)  # distinct per device channel, so wiring errors show


def test_kv_and_csv_decode_identically(decoder: IMUDecoder) -> None:
    kv = decoder.parse_kv(kv_line(RAW, t_dev=123456).strip())
    csv = decoder.parse_csv(csv_line(RAW, t_dev=123456).strip())
    assert kv is not None and csv is not None
    np.testing.assert_array_equal(kv[0], csv[0])
    assert kv[1] == csv[1] == 123456.0


def test_layout_maps_device_channels_onto_mujoco_sensors(decoder: IMUDecoder) -> None:
    """Every block of ``z`` holds the channels ``LAYOUT`` assigns to it, in order.

    Read out of ``LAYOUT`` rather than written out again: which chip sits on
    which link is declared once, in ``config/estimation.yaml`` (ADR-0002 P7),
    and a test that restates it becomes a fifth copy to keep in step. What is
    asserted here is the decoder's contract -- block order, axis order, column
    names -- which holds whichever way the arm is wired.
    """
    z = decoder.apply(RAW)
    dev = dict(zip(IMU_KEYS, RAW, strict=True))
    for i, block in enumerate(("link1_acc", "link2_acc", "link1_gyro", "link2_gyro")):
        np.testing.assert_array_equal(
            z[3 * i : 3 * i + 3], [dev[k] for k in LAYOUT[block]], err_msg=block
        )
    assert decoder.column_names()[0] == "link1_acc_x"


def test_axis_map_is_applied_per_block() -> None:
    swap_xy_flip_z = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    dec = IMUDecoder(IMU_KEYS, LAYOUT, axis_maps={"link2_acc": swap_xy_flip_z})
    z = dec.apply(RAW)
    dev = dict(zip(IMU_KEYS, RAW, strict=True))
    ax, ay, az = (dev[k] for k in LAYOUT["link2_acc"])
    np.testing.assert_array_equal(z[3:6], [ay, ax, -az])
    # Every other block passes through untouched.
    np.testing.assert_array_equal(z[0:3], [dev[k] for k in LAYOUT["link1_acc"]])


def test_to_raw_inverts_apply_with_axis_maps_and_bias() -> None:
    rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    dec = IMUDecoder(IMU_KEYS, LAYOUT, axis_maps={"link1_acc": rot, "link2_gyro": rot.T})
    dec.b = np.linspace(-0.5, 0.5, 12)
    raws = np.random.default_rng(0).normal(size=(7, 12))
    np.testing.assert_allclose(dec.to_raw(dec.apply(raws)), raws, atol=1e-12)
    np.testing.assert_allclose(dec.to_raw(dec.apply(raws[0])), raws[0], atol=1e-12)


def test_to_raw_marks_unused_device_keys_nan() -> None:
    dec = IMUDecoder(IMU_KEYS, {"link1_acc": ("IMU_0.ax", "IMU_0.ay", "IMU_0.az")})
    out = dec.to_raw(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(out[:3], [1, 2, 3])
    assert np.isnan(out[3:]).all()


def test_apply_is_vectorised_for_replay(decoder: IMUDecoder) -> None:
    raws = np.stack([RAW, RAW * 2])
    np.testing.assert_array_equal(decoder.apply(raws)[1], decoder.apply(RAW * 2))


def test_decoder_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="not in keys"):
        IMUDecoder(IMU_KEYS, {"link1_acc": ("IMU_9.ax", "IMU_1.ay", "IMU_1.az")})


def test_parse_distinguishes_text_from_broken_data(decoder: IMUDecoder) -> None:
    assert decoder.parse_kv(b"MPU6050 init OK") is None
    with pytest.raises(ValueError):
        decoder.parse_kv(b"IMU_0.ax:1.0 IMU_0.ay:2.0")  # data, but incomplete
    assert decoder.parse_csv(b"booting...") is None
    with pytest.raises(ValueError):
        decoder.parse_csv(b"1,2,3")


def test_malformed_line_is_counted_and_reader_survives(
    decoder: IMUDecoder, R12: np.ndarray
) -> None:
    texts: list[str] = []
    lines = [b"Teensy IMU logger v1", b"IMU_0.ax:1.0 IMU_0.ay:oops", kv_line(RAW)]
    port = FakePort(lines)
    with SerialIMUSensor(None, decoder, ROWS, R12, transport=port, on_text=texts.append) as s:
        assert wait_until(lambda: s.received == 1)
        assert s.rejected == 1
        assert s.running
        (m,) = s.drain()
    np.testing.assert_array_equal(m.z, decoder.apply(RAW))
    assert texts == ["Teensy IMU logger v1"]
    assert port.closed.is_set()


def test_overruns_are_counted_not_silent(decoder: IMUDecoder, R12: np.ndarray) -> None:
    lines = [kv_line(RAW + i) for i in range(12)]
    with SerialIMUSensor(None, decoder, ROWS, R12, transport=FakePort(lines), buffer_len=5) as s:
        assert wait_until(lambda: s.received == 12)
        out = s.drain()
    assert s.overruns == 7
    assert len(out) == 5
    np.testing.assert_array_equal(out[-1].z, decoder.apply(RAW + 11))


def test_dead_device_hands_out_data_then_raises(decoder: IMUDecoder, R12: np.ndarray) -> None:
    port = FakePort([kv_line(RAW)] * 3, fail_when_empty=True)
    s = SerialIMUSensor(None, decoder, ROWS, R12, transport=port)
    s.start()
    try:
        assert wait_until(lambda: not s.running)
        assert len(s.drain()) == 3
        with pytest.raises(SensorError):
            s.drain()
    finally:
        s.stop()


def test_clock_is_used_for_timestamps(decoder: IMUDecoder, R12: np.ndarray) -> None:
    lines = [kv_line(RAW, t_dev=1_000_000 + 1000 * i) for i in range(5)]
    # Lines must not arrive faster than the device sampled them (1 ms apart):
    # a burst would be physically impossible, and ClockSync rightly never
    # stamps a sample later than its arrival.
    port = FakePort(lines, period_s=0.005)
    sync = ClockSync(1e-6)
    with SerialIMUSensor(None, decoder, ROWS, R12, transport=port, clock=sync) as s:
        assert wait_until(lambda: s.received == 5)
        ts = [m.timestamp for m in s.drain()]
    assert sync.offset is not None
    # Spacing follows the device clock exactly (1 ms), not arrival jitter.
    np.testing.assert_allclose(np.diff(ts), 1e-3, atol=1e-9)


def test_rows_must_match_decoder(decoder: IMUDecoder) -> None:
    with pytest.raises(ValueError, match="rows"):
        SerialIMUSensor(None, decoder, np.arange(6), np.eye(6), transport=FakePort())


# -- calibration ----------------------------------------------------------

GYRO = np.arange(6, 12)
ACC = np.arange(0, 6)


def _still(n: int, gyro_bias: float, rng: np.random.Generator) -> np.ndarray:
    Z = rng.normal(0.0, 0.003, size=(n, 12))
    Z[:, GYRO] += gyro_bias
    Z[:, ACC] += 9.81
    return Z


def test_still_data_gives_valid_gyro_bias_and_R() -> None:
    rng = np.random.default_rng(0)
    res = calibration_from_samples(_still(400, 0.01, rng), np.zeros(12), gyro_idx=GYRO, acc_idx=ACC)
    assert res.valid, res.note
    np.testing.assert_allclose(res.bias[GYRO], 0.01, atol=1e-3)
    np.testing.assert_array_equal(res.bias[ACC], 0.0)  # no expected_rest -> untouched
    np.testing.assert_allclose(np.diag(res.R), 0.003**2, rtol=0.25)


def test_expected_rest_enables_accelerometer_bias() -> None:
    rng = np.random.default_rng(1)
    expected = np.zeros(12)
    expected[ACC] = 9.70
    res = calibration_from_samples(
        _still(400, 0.0, rng), np.zeros(12), gyro_idx=GYRO, acc_idx=ACC, expected_rest=expected
    )
    np.testing.assert_allclose(res.bias[ACC], 0.11, atol=2e-3)


def test_motion_during_calibration_is_rejected() -> None:
    rng = np.random.default_rng(2)
    Z = _still(400, 0.0, rng)
    Z[:, 6] += np.sin(np.linspace(0, 6, 400))  # link1 rotating
    res = calibration_from_samples(Z, np.zeros(12), gyro_idx=GYRO, acc_idx=ACC)
    assert not res.valid
    assert "moved" in res.note


def test_too_few_samples_is_rejected() -> None:
    res = calibration_from_samples(np.zeros((3, 12)), np.zeros(12), gyro_idx=GYRO, acc_idx=ACC)
    assert not res.valid


def test_live_calibrate_then_apply(decoder: IMUDecoder, R12: np.ndarray) -> None:
    raw = np.zeros(12)
    raw[6:12] = 0.02  # constant gyro bias on every device gyro channel
    port = FakePort(generator=lambda: kv_line(raw), period_s=0.001)
    s = SerialIMUSensor(
        None, decoder, ROWS, R12, transport=port, clock=ArrivalClock(), calib_duration_s=0.2
    )
    with s:
        assert s.wait_first(1.0)
        res = s.calibrate()
        assert res.valid, res.note
        s.apply_calibration(res)
        s.drain()
        assert wait_until(lambda: s.received > 0 and len(s._buf) > 0)
        m = s.drain()[-1]
    np.testing.assert_allclose(m.z[GYRO], 0.0, atol=1e-9)
    assert m.R is s.R


def test_invalid_calibration_is_refused(decoder: IMUDecoder, R12: np.ndarray) -> None:
    s = SerialIMUSensor(None, decoder, ROWS, R12, transport=FakePort())
    bad = calibration_from_samples(np.zeros((1, 12)), np.zeros(12), gyro_idx=GYRO, acc_idx=ACC)
    with pytest.raises(ValueError, match="invalid"):
        s.apply_calibration(bad)
