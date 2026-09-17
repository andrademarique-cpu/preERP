"""IMUs streamed as text lines over USB serial (Teensy 3.6 on the palletizer).

Two pieces, split so that live and recorded data go through the same code:

- :class:`IMUDecoder` is pure: bytes -> raw device vector -> calibrated vector
  in MuJoCo sensordata layout. :class:`~erp.sensors.replay.ReplaySensor` uses
  it on recorded CSVs, so a replayed sample is bit-identical to what the live
  path would have produced from the same line.
- :class:`SerialIMUSensor` owns the port, the reader thread and the clock.

Performance note, measured on the palletizer model: one 12-channel EKF update
costs ~360 us, four separate 3-channel updates ~1.4 ms, while parsing a line
costs 3-10 us. That is why one line becomes exactly **one** Measurement with
every channel, and why parsing is kept simple rather than clever.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from erp.core.types import Array, CalibrationResult, IntArray, Measurement
from erp.sensors.base import SensorError
from erp.sensors.clock import ArrivalClock, HostClock
from erp.sensors.stream import StreamSensor

__all__ = ["IMUDecoder", "LineTransport", "SerialIMUSensor", "calibration_from_samples"]

TIMESTAMP_KEYS: tuple[str, ...] = ("t", "ts", "timestamp", "time", "millis", "micros")

# key:value with optional exponent. Superset of the notebook's pattern, on bytes
# so no per-line decode is needed.
_KV = re.compile(rb"([A-Za-z0-9_.]+):([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")
_NUMERIC_START = frozenset(b"0123456789+-.")


class IMUDecoder:
    """Turn device lines into calibrated vectors in MuJoCo sensordata layout.

    The device sends channels named by ``keys`` (e.g. ``IMU_0.ax``). The
    ``layout`` says which of those channels feed which MuJoCo sensor, in the
    order the caller's ``rows`` list them -- this is where the physical
    wiring (IMU_1 on link1, IMU_0 on link2) is recorded, instead of in code.

    Calibrated output: ``z = A @ raw[idx] - b`` where ``A`` is block-diagonal,
    one ``axis_maps`` matrix per layout block, mapping device axes onto the
    MuJoCo *site* frame. The accelerometer and gyro of one chip share the same
    physical axes, so give both blocks the same matrix.
    """

    def __init__(
        self,
        keys: Sequence[str],
        layout: Mapping[str, Sequence[str]],
        axis_maps: Mapping[str, npt.ArrayLike] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        keys:
            Device channel names in the order the device sends them (and the
            order of columns in a raw CSV log).
        layout:
            MuJoCo sensor name -> the device keys of its channels, in axis
            order. Iteration order must match ``rows_of(model, *layout)``.
        axis_maps:
            Optional MuJoCo sensor name -> (d, d) matrix taking device axes to
            site axes (typically a signed permutation). Missing -> identity.
        """
        self.keys: tuple[str, ...] = tuple(keys)
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("keys contains duplicates")
        position = {k: i for i, k in enumerate(self.keys)}
        axis_maps = dict(axis_maps or {})
        unknown_maps = set(axis_maps) - set(layout)
        if unknown_maps:
            raise ValueError(f"axis_maps for sensors not in layout: {sorted(unknown_maps)}")

        idx: list[int] = []
        blocks: list[tuple[str, int]] = []
        for sensor, dev_keys in layout.items():
            missing = [k for k in dev_keys if k not in position]
            if missing:
                raise ValueError(f"layout[{sensor!r}] uses keys not in keys: {missing}")
            idx.extend(position[k] for k in dev_keys)
            blocks.append((sensor, len(dev_keys)))

        dim = len(idx)
        A = np.zeros((dim, dim))
        start = 0
        for sensor, d in blocks:
            M = np.asarray(axis_maps.get(sensor, np.eye(d)), dtype=np.float64)
            if M.shape != (d, d):
                raise ValueError(f"axis_maps[{sensor!r}] must be ({d}, {d}), got {M.shape}")
            A[start : start + d, start : start + d] = M
            start += d

        self.blocks: tuple[tuple[str, int], ...] = tuple(blocks)
        self.dim = dim
        self._idx: IntArray = np.asarray(idx, dtype=np.intp)
        self._At: Array = np.ascontiguousarray(A.T)
        self._keys_b = tuple(k.encode() for k in self.keys)
        self._ts_keys_b = tuple(k.encode() for k in TIMESTAMP_KEYS)
        self.b: Array = np.zeros(dim)
        """(dim,) bias subtracted after the axis map, calibrated units. Replaced
        (never written in place) by :meth:`SerialIMUSensor.apply_calibration`,
        so a reader thread mid-``apply`` sees either the old or the new one."""

    @property
    def A(self) -> Array:
        """(dim, dim) block-diagonal device-to-site axis map."""
        return np.ascontiguousarray(self._At.T)

    def column_names(self) -> list[str]:
        """``link1_acc_x``-style names for the calibrated columns."""
        names: list[str] = []
        for sensor, d in self.blocks:
            suffixes = "xyz" if d == 3 else [str(i) for i in range(d)]
            names.extend(f"{sensor}_{s}" for s in suffixes)
        return names

    def indices_of(self, sensors: Iterable[str]) -> IntArray:
        """Positions in ``z`` belonging to the named layout blocks."""
        wanted = set(sensors)
        out: list[int] = []
        start = 0
        for sensor, d in self.blocks:
            if sensor in wanted:
                out.extend(range(start, start + d))
            start += d
        return np.asarray(out, dtype=np.intp)

    def apply(self, raw: npt.ArrayLike) -> Array:
        """Raw device vector(s) in ``keys`` order -> calibrated ``z``.

        Accepts (len(keys),) or (N, len(keys)); vectorised for replay.
        """
        r = np.asarray(raw, dtype=np.float64)
        return np.asarray(r[..., self._idx] @ self._At - self.b, dtype=np.float64)

    def to_raw(self, z: npt.ArrayLike) -> Array:
        """Calibrated ``z`` -> raw device vector(s) in ``keys`` order; inverse of :meth:`apply`.

        Accepts (dim,) or (N, dim). Device keys the layout does not use come back
        as NaN. This is what logs should store: a raw log can be decoded again
        after the layout, axis maps or bias are corrected, while a log of ``z``
        has the configuration of the day it was recorded baked into it.
        """
        zz = np.asarray(z, dtype=np.float64)
        # z = raw[idx] @ A^T - b  =>  raw[idx] = (z + b) @ (A^T)^-1
        mapped = (zz + self.b) @ np.linalg.inv(self._At)
        out = np.full((*zz.shape[:-1], len(self.keys)), np.nan)
        out[..., self._idx] = mapped
        return out

    def parse_kv(self, line: bytes) -> tuple[Array, float | None] | None:
        """Parse ``IMU_0.ax:-9.71 IMU_0.ay:... [micros:123]`` lines.

        Returns ``None`` for text that is not a data line at all (boot banners,
        debug prints). Raises ``ValueError`` for a line that *is* data but is
        incomplete -- the caller counts those as rejected.
        """
        pairs = _KV.findall(line)
        if not pairs:
            return None
        values = dict(pairs)
        try:
            raw = np.array([float(values[k]) for k in self._keys_b], dtype=np.float64)
        except KeyError as exc:
            raise ValueError(f"data line missing key {exc}") from None
        t_dev: float | None = None
        for k in self._ts_keys_b:
            v = values.get(k)
            if v is not None:
                t_dev = float(v)
                break
        return raw, t_dev

    def parse_csv(self, line: bytes, *, has_time: bool = True) -> tuple[Array, float | None] | None:
        """Parse fixed-order ``t_dev,v0,...,vN`` lines (values in ``keys`` order).

        The recommended firmware format: half the bytes of key:value and no
        regex. Returns ``None`` for non-numeric text, raises ``ValueError`` on a
        wrong column count or a bad number.
        """
        if not line or line[0] not in _NUMERIC_START:
            return None
        vals = np.array(line.split(b","), dtype=np.float64)
        expected = len(self.keys) + (1 if has_time else 0)
        if vals.size != expected:
            raise ValueError(f"expected {expected} columns, got {vals.size}")
        if has_time:
            return vals[1:], float(vals[0])
        return vals, None


class LineTransport(Protocol):
    """What :class:`SerialIMUSensor` needs from a port. ``serial.Serial`` fits.

    ``readline`` must return ``b""`` on timeout rather than block forever, or
    :meth:`SerialIMUSensor.stop` can hang.
    """

    def readline(self) -> bytes: ...

    def close(self) -> None: ...


def calibration_from_samples(
    Z: npt.ArrayLike,
    current_bias: npt.ArrayLike,
    *,
    gyro_idx: npt.ArrayLike,
    acc_idx: npt.ArrayLike,
    expected_rest: npt.ArrayLike | None = None,
    still_gyro_std: float = 0.02,
    min_samples: int = 20,
) -> CalibrationResult:
    """Static calibration from samples taken while the arm is not moving.

    Parameters
    ----------
    Z:
        (N, dim) calibrated samples, produced with ``current_bias`` applied.
    current_bias:
        (dim,) bias in effect while ``Z`` was recorded.
    gyro_idx, acc_idx:
        Positions of gyro (rad/s) and accelerometer (m/s^2) channels in ``z``.
    expected_rest:
        (dim,) what MuJoCo's sensors read at the rest pose, e.g. ``h(x_home)``
        at the rows. Required to estimate accelerometer bias: at rest an
        accelerometer reads gravity, not zero. Without it only the gyro bias
        (expected 0 rad/s at rest) is estimated.
    still_gyro_std:
        rad/s. A gyro channel noisier than this means the arm moved, and the
        result is marked invalid instead of baking motion into the bias.
    min_samples:
        Fewer samples than this -> invalid.

    Returns
    -------
    A :class:`CalibrationResult` whose ``bias`` is the *total* bias to use
    (current + correction) and whose ``R`` is the sample covariance.
    """
    Z_a = np.atleast_2d(np.asarray(Z, dtype=np.float64))
    b0 = np.asarray(current_bias, dtype=np.float64)
    dim = b0.size
    g = np.asarray(gyro_idx, dtype=np.intp)
    a = np.asarray(acc_idx, dtype=np.intp)
    n = Z_a.shape[0] if Z_a.size else 0

    if n < min_samples:
        return CalibrationResult(
            bias=b0.copy(), scale=np.ones(dim), R=np.zeros((dim, dim)), valid=False,
            note=f"only {n} samples, need {min_samples}",
        )

    mean = Z_a.mean(axis=0)
    R = np.atleast_2d(np.cov(Z_a, rowvar=False)) + 1e-12 * np.eye(dim)
    delta = np.zeros(dim)
    delta[g] = mean[g]
    notes = [f"{n} samples"]
    if expected_rest is not None:
        exp = np.asarray(expected_rest, dtype=np.float64)
        delta[a] = mean[a] - exp[a]
    else:
        notes.append("accelerometer bias not estimated (no expected_rest)")

    gyro_std = float(Z_a[:, g].std(axis=0).max()) if g.size else 0.0
    valid = gyro_std <= still_gyro_std
    if not valid:
        notes.append(f"gyro std {gyro_std:.4f} rad/s > {still_gyro_std} -- arm moved?")
    return CalibrationResult(
        bias=b0 + delta, scale=np.ones(dim), R=R, valid=valid, note="; ".join(notes)
    )


class SerialIMUSensor(StreamSensor):
    """Live IMU stream from a USB serial device (Teensy 3.6).

    One line -> one :class:`~erp.core.types.Measurement` holding every channel
    of both IMUs, stamped by ``clock`` at the sample instant.

    The Teensy's USB serial ignores ``baudrate`` (it is always full-speed USB),
    so line length does not cap the sample rate; the firmware loop does.
    """

    def __init__(
        self,
        port: str | None,
        decoder: IMUDecoder,
        rows: npt.ArrayLike,
        R: npt.ArrayLike,
        *,
        fmt: Literal["kv", "csv"] = "kv",
        clock: HostClock | None = None,
        name: str = "imu",
        baudrate: int = 115200,
        read_timeout_s: float = 0.1,
        buffer_len: int = 4096,
        transport: LineTransport | None = None,
        on_text: Callable[[str], None] | None = None,
        gyro_sensors: Iterable[str] | None = None,
        acc_sensors: Iterable[str] | None = None,
        expected_rest: npt.ArrayLike | None = None,
        calib_duration_s: float = 2.0,
        still_gyro_std: float = 0.02,
    ) -> None:
        """
        Parameters
        ----------
        port:
            e.g. ``"COM7"``. Ignored when ``transport`` is given.
        decoder:
            Channel layout and calibration; ``decoder.dim`` must equal ``len(rows)``.
        rows, R:
            sensordata indices of ``decoder``'s layout, and their noise block
            (m/s^2 squared for accel, (rad/s)^2 for gyro).
        fmt:
            ``"kv"`` for ``key:value`` lines (current firmware), ``"csv"`` for
            ``t_dev,v0..vN`` lines.
        clock:
            Device-to-host time mapping. Default ``ArrivalClock(0.0)``, which is
            what the firmware without timestamps allows; use
            ``ClockSync(1e-6)`` once it sends ``micros()``.
        baudrate:
            Passed to pyserial; irrelevant for Teensy USB serial.
        read_timeout_s:
            ``readline`` timeout, seconds. Bounds how long :meth:`stop` waits.
        transport:
            Pre-opened object with ``readline()``/``close()`` (tests, other links).
        on_text:
            Called with non-data lines (boot banners, debug prints).
        gyro_sensors, acc_sensors:
            Layout names that are gyros / accelerometers, for :meth:`calibrate`.
            Default: names containing ``"gyro"`` / ``"acc"``.
        expected_rest:
            (k,) MuJoCo sensor reading at the rest pose, for accelerometer bias.
        calib_duration_s:
            Seconds of still data :meth:`calibrate` collects.
        still_gyro_std:
            rad/s threshold above which calibration is rejected as "moved".
        """
        super().__init__(name, rows, R, buffer_len=buffer_len)
        if decoder.dim != self.rows.size:
            raise ValueError(f"decoder has {decoder.dim} channels but rows has {self.rows.size}")
        if fmt not in ("kv", "csv"):
            raise ValueError(f"fmt must be 'kv' or 'csv', got {fmt!r}")
        self.decoder = decoder
        self.port = port
        self.fmt = fmt
        self.clock: HostClock = clock if clock is not None else ArrivalClock(0.0)
        self.baudrate = baudrate
        self.read_timeout_s = read_timeout_s
        self.on_text = on_text
        self._injected = transport
        self._transport: LineTransport | None = transport
        names = [s for s, _ in decoder.blocks]
        gyro = list(gyro_sensors) if gyro_sensors is not None else [s for s in names if "gyro" in s]
        acc = list(acc_sensors) if acc_sensors is not None else [s for s in names if "acc" in s]
        self._gyro_idx = decoder.indices_of(gyro)
        self._acc_idx = decoder.indices_of(acc)
        self.expected_rest = None if expected_rest is None else np.asarray(expected_rest, float)
        self.calib_duration_s = calib_duration_s
        self.still_gyro_std = still_gyro_std
        self._parse = decoder.parse_kv if fmt == "kv" else decoder.parse_csv

    def _open(self) -> None:
        if self._injected is not None:
            self._transport = self._injected
            return
        if self.port is None:
            raise ValueError("port is required when no transport is given")
        import serial  # lazy: erp.sensors must import without pyserial

        port = serial.Serial(self.port, baudrate=self.baudrate, timeout=self.read_timeout_s)
        port.reset_input_buffer()
        self._transport = port

    def _close(self) -> None:
        t = self._transport
        if t is not None:
            try:
                t.close()
            except Exception:
                pass

    def _poll(self) -> Measurement | None:
        transport = self._transport
        if transport is None:
            raise SensorError(f"{self.name}: not started")
        line = transport.readline()
        t_arrival = time.perf_counter()  # first thing after the bytes land
        if not line:
            return None
        line = line.strip()
        if not line:
            return None
        try:
            parsed = self._parse(line)
        except ValueError:
            self.rejected += 1
            return None
        if parsed is None:
            if self.on_text is not None:
                self.on_text(line.decode("utf-8", errors="replace"))
            return None
        raw, t_dev = parsed
        return Measurement(
            self.decoder.apply(raw),
            self.clock.to_host(t_dev, t_arrival),
            self._rows,
            self._R,
            self.name,
        )

    def calibrate(self) -> CalibrationResult:
        """Collect ``calib_duration_s`` of data with the arm still; do not apply.

        The sensor must be started. Consumes (drains) the samples it uses, so
        call it before the trajectory loop, not during. Check ``.valid`` and
        then call :meth:`apply_calibration`.
        """
        if not self.running:
            raise SensorError(f"{self.name}: start() the sensor before calibrate()")
        self.drain()
        bias_in_effect = self.decoder.b
        samples: list[Array] = []
        deadline = time.perf_counter() + self.calib_duration_s
        while time.perf_counter() < deadline:
            samples.extend(m.z for m in self.drain())
            time.sleep(0.01)
        samples.extend(m.z for m in self.drain())
        Z = np.stack(samples) if samples else np.empty((0, self.decoder.dim))
        return calibration_from_samples(
            Z,
            bias_in_effect,
            gyro_idx=self._gyro_idx,
            acc_idx=self._acc_idx,
            expected_rest=self.expected_rest,
            still_gyro_std=self.still_gyro_std,
        )

    def apply_calibration(self, result: CalibrationResult) -> None:
        """Use ``result``'s bias and ``R`` for every sample from now on."""
        if not result.valid:
            raise ValueError(f"refusing invalid calibration: {result.note}")
        bias = np.asarray(result.bias, dtype=np.float64)
        if bias.shape != (self.decoder.dim,):
            raise ValueError(f"bias must be ({self.decoder.dim},), got {bias.shape}")
        self.decoder.b = bias.copy()
        self.set_R(result.R)
