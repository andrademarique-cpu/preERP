"""``config/estimation.yaml`` into typed, frozen dataclasses.

Landed at ADR-0002 P7. Until then the YAML was documentation that could not
drift-check itself: no loader read the file, and the IMU wiring it describes
was written out **four** times -- here, in ``scripts/make_golden_run.py``, in
``software/tests/conftest.py`` and in :class:`~erp.sensors.IMUDecoder`'s own
docstring -- with the last two disagreeing with the first two about which chip
sits on which link (ADR-0002 3.3, whose table lists only three). The config is
the copy carrying the fitted residuals (gyro rms 0.06-0.10 rad/s against 0.34
for the swapped assignment) and the copy the golden fixture was generated from,
so the config is the one that wins. This module is what turns that from a
statement into something a test can hold to.

Why the values are checked here rather than trusted
---------------------------------------------------
A config file is an untyped boundary in exactly the way the MuJoCo API is:
``yaml.safe_load`` returns ``Any``, and a wrong value in it yields a filter
that *runs* and quietly answers worse, rather than one that raises. So every
value read below is coerced to a concrete type by one of the ``_as_*`` helpers,
which raise :class:`ConfigError` naming the full key path on failure. Nothing
typed ``Any`` reaches the dataclasses -- the same containment
:mod:`erp.sim.mujoco` applies to ``mj.*``.

Import boundary
---------------
This module imports :class:`~erp.sensors.IMUDecoder`, and :mod:`erp.io` therefore
does **not** re-export it. ``import erp.io`` has to stay a ``core``-only import,
because :mod:`erp.fusion` may legitimately use ``erp.io`` and may never reach
``erp.sensors``, not even one hop away through a package ``__init__``. That is
the indirect reach-through CLAUDE.md warns the CI grep cannot see, so
``test_config.py`` asserts it in a subprocess instead of trusting the comment.
Import this module by name: ``from erp.io.config import load_config``.

What is deliberately *not* here
-------------------------------
``build_sensor()``. ADR-0002 5.2 put it in this module, but 4.8 reserves naming
``SerialIMUSensor``, ``SimSensor``, ``DryRunArm`` or ``MyPalletizerArm`` to
``erp/runtime/session.py`` and adds a CI grep at P7.5 that would fail on it the
day it lands. :class:`~erp.sensors.IMUDecoder` is not a driver -- it neither
opens a port nor starts a thread -- so :func:`build_decoder` is free to live
here while construction of the sensor itself waits for P7.5.

Row indices are also absent, for a harder reason: ``rows_of`` needs an
``MjModel``, and ``io/`` may not import mujoco. The config fixes the *order* of
the layout blocks, which is what makes ``rows_of(model, *cfg.layout)``
contiguous; checking that it actually is belongs to a mujoco-marked test.

Units are part of each key's meaning and are repeated on every field, because
silent unit and frame mismatches are the main correctness risk in this stack.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np
import yaml

from erp.core.types import Array
from erp.io.paths import resolve_repo_path
from erp.sensors.imu_serial import IMUDecoder

__all__ = [
    "DEFAULT_IMU_KEY",
    "ConfigError",
    "EstimationConfig",
    "ImuConfig",
    "InputsConfig",
    "TimeConfig",
    "build_decoder",
    "default_config_path",
    "load_config",
]

DEFAULT_IMU_KEY = "palletizer_imu"
"""The ``sensors:`` entry :func:`load_config` reads unless told otherwise."""

_LINE_FORMATS = ("kv", "csv")
_CLOCK_KINDS = ("arrival", "sync")
_TIME_BASES = ("monotonic_host",)


class ConfigError(ValueError):
    """A malformed or internally inconsistent estimation config.

    The message always carries the full key path
    (``sensors.palletizer_imu.axis_maps.link1_acc``). A type complaint about a
    file nested five levels deep is otherwise unactionable -- the same
    reasoning behind :func:`erp.io.paths.resolve_repo_path`'s message.
    """


# --------------------------------------------------------------------------
# Coercions at the untyped boundary. Each takes Any and returns a concrete
# type or raises. `bool` is rejected wherever a number is wanted: it is an
# `int` subclass in Python, so `rate_hz: true` would otherwise load as 1.
# --------------------------------------------------------------------------


def _mapping(node: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(node, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(node).__name__}")
    return {str(k): v for k, v in node.items()}


def _require(node: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in node:
        raise ConfigError(f"{path}.{key}: missing")
    return node[key]


def _as_float(node: Mapping[str, Any], key: str, path: str) -> float:
    value = _require(node, key, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}.{key}: expected a number, got {value!r}")
    return float(value)


def _as_int(node: Mapping[str, Any], key: str, path: str) -> int:
    value = _require(node, key, path)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path}.{key}: expected an integer, got {value!r}")
    return int(value)


def _as_str(node: Mapping[str, Any], key: str, path: str) -> str:
    value = _require(node, key, path)
    if not isinstance(value, str):
        raise ConfigError(f"{path}.{key}: expected a string, got {value!r}")
    return value


def _as_choice(node: Mapping[str, Any], key: str, path: str, allowed: Sequence[str]) -> str:
    value = _as_str(node, key, path)
    if value not in allowed:
        raise ConfigError(f"{path}.{key}: expected one of {list(allowed)}, got {value!r}")
    return value


def _as_positive(node: Mapping[str, Any], key: str, path: str) -> float:
    value = _as_float(node, key, path)
    if value <= 0.0:
        raise ConfigError(f"{path}.{key}: must be > 0, got {value}")
    return value


def _as_non_negative(node: Mapping[str, Any], key: str, path: str) -> float:
    value = _as_float(node, key, path)
    if value < 0.0:
        raise ConfigError(f"{path}.{key}: must be >= 0, got {value}")
    return value


def _as_str_tuple(value: Any, path: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, list):
        raise ConfigError(f"{path}: expected a list of strings, got {value!r}")
    out: list[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise ConfigError(f"{path}[{i}]: expected a string, got {item!r}")
        out.append(item)
    return tuple(out)


def _as_axis_matrix(value: Any, dim: int, path: str) -> Array:
    """A (dim, dim) device-to-site axis map, checked for orthogonality.

    ``z_site = M @ z_chip``. The maps in use are signed permutations -- Rz(+-90
    deg) about the joint axis -- and the check is that ``M @ M.T == I``, not
    that ``M`` is a permutation, so a genuine rotation stays admissible.

    Orthogonality is enforced rather than assumed because a matrix that is
    merely *close* to one rescales the measurement: the reading still looks
    like an acceleration in m/s^2 and still passes every downstream shape
    check, it is simply wrong by a factor no ``R`` describes. That is precisely
    the silent unit error this project treats as its main correctness risk.
    """
    try:
        M = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: not a numeric matrix ({exc})") from None
    if M.shape != (dim, dim):
        raise ConfigError(f"{path}: expected a ({dim}, {dim}) matrix, got {M.shape}")
    if not np.allclose(M @ M.T, np.eye(dim), atol=1e-9):
        raise ConfigError(
            f"{path}: not orthogonal (M @ M.T != I). An axis map that is not a "
            "rotation or signed permutation silently rescales the measurement."
        )
    M.flags.writeable = False
    return M


# --------------------------------------------------------------------------
# The dataclasses
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TimeConfig:
    """The shared host time base and how far behind it the estimate runs.

    Attributes
    ----------
    base:
        Name of the time base. Only ``"monotonic_host"`` is accepted:
        :mod:`erp.core.types` fixes every timestamp as absolute
        ``time.perf_counter()`` seconds, so a config selecting anything else
        would be describing a stack that does not exist.
    buffer_horizon_s:
        Seconds :class:`~erp.fusion.FilterRunner` holds a measurement before
        releasing it, absorbing transport latency and cross-sensor reordering.
        The estimate trails real time by roughly this much. ``0.0`` is correct
        for a replayed log and for a single stream; it earns its keep when a
        second stream with a different latency appears.
    """

    base: str
    buffer_horizon_s: float


@dataclass(frozen=True, slots=True)
class InputsConfig:
    """Setpoint generation, independent of every sensor rate.

    Attributes
    ----------
    rate_hz:
        Setpoint rate, Hz. This is a *scheduling* parameter -- ADR-0002 4.7 --
        so :class:`~erp.core.clock.RateLoop` paces off its nominal period
        rather than off the trajectory's own sample axis.
    hold:
        Inter-sample behaviour of the command. ``"zoh"`` is what the hardware
        does, not an approximation of it.
    actuation_delay_s:
        Seconds between latching a command and it reaching the plant.
        Unmeasured on the real chain, and **not** the ~0.395 s the arm lags the
        MuJoCo replay by -- that figure is arm transport and queueing observed
        end to end, and is carried by :attr:`ImuConfig.arm_lag_s`.
    """

    rate_hz: float
    hold: str
    actuation_delay_s: float


@dataclass(frozen=True, slots=True, eq=False)
class ImuConfig:
    """One Teensy streaming two MPU-style IMUs, and the wiring it implies.

    ``eq=False`` because :attr:`axis_maps` holds arrays: the generated
    ``__eq__`` would compare them elementwise and then call ``bool()`` on the
    result, which raises. Compare fields explicitly when value equality is
    meant -- the convention :mod:`erp.core.types` already follows.

    Attributes
    ----------
    name:
        The ``sensors:`` key this was read from, carried so an error message
        can name it.
    type, device:
        Declared sensor class and device family, e.g. ``"SerialIMUSensor"`` and
        ``"teensy36"``. Recorded, not dispatched on: choosing a class from a
        string is ``erp/runtime/session.py``'s job at P7.5 (ADR-0002 4.8).
    port:
        Host port name, e.g. ``"COM7"``. A default, not a promise: the port a
        device enumerates on moves between boots.
    baudrate:
        Passed to pyserial and ignored by the Teensy, whose USB serial is
        always full-speed. Line length therefore does not cap the sample rate;
        the firmware loop does.
    fmt:
        ``"kv"`` for ``IMU_0.ax:-9.71 ...`` lines (current firmware), ``"csv"``
        for ``t_dev,v0..vN``. The csv form halves the bytes and drops the
        regex, and is preferred once the firmware changes.
    clock:
        ``"arrival"`` stamps on arrival minus :attr:`latency_s` -- bias
        removed, jitter kept -- which is all a firmware sending no timestamp
        allows. ``"sync"`` selects :class:`~erp.sensors.ClockSync`, valid only
        once lines carry ``micros()``.
    latency_s:
        Seconds from sample instant to host arrival. Unmeasured; 0.0 means the
        arrival instant is used unshifted, not that latency is known to be zero.
    device_scale:
        Seconds per device tick. ``1e-6`` for ``micros()``.
    clock_window_s:
        Seconds of history :class:`~erp.sensors.ClockSync` takes its
        sliding-window minimum over.
    buffer_len:
        Samples the reader thread holds between drains.
    rate_hz:
        Measured sample rate, Hz (50 +- 1 ms on the committed log). Against a
        2 ms physics step this leaves ~25 predicts per update, which is the
        current accuracy limit -- not the filter.
    keys:
        Device channel names in the order the device sends them, which is also
        the column order of a raw CSV log.
    layout:
        MuJoCo sensor name -> its device channels, in axis order. **Iteration
        order is significant**: it must match ``rows_of(model, *layout)``, i.e.
        the XML ``<sensor>`` block, or ``z`` and ``rows`` describe different
        channels. Read-only.
    axis_maps:
        MuJoCo sensor name -> (d, d) matrix with ``z_site = M @ z_chip``. The
        accelerometer and gyro of one chip share physical axes and so share a
        matrix. Fitted 2026-09-16 against a MuJoCo replay over all 48 signed
        permutations and both chip-to-link assignments, reproduced on two
        recordings. Arrays are flagged read-only.
    arm_lag_s:
        Seconds the real arm lags the MuJoCo replay, ~0.395, from arm transport
        and queueing rather than from the IMUs. Log a tail after the last
        setpoint or the motion is cut.
    sig_acc, sig_gyro:
        Nominal per-channel sigmas, m/s^2 and rad/s. **Nominal, not measured.**
        :meth:`~erp.sensors.SerialIMUSensor.calibrate` exists to replace them
        and has not been run on the arm; ADR-0002 P6 measured that a rest-window
        ``R`` comes out 3-8x tighter, because it is the noise floor at rest and
        says nothing about motion or model error, so adopting it would sharpen
        an already over-confident filter.
    still_gyro_std:
        rad/s above which a calibration window is rejected as "moved".
    """

    name: str
    type: str
    device: str
    port: str
    baudrate: int
    fmt: Literal["kv", "csv"]
    clock: Literal["arrival", "sync"]
    latency_s: float
    device_scale: float
    clock_window_s: float
    buffer_len: int
    rate_hz: float
    keys: tuple[str, ...]
    layout: Mapping[str, tuple[str, ...]]
    axis_maps: Mapping[str, Array]
    arm_lag_s: float
    sig_acc: float
    sig_gyro: float
    still_gyro_std: float

    @property
    def dim(self) -> int:
        """Number of calibrated channels, i.e. ``len(rows)`` for this sensor."""
        return sum(len(v) for v in self.layout.values())


@dataclass(frozen=True, slots=True, eq=False)
class EstimationConfig:
    """Everything ``config/estimation.yaml`` declares, checked and typed.

    Attributes
    ----------
    time, inputs, imu:
        The three sections. ``imu`` is the single IMU the arm carries; a second
        stream would make this a tuple, and the cross-section latency check in
        :func:`load_config` would become a max over it.
    source:
        Absolute path the config was read from, kept so a downstream error can
        say which file it is complaining about.
    """

    time: TimeConfig
    inputs: InputsConfig
    imu: ImuConfig
    source: Path


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def default_config_path() -> Path:
    """Absolute path to ``config/estimation.yaml`` in this checkout."""
    return resolve_repo_path("config", "estimation.yaml")


def _load_imu(node: Mapping[str, Any], name: str, path: str) -> ImuConfig:
    keys = _as_str_tuple(_require(node, "keys", path), f"{path}.keys")
    if not keys:
        raise ConfigError(f"{path}.keys: empty")
    if len(set(keys)) != len(keys):
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        raise ConfigError(f"{path}.keys: duplicated {duplicates}")

    layout_node = _mapping(_require(node, "layout", path), f"{path}.layout")
    if not layout_node:
        raise ConfigError(f"{path}.layout: empty")
    layout: dict[str, tuple[str, ...]] = {}
    for sensor, channels in layout_node.items():
        block = _as_str_tuple(channels, f"{path}.layout.{sensor}")
        if not block:
            raise ConfigError(f"{path}.layout.{sensor}: empty")
        unknown = [c for c in block if c not in keys]
        if unknown:
            raise ConfigError(f"{path}.layout.{sensor}: channels not in keys: {unknown}")
        layout[sensor] = block

    used = [c for block in layout.values() for c in block]
    if len(set(used)) != len(used):
        repeated = sorted({c for c in used if used.count(c) > 1})
        raise ConfigError(
            f"{path}.layout: channel(s) {repeated} appear in more than one block. "
            "One device channel feeds exactly one MuJoCo sensor."
        )

    axis_node = _mapping(node.get("axis_maps", {}) or {}, f"{path}.axis_maps")
    unknown_maps = sorted(set(axis_node) - set(layout))
    if unknown_maps:
        raise ConfigError(f"{path}.axis_maps: sensors not in layout: {unknown_maps}")
    axis_maps = {
        sensor: _as_axis_matrix(value, len(layout[sensor]), f"{path}.axis_maps.{sensor}")
        for sensor, value in axis_node.items()
    }

    return ImuConfig(
        name=name,
        type=_as_str(node, "type", path),
        device=_as_str(node, "device", path),
        port=_as_str(node, "port", path),
        baudrate=_as_int(node, "baudrate", path),
        fmt=cast(Literal["kv", "csv"], _as_choice(node, "format", path, _LINE_FORMATS)),
        clock=cast(Literal["arrival", "sync"], _as_choice(node, "clock", path, _CLOCK_KINDS)),
        latency_s=_as_non_negative(node, "latency_s", path),
        device_scale=_as_positive(node, "device_scale", path),
        clock_window_s=_as_positive(node, "clock_window_s", path),
        buffer_len=_as_int(node, "buffer_len", path),
        rate_hz=_as_positive(node, "rate_hz", path),
        keys=keys,
        layout=MappingProxyType(layout),
        axis_maps=MappingProxyType(axis_maps),
        arm_lag_s=_as_non_negative(node, "arm_lag_s", path),
        sig_acc=_as_positive(node, "sig_acc", path),
        sig_gyro=_as_positive(node, "sig_gyro", path),
        still_gyro_std=_as_positive(node, "still_gyro_std", path),
    )


def load_config(
    path: Path | str | None = None,
    *,
    imu_key: str = DEFAULT_IMU_KEY,
) -> EstimationConfig:
    """Read and validate ``config/estimation.yaml``.

    Parameters
    ----------
    path:
        The file to read. Defaults to :func:`default_config_path`, i.e. the
        config in this checkout, located from the module rather than from
        ``Path.cwd()`` so the answer does not depend on where a kernel started.
    imu_key:
        Which entry under ``sensors:`` to load.

    Raises
    ------
    ConfigError
        For anything malformed, with the key path in the message. Also for the
        one cross-section invariant the file states in prose: a
        ``buffer_horizon_s`` below the sensor's own ``latency_s`` would release
        each measurement before it could have arrived, so the runner would
        discard exactly the samples the horizon exists to keep.
    FileNotFoundError
        When ``path`` does not exist.
    """
    source = Path(path) if path is not None else default_config_path()
    if not source.is_file():
        raise FileNotFoundError(f"No estimation config at {source}")
    with source.open("r", encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh)
    if raw is None:
        raise ConfigError(f"{source}: file is empty")
    root = _mapping(raw, str(source))

    time_node = _mapping(_require(root, "time", "top level"), "time")
    time_cfg = TimeConfig(
        base=_as_choice(time_node, "base", "time", _TIME_BASES),
        buffer_horizon_s=_as_non_negative(time_node, "buffer_horizon_s", "time"),
    )

    inputs_node = _mapping(_require(root, "inputs", "top level"), "inputs")
    inputs_cfg = InputsConfig(
        rate_hz=_as_positive(inputs_node, "rate_hz", "inputs"),
        hold=_as_choice(inputs_node, "hold", "inputs", ("zoh",)),
        actuation_delay_s=_as_non_negative(inputs_node, "actuation_delay_s", "inputs"),
    )

    sensors_node = _mapping(_require(root, "sensors", "top level"), "sensors")
    if imu_key not in sensors_node:
        raise ConfigError(f"sensors.{imu_key}: missing (have {sorted(sensors_node)})")
    imu_cfg = _load_imu(
        _mapping(sensors_node[imu_key], f"sensors.{imu_key}"),
        name=imu_key,
        path=f"sensors.{imu_key}",
    )

    if time_cfg.buffer_horizon_s < imu_cfg.latency_s:
        raise ConfigError(
            f"time.buffer_horizon_s ({time_cfg.buffer_horizon_s} s) is below "
            f"sensors.{imu_key}.latency_s ({imu_cfg.latency_s} s): the runner would "
            "release each measurement before it could have arrived."
        )

    return EstimationConfig(time=time_cfg, inputs=inputs_cfg, imu=imu_cfg, source=source)


def build_decoder(cfg: ImuConfig) -> IMUDecoder:
    """The :class:`~erp.sensors.IMUDecoder` this config describes.

    The decoder is pure -- no port, no thread -- which is why it can be built
    here while constructing the sensor around it waits for
    ``erp/runtime/session.py`` at P7.5 (see this module's docstring).

    The bias is left at zero: it is not configuration. It comes from
    :func:`erp.calibration.rest_bias` on a recorded rest window, or from
    :meth:`~erp.sensors.SerialIMUSensor.calibrate` on a live one, and is applied
    to the decoder afterwards.
    """
    return IMUDecoder(cfg.keys, cfg.layout, cfg.axis_maps)
