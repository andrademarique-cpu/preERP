"""Shared fixtures: a fake serial port and the palletizer IMU layout."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Iterator

import numpy as np
import pytest

from erp.sensors import IMUDecoder

# Same key order the notebook's IMU_KEYS uses (and the raw CSV columns).
IMU_KEYS = [
    "IMU_0.ax", "IMU_0.ay", "IMU_0.az",
    "IMU_1.ax", "IMU_1.ay", "IMU_1.az",
    "IMU_0.wx", "IMU_0.wy", "IMU_0.wz",
    "IMU_1.wx", "IMU_1.wy", "IMU_1.wz",
]

# Physical wiring: IMU_1 sits on link1, IMU_0 on link2. Order matches the XML's
# <sensor> block, so rows_of(model, *LAYOUT) == arange(12).
LAYOUT = {
    "link1_acc": ("IMU_1.ax", "IMU_1.ay", "IMU_1.az"),
    "link2_acc": ("IMU_0.ax", "IMU_0.ay", "IMU_0.az"),
    "link1_gyro": ("IMU_1.wx", "IMU_1.wy", "IMU_1.wz"),
    "link2_gyro": ("IMU_0.wx", "IMU_0.wy", "IMU_0.wz"),
}
ROWS = np.arange(12)


def kv_line(raw: Iterable[float], t_dev: float | None = None) -> bytes:
    parts = [f"{k}:{v:.6f}" for k, v in zip(IMU_KEYS, raw, strict=True)]
    if t_dev is not None:
        parts.append(f"micros:{t_dev:.0f}")
    return (" ".join(parts) + "\r\n").encode()


def csv_line(raw: Iterable[float], t_dev: float) -> bytes:
    return (",".join([f"{t_dev:.0f}", *(f"{v:.6f}" for v in raw)]) + "\r\n").encode()


class FakePort:
    """``readline()`` over canned lines; then ``b""`` (like a pyserial timeout).

    ``fail_when_empty`` raises instead, imitating an unplugged device.
    ``generator`` produces lines forever (for time-based calibration).
    """

    def __init__(
        self,
        lines: Iterable[bytes] = (),
        *,
        fail_when_empty: bool = False,
        generator: Callable[[], bytes] | None = None,
        period_s: float = 0.0,
    ) -> None:
        self._lines: Iterator[bytes] = iter(list(lines))
        self.fail_when_empty = fail_when_empty
        self.generator = generator
        self.period_s = period_s
        self.closed = threading.Event()

    def readline(self) -> bytes:
        if self.closed.is_set():
            raise OSError("port closed")
        if self.period_s:
            time.sleep(self.period_s)
        line = next(self._lines, None)
        if line is not None:
            return line
        if self.generator is not None:
            return self.generator()
        if self.fail_when_empty:
            raise OSError("device disconnected")
        time.sleep(0.002)
        return b""

    def close(self) -> None:
        self.closed.set()


def wait_until(pred: Callable[[], bool], timeout: float = 2.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if pred():
            return True
        time.sleep(0.002)
    return pred()


@pytest.fixture
def decoder() -> IMUDecoder:
    return IMUDecoder(IMU_KEYS, LAYOUT)


@pytest.fixture
def R12() -> np.ndarray:
    return np.diag([0.05**2] * 6 + [0.005**2] * 6)
