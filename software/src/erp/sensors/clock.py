"""Device clock -> host clock conversion.

A sample's arrival time on the host is its sample time plus a latency that is
neither constant nor small: the Teensy sends over USB, the host receives in
packets, and the OS schedules the reader thread. Stamping on arrival therefore
adds jitter *and* a bias, and a filter reads both as dynamics.

When the device sends its own sample time (Teensy ``micros()``), the latency
of every sample is ``arrival - (device_time + offset)`` with one unknown offset
shared by all samples. Latency is never negative, so the sample with the
smallest ``arrival - device_time`` is the one that waited least, and that
minimum is the best available estimate of the offset. That is what
:class:`ClockSync` tracks.
"""

from __future__ import annotations

from collections import deque
from typing import Protocol

__all__ = ["ArrivalClock", "ClockSync", "HostClock"]


class HostClock(Protocol):
    """Anything that maps (device time, arrival time) to a host sample time."""

    def to_host(self, t_dev: float | None, t_arrival: float) -> float:
        """Sample instant in seconds, absolute ``time.perf_counter()`` base.

        ``t_dev`` is the raw device timestamp in device ticks, or ``None`` when
        the line carried none; ``t_arrival`` is ``perf_counter()`` read right
        after the line arrived.
        """
        ...


class ArrivalClock:
    """Stamp with arrival time minus a fixed latency estimate.

    The fallback while the firmware sends no timestamp. It cannot remove
    jitter, only a known average delay; prefer :class:`ClockSync` whenever the
    device can send ``micros()``.
    """

    def __init__(self, latency_s: float = 0.0) -> None:
        """``latency_s``: assumed sample-to-arrival delay, seconds, >= 0."""
        if latency_s < 0.0:
            raise ValueError(f"latency_s must be >= 0, got {latency_s}")
        self.latency_s = float(latency_s)

    def to_host(self, t_dev: float | None, t_arrival: float) -> float:
        return t_arrival - self.latency_s


class ClockSync:
    """Map a device clock onto the host clock via a sliding-window minimum.

    ``offset = min over the last window_s of (t_arrival - t_dev_seconds)``,
    kept with a monotonic deque so each sample costs O(1) amortised. The window
    lets the estimate follow crystal drift between the two clocks (tens of ppm,
    i.e. tens of microseconds per second); it must still be long enough to
    contain at least one low-latency sample.

    Device ticks are unwrapped before use: a 32-bit ``micros()`` wraps every
    ~71.6 min, and a wrap read as a jump back of 71 minutes would stamp every
    later sample in the past.
    """

    def __init__(
        self,
        device_scale: float = 1e-6,
        *,
        window_s: float = 2.0,
        wrap_ticks: float | None = 2.0**32,
        fallback_latency_s: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        device_scale:
            Seconds per device tick: 1e-6 for ``micros()``, 1e-3 for ``millis()``.
        window_s:
            Host seconds of history the minimum is taken over.
        wrap_ticks:
            Counter modulus in device ticks, or ``None`` if it never wraps.
        fallback_latency_s:
            Latency assumed for a line that arrives without a device timestamp.
        """
        if device_scale <= 0.0:
            raise ValueError(f"device_scale must be > 0, got {device_scale}")
        if window_s <= 0.0:
            raise ValueError(f"window_s must be > 0, got {window_s}")
        self.device_scale = float(device_scale)
        self.window_s = float(window_s)
        self.wrap_ticks = wrap_ticks
        self.fallback_latency_s = float(fallback_latency_s)
        self._window: deque[tuple[float, float]] = deque()  # (t_arrival, delta), delta increasing
        self._last_ticks: float | None = None
        self._wrap_base = 0.0

    @property
    def offset(self) -> float | None:
        """Current host-minus-device offset, seconds; ``None`` before any sample."""
        return self._window[0][1] if self._window else None

    def to_host(self, t_dev: float | None, t_arrival: float) -> float:
        if t_dev is None:
            return t_arrival - self.fallback_latency_s

        wrap, last = self.wrap_ticks, self._last_ticks
        if wrap is not None and last is not None and t_dev < last - wrap / 2:
            self._wrap_base += wrap
        self._last_ticks = t_dev
        t_dev_s = (t_dev + self._wrap_base) * self.device_scale

        delta = t_arrival - t_dev_s
        win = self._window
        while win and win[-1][1] >= delta:
            win.pop()
        win.append((t_arrival, delta))
        horizon = t_arrival - self.window_s
        while win[0][0] < horizon:
            win.popleft()
        return t_dev_s + win[0][1]
