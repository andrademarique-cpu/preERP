"""ClockSync against a synthetic device clock, paired with a stamping that must fail."""

from __future__ import annotations

import numpy as np

from erp.sensors import ArrivalClock, ClockSync

BOUND_S = 1.0e-3  # sample-time error the sync must stay under


def _stream(n: int = 5000, rate_hz: float = 1000.0, seed: int = 0) -> tuple[np.ndarray, ...]:
    """True sample times, device micros (wrapped, drifting), and arrival times.

    Latency = 0.2 ms floor + exponential(2 ms), a USB-like heavy tail.
    """
    rng = np.random.default_rng(seed)
    t_true = 1000.123 + np.arange(n) / rate_hz  # host seconds
    drift = 20e-6  # device crystal 20 ppm fast
    ticks = (np.arange(n) / rate_hz) * (1 + drift) * 1e6 + 7.0e8
    latency = 2.0e-4 + rng.exponential(2.0e-3, size=n)
    return t_true, ticks, t_true + latency


def _errors(clock: ArrivalClock | ClockSync, ticks: np.ndarray, arrival: np.ndarray,
            t_true: np.ndarray, *, wrap: float | None = None) -> np.ndarray:
    dev = ticks % wrap if wrap is not None else ticks
    est = np.array([clock.to_host(float(d), float(a)) for d, a in zip(dev, arrival, strict=True)])
    return np.abs(est - t_true)


def test_clock_sync_recovers_sample_time() -> None:
    t_true, ticks, arrival = _stream()
    err = _errors(ClockSync(1e-6, window_s=1.0), ticks, arrival, t_true)
    settled = err[500:]  # after half a second of history
    assert settled.max() < BOUND_S, settled.max()


def test_arrival_stamping_fails_the_same_bound() -> None:
    """The falsification: without sync, USB latency alone breaks the bound."""
    t_true, ticks, arrival = _stream()
    err = _errors(ArrivalClock(0.0), ticks, arrival, t_true)[500:]
    assert err.max() > BOUND_S
    # Subtracting the *mean* latency removes the bias but not the jitter.
    err_mean = _errors(ArrivalClock(2.2e-3), ticks, arrival, t_true)[500:]
    assert err_mean.max() > BOUND_S


def test_micros_wraparound_does_not_jump_back() -> None:
    t_true, ticks, arrival = _stream()
    wrap = 2.0**32
    shifted = ticks + (wrap - ticks[2500])  # wraps exactly mid-stream
    clock = ClockSync(1e-6, window_s=1.0, wrap_ticks=wrap)
    err = _errors(clock, shifted, arrival, t_true, wrap=wrap)[500:]
    assert err.max() < BOUND_S


def test_missing_device_time_falls_back_to_arrival() -> None:
    clock = ClockSync(1e-6, fallback_latency_s=0.004)
    assert clock.to_host(None, 10.0) == 10.0 - 0.004
    assert clock.offset is None
