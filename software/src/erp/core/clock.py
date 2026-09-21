"""Host time base and fixed-rate scheduling.

Two things live here, and they are the same idea seen from both ends: a
:class:`Clock` is where time is *read*, a :class:`RateLoop` is where time is
*waited on*. Keeping both in one pure module is what lets a three-second
trajectory run as a millisecond-long test.

**Not to be confused with :mod:`erp.sensors.clock`.** That module converts a
*device* timestamp into host time (``HostClock``, ``ClockSync``,
``ArrivalClock``) and answers "when was this sample taken?". This one answers
"what time is it, and when should the next thing happen?". They meet only in
that both speak absolute host seconds.

The invariant the whole design rests on (ADR-0002 4.7):

    The command period is a **scheduling parameter**. The sensor period is a
    **property of the device**. Neither is derived from the other, and no code
    may assume an integer ratio between them.

At 25 Hz commands and 20 Hz IMU those ticks coincide only every 200 ms. A loop
written as "one measurement per control tick" -- the most common way to write
this -- drops one sample in five here, and breaks differently at every rate.
Nothing in this module relates the two rates, which is the point.

ADR-0002 4.7 calls this "the ONLY module allowed to read a wall clock". That
is the target, not today's state: :mod:`erp.sensors.imu_serial` reads
``perf_counter`` for the arrival stamp and sleeps in ``calibrate``, and
:class:`~erp.robot.dry_run.DryRunArm` defaults its clock to ``perf_counter``.
The arrival stamp is arguably intrinsic to the transport boundary; the other
two are candidates for this seam. A CI grep is meant to enforce the rule from
P7.5, and it will have to name those exceptions or retire them.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from erp.core.types import Array

__all__ = ["Clock", "RateLoop", "Tick", "VirtualClock", "WallClock"]


class Clock(Protocol):
    """A monotonic host time base that can also be waited on."""

    def now(self) -> float:
        """Seconds, monotonic, arbitrary origin. The one host time base."""
        ...

    def sleep_until(self, t: float) -> None:
        """Block -- or, virtually, jump -- until :meth:`now` >= ``t``.

        Must return immediately when ``t`` is already in the past, rather than
        sleeping a negative duration or raising.
        """
        ...


class WallClock:
    """``time.perf_counter`` + ``time.sleep``. The clock M2 and M3 run on.

    ``perf_counter`` and not ``time.time``: the latter can step backwards when
    the system clock is adjusted, and a measurement stamped before its
    predecessor is indistinguishable from an out-of-order sample.
    """

    __slots__ = ()

    def now(self) -> float:
        return time.perf_counter()

    def sleep_until(self, t: float) -> None:
        remaining = float(t) - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)


class VirtualClock:
    """A counter whose :meth:`sleep_until` jumps instantly. The clock M1 runs on.

    Makes a 3 s trajectory a millisecond-long test, and makes scheduling
    deterministic: no OS sleep granularity and no jitter, so a scheduling test
    that fails under this clock has failed on logic, not on machine load.

    **It cannot drive a live threaded sensor.** A reader thread runs on real
    time and will not jump with it, so M1 is restricted to replay sources.
    Attempting M1 with a ``StreamSensor`` must raise at session construction
    rather than deadlock at runtime -- that check belongs to P7.5, and this
    docstring is the reason it exists.
    """

    __slots__ = ("_t",)

    def __init__(self, t0: float = 0.0) -> None:
        self._t = float(t0)

    def now(self) -> float:
        return self._t

    def sleep_until(self, t: float) -> None:
        """Jump forward to ``t``. Never backwards -- the clock is monotonic."""
        if float(t) > self._t:
            self._t = float(t)

    def advance(self, dt: float) -> None:
        """Move the clock on by ``dt`` seconds, simulating work taking time."""
        if dt < 0.0:
            raise ValueError("a monotonic clock cannot go backwards")
        self._t += float(dt)


@dataclass(frozen=True)
class Tick:
    """One scheduled iteration.

    ``t_target`` and ``t_actual`` are absolute clock seconds, not offsets from
    ``t0`` -- the same base :attr:`~erp.core.types.Measurement.timestamp` uses,
    so a tick and a sample can be compared without either side rebasing.
    """

    i: int
    t_target: float
    t_actual: float

    @property
    def slip(self) -> float:
        """How late this tick actually ran, in seconds. Never negative."""
        return self.t_actual - self.t_target

    @property
    def late(self) -> bool:
        """True when the deadline had already passed before we could wait."""
        return self.t_actual > self.t_target


class RateLoop:
    """Fixed-rate deadline scheduler over a :class:`Clock`.

    Deadlines are **absolute** -- ``t0 + i / rate_hz`` -- and never cumulative
    sleeps. Measured on this machine, 76 ticks at 25 Hz: a cumulative
    ``sleep(1/rate)`` loop finishes **65.4 ms** late and its error grows
    monotonically, because every call oversleeps a little (``sleep(2 ms)``
    measures 2.49 ms here, 2.86 ms in the figure viewer.ipynb recorded) and a
    cumulative loop banks that error every tick. Absolute deadlines finish
    **0.0 ms** late with a worst per-tick error of 0.7 ms: the error is
    bounded by one sleep's overshoot instead of accumulating.

    ``run_trajectory`` already does this correctly; this lifts it out of the
    notebook so it can be tested and reused.

    Reporting the achieved rate is not decoration. At 100 Hz the arm's ``send``
    plus ``get_angles`` round trip does not fit in 10 ms, the loop silently
    falls behind, and because ``MyPalletizer260`` has no ``set_fresh_mode`` the
    surplus setpoints **queue in the firmware** and the lag accumulates for the
    whole run. :attr:`missed` is what makes that visible rather than something
    you discover in the residuals afterwards.

    Parameters
    ----------
    clock:
        The time base. :class:`VirtualClock` makes the loop instant.
    rate_hz:
        Nominal ticks per second.
    count:
        Number of ticks, or ``None`` for an unbounded loop the caller breaks
        out of.
    t0:
        Absolute start. Defaults to ``clock.now()`` read when iteration
        begins, not at construction, so building the loop does not start it.
        Pass one explicitly to share an origin with a sensor.
    """

    def __init__(
        self,
        clock: Clock,
        rate_hz: float,
        *,
        count: int | None = None,
        t0: float | None = None,
    ) -> None:
        if rate_hz <= 0.0:
            raise ValueError(f"rate_hz must be > 0, got {rate_hz}")
        if count is not None and count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        self.clock = clock
        self.rate_hz = float(rate_hz)
        self.period = 1.0 / float(rate_hz)
        self.count = count
        self.t0 = None if t0 is None else float(t0)
        self._actual: list[float] = []
        self._target: list[float] = []
        self.missed = 0

    def __iter__(self) -> Iterator[Tick]:
        if self.t0 is None:
            self.t0 = self.clock.now()
        t0 = self.t0
        i = 0
        while self.count is None or i < self.count:
            target = t0 + i * self.period
            if self.clock.now() > target:
                # The deadline passed while the body was still running. Counted,
                # not corrected: silently skipping to the next slot would hide
                # exactly the overrun this counter exists to surface.
                self.missed += 1
            self.clock.sleep_until(target)
            actual = self.clock.now()
            self._target.append(target)
            self._actual.append(actual)
            yield Tick(i=i, t_target=target, t_actual=actual)
            i += 1

    def __len__(self) -> int:
        """Ticks completed so far, not ticks scheduled."""
        return len(self._actual)

    @property
    def t_actual(self) -> Array:
        """(n,) absolute times the completed ticks actually ran at."""
        return np.asarray(self._actual, dtype=np.float64)

    @property
    def t_target(self) -> Array:
        """(n,) absolute deadlines the completed ticks were aiming at."""
        return np.asarray(self._target, dtype=np.float64)

    @property
    def slip_s(self) -> Array:
        """(n,) per-tick lateness. Bounded for absolute deadlines."""
        return self.t_actual - self.t_target

    @property
    def achieved_hz(self) -> float:
        """Ticks per second actually managed. ``nan`` before the second tick.

        Computed the way ``run_trajectory`` reports it: ``(n - 1)`` intervals
        over the span, not ``n`` -- a 76-tick run covers 75 periods.
        """
        t = self.t_actual
        if t.size < 2 or t[-1] == t[0]:
            return float("nan")
        return float((t.size - 1) / (t[-1] - t[0]))

    @property
    def jitter_ms(self) -> float:
        """Standard deviation of the realised gaps, ms. ``nan`` under 2 ticks."""
        t = self.t_actual
        if t.size < 2:
            return float("nan")
        return float(np.std(np.diff(t)) * 1e3)

    @property
    def worst_gap_s(self) -> float:
        """Largest realised gap between consecutive ticks. ``nan`` under 2."""
        t = self.t_actual
        if t.size < 2:
            return float("nan")
        return float(np.max(np.diff(t)))

    def summary(self) -> str:
        """One-line report. Returned, not printed -- the caller decides."""
        return (
            f"{len(self)} ticks, nominal {self.rate_hz:.1f} Hz, "
            f"achieved {self.achieved_hz:.2f} Hz "
            f"(jitter {self.jitter_ms:.2f} ms, worst gap {self.worst_gap_s * 1e3:.1f} ms, "
            f"{self.missed} missed)"
        )
