"""ADR-0002 P3.5: absolute deadlines, and the cumulative loop that drifts.

The pairing is the same one `test_clock.py` uses for `ClockSync`: the thing
that works is asserted against a bound, and the obvious wrong way to write it
is asserted to miss that bound. Here the wrong way is a cumulative
`sleep(1 / rate)` loop, which is how this is written when nobody has measured
what `time.sleep` actually costs.

Most of the file runs on a fake clock, so it is deterministic and instant. The
one `WallClock` test is marked `slow` and asserts only a loose bound -- a
timing test tight enough to be interesting is also tight enough to fail on a
loaded CI runner, and a flaky test teaches people to ignore failures.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from erp.core.clock import Clock, RateLoop, Tick, VirtualClock, WallClock

RATE = 25.0
PERIOD = 1.0 / RATE
N = 76  # the notebook's 3 s trajectory resampled to 25 Hz


class OversleepClock:
    """A virtual clock whose `sleep_until` always overshoots by a fixed amount.

    This is what makes the falsification deterministic. Real `time.sleep` on
    Windows overshoots by a variable amount -- measured on this machine,
    `sleep(2 ms)` takes 2.49 ms and `sleep(40 ms)` takes 40.33 ms -- and a test
    built on the real one measures the machine's load as much as the loop's
    logic. Fixing the overshoot isolates the question actually being asked:
    does the error accumulate, or stay bounded?
    """

    def __init__(self, overshoot_s: float = 0.000_86, t0: float = 0.0) -> None:
        self.overshoot_s = float(overshoot_s)
        self._t = float(t0)
        self.sleeps = 0

    def now(self) -> float:
        return self._t

    def sleep_until(self, t: float) -> None:
        if float(t) > self._t:
            self.sleeps += 1
            self._t = float(t) + self.overshoot_s

    def advance(self, dt: float) -> None:
        self._t += float(dt)


def cumulative_loop(clock, rate_hz: float, count: int) -> np.ndarray:
    """The wrong way, written out: sleep a period, repeat.

    No absolute deadline anywhere -- each sleep is relative to whenever the
    previous one happened to finish, so every overshoot is banked forever.
    """
    out = []
    for _ in range(count):
        clock.sleep_until(clock.now() + 1.0 / rate_hz)
        out.append(clock.now())
    return np.asarray(out)


# ------------------------------------------------------- the clocks themselves

@pytest.mark.parametrize("make", [WallClock, VirtualClock, OversleepClock])
def test_every_clock_honours_the_protocol(make) -> None:
    """Checked behaviourally, not with `isinstance`.

    `Clock` is a plain Protocol, like `HostClock` and `LineTransport` beside
    it. Making it `runtime_checkable` would let `isinstance` pass, but that
    only verifies the method *names* exist -- which is a weaker statement than
    the one worth making here: that `sleep_until` actually advances the clock
    and returns immediately on a deadline already past.
    """
    clock: Clock = make()
    t = clock.now()
    assert isinstance(t, float)
    clock.sleep_until(t + 0.01)
    assert clock.now() >= t + 0.01
    before = clock.now()
    clock.sleep_until(before - 1.0)  # already past
    assert clock.now() == pytest.approx(before, abs=0.05)


def test_virtual_clock_jumps_and_never_goes_backwards() -> None:
    c = VirtualClock(10.0)
    assert c.now() == 10.0
    c.sleep_until(12.5)
    assert c.now() == 12.5
    c.sleep_until(11.0)  # already past -- must return, not rewind
    assert c.now() == 12.5
    c.advance(0.5)
    assert c.now() == 13.0
    with pytest.raises(ValueError, match="cannot go backwards"):
        c.advance(-1.0)


def test_wall_clock_does_not_sleep_on_a_past_deadline() -> None:
    """A negative remaining time must not become `sleep(-x)` or a long wait."""
    c = WallClock()
    t = c.now()
    started = time.perf_counter()
    c.sleep_until(t - 5.0)
    assert time.perf_counter() - started < 0.05


def test_wall_clock_is_monotonic() -> None:
    c = WallClock()
    samples = [c.now() for _ in range(1000)]
    assert all(b >= a for a, b in itertools.pairwise(samples))


# --------------------------------------------------- deadlines under VirtualClock

def test_under_a_virtual_clock_the_deadlines_are_exact() -> None:
    """No OS in the way, so any slip at all would be a logic error."""
    loop = RateLoop(VirtualClock(), RATE, count=N, t0=0.0)
    ticks = list(loop)
    assert len(ticks) == N
    assert [t.i for t in ticks] == list(range(N))
    assert np.allclose(loop.t_target, np.arange(N) * PERIOD, rtol=0, atol=1e-12)
    assert np.all(loop.slip_s == 0.0)
    assert loop.achieved_hz == pytest.approx(RATE, rel=1e-12)
    assert loop.jitter_ms == pytest.approx(0.0, abs=1e-9)
    assert loop.missed == 0
    assert not any(t.late for t in ticks)


def test_a_three_second_trajectory_completes_in_milliseconds() -> None:
    """The reason M1 gets a VirtualClock at all."""
    started = time.perf_counter()
    loop = RateLoop(VirtualClock(), RATE, count=N, t0=0.0)
    ticks = list(loop)
    wall_s = time.perf_counter() - started
    assert ticks[-1].t_target == pytest.approx(3.0, abs=1e-9)  # 3 s of sim time
    assert wall_s < 0.05  # in under 50 ms of real time


def test_work_inside_the_loop_does_not_shift_later_deadlines() -> None:
    """The whole point of an absolute deadline: one slow tick is not contagious."""
    clock = VirtualClock()
    loop = RateLoop(clock, RATE, count=10, t0=0.0)
    for tick in loop:
        if tick.i == 3:
            clock.advance(0.025)  # 25 ms of work inside a 40 ms period
    assert np.allclose(loop.t_target, np.arange(10) * PERIOD, rtol=0, atol=1e-12)
    assert loop.missed == 0
    assert loop.achieved_hz == pytest.approx(RATE, rel=1e-9)


def test_an_overrun_is_counted_not_hidden() -> None:
    """Work longer than the period must show up, not be silently absorbed.

    This is the counter that makes the 100 Hz failure visible: `send` plus
    `get_angles` does not fit in 10 ms, and because MyPalletizer260 has no
    `set_fresh_mode` the surplus setpoints queue in the firmware.
    """
    clock = VirtualClock()
    loop = RateLoop(clock, RATE, count=10, t0=0.0)
    for _ in loop:
        clock.advance(0.055)  # 55 ms of work in a 40 ms period
    assert loop.missed == 9  # every tick after the first
    assert loop.achieved_hz == pytest.approx(1 / 0.055, rel=1e-9)
    assert loop.achieved_hz < RATE
    assert "9 missed" in loop.summary()


# ------------------------------------------------- the falsification

def test_absolute_deadlines_stay_bounded_while_cumulative_sleeps_drift() -> None:
    """The pairing this phase exists for, on a clock with fixed overshoot.

    Each `sleep_until` overshoots by 0.86 ms. Over 76 ticks:

      * cumulative: error is i * 0.86 ms, so the last tick lands ~64.5 ms late
        and the total span is stretched by the same amount;
      * absolute:   error is one overshoot, 0.86 ms, no matter how long it runs.
    """
    overshoot = 0.000_86
    ideal = np.arange(N) * PERIOD

    loop = RateLoop(OversleepClock(overshoot), RATE, count=N, t0=0.0)
    list(loop)
    abs_err = np.abs(loop.t_actual - ideal)

    cum = cumulative_loop(OversleepClock(overshoot), RATE, N)
    cum_err = np.abs(cum - (ideal + PERIOD))  # first cumulative tick is at one period

    # Absolute: bounded by a single overshoot, however long the run. Tick 0 is
    # exact -- its deadline is t0 itself, so there is nothing to sleep for --
    # and every tick after it is late by exactly one overshoot, not by more.
    assert abs_err[0] == 0.0
    assert abs_err.max() == pytest.approx(overshoot, abs=1e-12)
    assert np.allclose(abs_err[1:], overshoot, rtol=0, atol=1e-12)
    assert abs_err[-1] == pytest.approx(abs_err[1], abs=1e-12)

    # Cumulative: grows linearly. Every tick sleeps, including the first, so
    # the final error is N overshoots -- 76 * 0.86 = 65.4 ms. That is the same
    # figure a real 76-tick loop at 25 Hz produced on this machine (+65.4 ms
    # against +0.0 ms for absolute deadlines), which is the check that this
    # fixed-overshoot model is not a straw man.
    assert cum_err[-1] == pytest.approx(N * overshoot, rel=1e-9)
    assert cum_err[-1] * 1e3 == pytest.approx(65.4, abs=0.5)
    assert cum_err[-1] > 75 * abs_err.max()

    # And the failure is monotone -- it is drift, not noise.
    assert np.all(np.diff(cum_err) > 0)


def test_the_cumulative_loop_misses_the_bound_the_real_one_holds() -> None:
    """Stated as a bound, the way test_clock.py states its 1 ms one."""
    bound_ms = 5.0
    overshoot = 0.000_86
    ideal = np.arange(N) * PERIOD

    loop = RateLoop(OversleepClock(overshoot), RATE, count=N, t0=0.0)
    list(loop)
    assert np.abs(loop.t_actual - ideal).max() * 1e3 < bound_ms

    cum = cumulative_loop(OversleepClock(overshoot), RATE, N)
    assert np.abs(cum - (ideal + PERIOD)).max() * 1e3 > bound_ms


# ------------------------------------------------------------- bookkeeping

def test_t0_is_taken_at_first_iteration_not_at_construction() -> None:
    """So a loop can be built early and started late, sharing an origin."""
    clock = VirtualClock()
    loop = RateLoop(clock, RATE, count=3)
    clock.advance(10.0)
    assert loop.t0 is None
    ticks = list(loop)
    assert loop.t0 == 10.0
    assert ticks[0].t_target == 10.0


def test_an_explicit_t0_is_honoured() -> None:
    """`run_trajectory` takes t0 from outside to align a SimSensor with it."""
    loop = RateLoop(VirtualClock(100.0), RATE, count=3, t0=100.0)
    assert [t.t_target for t in loop] == [100.0, 100.04, 100.08]


def test_stats_are_nan_before_two_ticks_rather_than_dividing_by_zero() -> None:
    loop = RateLoop(VirtualClock(), RATE, count=1, t0=0.0)
    list(loop)
    assert len(loop) == 1
    assert np.isnan(loop.achieved_hz)
    assert np.isnan(loop.jitter_ms)
    assert np.isnan(loop.worst_gap_s)


def test_an_unbounded_loop_can_be_broken_out_of() -> None:
    loop = RateLoop(VirtualClock(), RATE, t0=0.0)
    for tick in loop:
        if tick.i == 4:
            break
    assert len(loop) == 5
    assert loop.achieved_hz == pytest.approx(RATE, rel=1e-9)


def test_tick_slip_is_the_lateness() -> None:
    t = Tick(i=2, t_target=0.08, t_actual=0.0812)
    assert t.slip == pytest.approx(0.0012)
    assert t.late
    assert not Tick(i=0, t_target=1.0, t_actual=1.0).late


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"rate_hz": 0.0}, "rate_hz must be"),
    ({"rate_hz": -25.0}, "rate_hz must be"),
    ({"count": -1}, "count must be"),
])
def test_bad_construction_is_refused(kwargs, match) -> None:
    args = {"clock": VirtualClock(), "rate_hz": RATE} | kwargs
    with pytest.raises(ValueError, match=match):
        RateLoop(**args)


def test_zero_ticks_is_allowed_and_reports_nothing() -> None:
    loop = RateLoop(VirtualClock(), RATE, count=0, t0=0.0)
    assert list(loop) == []
    assert len(loop) == 0
    assert np.isnan(loop.achieved_hz)


# -------------------------------- is the lift faithful to what run_trajectory does?

@pytest.mark.parametrize(
    ("span", "rate"), [(3.0, 25.0), (3.0, 30.0), (3.0, 7.0), (3.0, 20.0), (1.0, 25.0)]
)
def test_rateloop_matches_run_trajectory_when_the_span_holds_whole_periods(span, rate) -> None:
    """The notebook's own configuration, proved equivalent before P5 relies on it.

    `run_trajectory` does not pace off `i / rate_hz`. It paces off `t_r[i]`,
    the resampled trajectory's own time axis, which `resample` builds as
    `linspace(t[0], t[-1], round(span * rate) + 1)`. When `span * rate` is a
    whole number the two schedules are identical to the last bit, which the
    notebook's 3 s at 25 Hz is.
    """
    from erp.trajectory import resample

    t = np.linspace(0.0, span, int(span / 0.002))
    t_r, _ = resample(t, np.zeros((t.size, 3)), rate)
    loop = RateLoop(VirtualClock(), rate, count=len(t_r), t0=0.0)
    list(loop)
    assert np.allclose(loop.t_target, t_r, rtol=0, atol=1e-12)


def test_but_they_diverge_when_the_span_is_not_a_whole_number_of_periods() -> None:
    """The trap waiting for P5, pinned here so it is found by a test, not a plot.

    The two schedules encode different policies, and the difference only shows
    when `span * rate` is fractional:

      * `resample` preserves the trajectory's endpoints and stretches the
        spacing to fit -- the arm finishes exactly where it was told to;
      * `RateLoop` holds the nominal period exactly and lets the end fall
        where it falls.

    At a 2.5 s span and 25 Hz, `span * rate` is 62.5, `round` takes it to 62
    (banker's rounding, to even), and the schedules drift apart by up to 20 ms
    -- half a period. Whichever policy P5 picks, it has to pick one on purpose.
    """
    from erp.trajectory import resample

    span, rate = 2.5, 25.0
    t = np.linspace(0.0, span, int(span / 0.002))
    t_r, _ = resample(t, np.zeros((t.size, 3)), rate)
    loop = RateLoop(VirtualClock(), rate, count=len(t_r), t0=0.0)
    list(loop)

    assert span * rate == 62.5
    assert len(t_r) == 63  # round(62.5) -> 62, then + 1
    divergence_ms = float(np.abs(loop.t_target - t_r).max() * 1e3)
    assert divergence_ms == pytest.approx(20.0, abs=0.5)
    assert divergence_ms == pytest.approx(0.5 / rate * 1e3, abs=0.5)  # half a period
    # resample lands on the trajectory's end; RateLoop lands a half period short.
    assert t_r[-1] == pytest.approx(span, abs=1e-12)
    assert loop.t_target[-1] == pytest.approx(span - 0.5 / rate, abs=1e-9)


# ----------------------------------------- the one test that touches real time

def test_a_real_wall_clock_loop_holds_its_rate() -> None:
    """Loose bound on purpose: this measures the machine as much as the loop.

    ~0.3 s of real time, deliberately kept short enough not to need its own
    marker -- one more dimension on the fast-loop command costs more than it
    saves. The deterministic proof is the OversleepClock pair above; this
    exists so `WallClock` is known to be wired correctly at all, not to
    characterise the OS scheduler. A tighter bound here would fail on a loaded
    runner, and a flaky test teaches people to ignore failures.
    """
    loop = RateLoop(WallClock(), 50.0, count=15)
    list(loop)
    assert loop.achieved_hz == pytest.approx(50.0, rel=0.15)
    assert np.abs(loop.t_actual - loop.t_target).max() < 0.05
    assert loop.missed <= 1
