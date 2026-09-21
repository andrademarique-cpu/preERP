"""ADR-0002 P5: `FilterRunner`, the only object that turns a timestamp into steps.

No mujoco here. The runner talks to the `Estimator` protocol, and an `EKF` over
`LinearDynamics` satisfies it, so the whole scheduling layer is testable in the
fast loop -- which is the point of the `DiscreteDynamics` seam P4 built.

Two things this file is organised around:

* **The obligation** (ADR-0002 6): one 50 ms advance equals 25 x 2 ms predicts.
* **The falsification**: out-of-order input increments `.discarded`. A runner
  that quietly applied a late sample would look identical on every plot.

`legacy_run_imu_ekf` below is the independent oracle, the same role
`closed_form_kf` plays in `test_ekf_linear.py`: it is the notebook's loop
written out verbatim, so "P5 moved code without moving numbers" is asserted
here bit-for-bit and not only through the mujoco-bound golden fixture.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from erp.core.clock import VirtualClock
from erp.core.types import Measurement
from erp.estimators import EKF
from erp.fusion import FilterRunner
from erp.models.linear import LinearDynamics

DT = 0.002  # the arm's physics step, so the 25-predicts arithmetic is the real one
IMU_PERIOD = 0.05  # the Teensy's ~20 Hz
RNG = np.random.default_rng(20260921)


# --------------------------------------------------------------- the fixtures


def make_filter(*, nz: int = 3) -> EKF:
    """1-D constant velocity, observed through `nz` channels. Well conditioned."""
    A = np.array([[1.0, DT], [0.0, 1.0]])
    C = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])[:nz]
    Q = np.diag([1e-8, 1e-6])
    R = np.diag([0.25, 0.10, 0.50])[:nz, :nz]
    return EKF(np.array([0.0, 1.0]), np.diag([1.0, 1.0]), Q, R, LinearDynamics(A, C=C, dt=DT))


def meas(t: float, z, rows, R, source: str = "imu") -> Measurement:
    """One measurement, with `rows`/`R` shared read-only the way a Sensor builds them."""
    rows_a = np.asarray(rows, dtype=np.intp)
    R_a = np.asarray(R, dtype=np.float64)
    rows_a.flags.writeable = False
    R_a.flags.writeable = False
    return Measurement(
        z=np.asarray(z, dtype=np.float64),
        timestamp=float(t),
        rows=rows_a,
        R=R_a,
        source=source,
    )


def imu_like(n: int = 8, *, rows=(0,), t0: float = IMU_PERIOD, source: str = "imu"):
    """`n` samples at the IMU's 50 ms period, carrying the filter's own R block."""
    R_full = make_filter().R
    rows_a = np.asarray(rows, dtype=np.intp)
    block = R_full[np.ix_(rows_a, rows_a)]
    return [
        meas(t0 + i * IMU_PERIOD, RNG.normal(0.0, 0.5, size=rows_a.size), rows_a, block, source)
        for i in range(n)
    ]


def legacy_run_imu_ekf(ekf, t_meas, Z, rows):
    """The notebook's `run_imu_ekf`, copied verbatim. The oracle, not the code under test.

    Deliberately keeps everything P5 changed: `round(tk / dt)` placement, a
    `k_now` cursor that never rewinds, no late branch, no counter, and
    `update(z, rows)` with no per-measurement R. The one cosmetic difference
    from the cell is dropping its `int(...)` cast, which ruff flags as
    redundant and P3 already established is a no-op on a float.
    """
    dt = ekf.dyn.dt
    z_full = np.zeros(ekf.dyn.nz)
    k_now = 0
    T, XH, PP, NIS = [0.0], [ekf.x.copy()], [ekf.P.copy()], []
    for tk, zk in zip(t_meas, Z, strict=True):
        while k_now < round(tk / dt):
            ekf.predict()
            k_now += 1
            T.append(k_now * dt)
            XH.append(ekf.x.copy())
            PP.append(ekf.P.copy())
        z_full[rows] = zk
        _, nis = ekf.update(z_full, rows)
        NIS.append(nis)
        T.append(k_now * dt)
        XH.append(ekf.x.copy())
        PP.append(ekf.P.copy())
    return np.array(T), np.array(XH), np.array(PP), np.array(NIS)


# ------------------------------------------------------------- the obligation


def test_one_50ms_advance_equals_25_predicts_of_2ms() -> None:
    """ADR-0002 6's P5 row, stated in the units the arm actually runs at."""
    runner = FilterRunner(make_filter(), t0=0.0)
    assert runner.advance_to(IMU_PERIOD) == 25
    assert runner.steps == 25
    assert runner.t_filter == pytest.approx(IMU_PERIOD, abs=1e-15)


def test_advancing_in_one_jump_equals_advancing_step_by_step() -> None:
    """The step count is all that matters; how the caller asks for it is not."""
    one_jump = FilterRunner(make_filter(), t0=0.0)
    one_jump.advance_to(IMU_PERIOD)

    stepwise = FilterRunner(make_filter(), t0=0.0)
    for k in range(1, 26):
        stepwise.advance_to(k * DT)

    assert stepwise.steps == one_jump.steps == 25
    assert np.array_equal(stepwise.est.x, one_jump.est.x)
    assert np.array_equal(stepwise.est.P, one_jump.est.P)


def test_the_runner_reproduces_run_imu_ekf_bit_for_bit() -> None:
    """P5 moved code; it must not have moved numbers. The mujoco-free half of --check.

    `array_equal`, not `allclose`: the golden fixture is compared at
    `rtol=1e-12`, and the only way to keep that honest across a refactor is for
    the refactor to be exact.
    """
    ms = imu_like(10, rows=(0, 2))
    t_meas = np.array([m.timestamp for m in ms])
    Z = np.stack([m.z for m in ms])
    rows = ms[0].rows

    T, XH, PP, NIS = legacy_run_imu_ekf(make_filter(), t_meas, Z, rows)
    hist = FilterRunner(make_filter(), t0=0.0).run(ms)

    assert np.array_equal(hist.t, T)
    assert np.array_equal(hist.x, XH)
    assert np.array_equal(hist.P, PP)
    assert np.array_equal(hist.nis, NIS)
    assert hist.discarded == 0


# ----------------------------------------------------------- the falsification


def test_an_out_of_order_measurement_is_dropped_and_counted() -> None:
    """The falsification ADR-0002 6 pairs with the obligation.

    `run_imu_ekf` had no branch for this: its `while` simply did not run and the
    stale sample was applied anyway, against a covariance already propagated
    past it. Nothing in the output said so.
    """
    ms = imu_like(4)
    late = meas(ms[0].timestamp, ms[0].z, ms[0].rows, ms[0].R)  # a repeat of the first

    runner = FilterRunner(make_filter(), t0=0.0)
    runner.run([*ms, late])

    assert runner.discarded == 1
    assert len(runner.history().nis) == len(ms)


def test_the_dropped_sample_does_not_touch_the_filter() -> None:
    """Counted is not enough -- it must also have had no effect."""
    ms = imu_like(4)
    clean = FilterRunner(make_filter(), t0=0.0)
    clean.run(ms)

    dirty = FilterRunner(make_filter(), t0=0.0)
    dirty.run([*ms, meas(ms[0].timestamp, ms[0].z, ms[0].rows, ms[0].R)])

    assert dirty.discarded == 1
    assert np.array_equal(dirty.est.x, clean.est.x)
    assert np.array_equal(dirty.est.P, clean.est.P)


def test_ingest_returns_none_exactly_when_it_drops() -> None:
    runner = FilterRunner(make_filter(), t0=0.0)
    ms = imu_like(3)
    assert all(runner.ingest(m) is not None for m in ms)
    assert runner.ingest(meas(ms[0].timestamp, ms[0].z, ms[0].rows, ms[0].R)) is None


def test_sub_step_jitter_is_not_mistaken_for_out_of_order() -> None:
    """A sample 0.9 ms early still belongs to the step the filter is on.

    The criterion is the step, not the timestamp. Were it the timestamp, a
    merely irregular stream would report drops it never suffered -- and the
    IMU's own period is 50 +- 1 ms, so this is the normal case, not a corner.
    """
    rows, R = np.asarray([0], dtype=np.intp), np.asarray([[0.25]])
    runner = FilterRunner(make_filter(), t0=0.0)
    runner.ingest(meas(0.050, [0.1], rows, R))
    assert runner.steps == 25

    assert runner.ingest(meas(0.0491, [0.1], rows, R)) is not None  # rounds to step 25
    assert runner.discarded == 0
    assert runner.steps == 25  # applied in place, no predicts, no rewind

    assert runner.ingest(meas(0.0489, [0.1], rows, R)) is None  # rounds to step 24
    assert runner.discarded == 1


# --------------------------------------------- two streams, unequal latencies


def two_streams(latency_fast: float, latency_slow: float):
    """Samples from two sensors, handed over in ARRIVAL order rather than sample order.

    This is the shape the real system has: the IMU stamps the sample instant and
    the transport delay decides when the loop sees it, so two sensors with
    different delays hand their samples over interleaved.
    """
    rows_f, rows_s = np.asarray([0], dtype=np.intp), np.asarray([1], dtype=np.intp)
    R_f, R_s = np.asarray([[0.25]]), np.asarray([[0.10]])
    samples = []
    for i in range(6):
        t_f = 0.05 + i * IMU_PERIOD
        t_s = 0.06 + i * IMU_PERIOD
        samples.append((t_f + latency_fast, meas(t_f, [0.1], rows_f, R_f, "fast")))
        samples.append((t_s + latency_slow, meas(t_s, [0.1], rows_s, R_s, "slow")))
    samples.sort(key=lambda pair: pair[0])
    return [m for _, m in samples]


def test_unequal_latencies_arrive_out_of_order_and_the_buffer_sorts_them() -> None:
    """`buffer_horizon` buys ordering with latency. There is no third option."""
    arrivals = two_streams(0.0, 0.06)
    stamps = [m.timestamp for m in arrivals]
    assert stamps != sorted(stamps), "the fixture must actually be out of order"

    runner = FilterRunner(make_filter(nz=2), t0=0.0, buffer_horizon=0.07)
    hist = runner.run(arrivals)

    assert runner.discarded == 0
    assert runner.pending == 0  # flush() emptied the buffer
    assert len(hist.nis) == len(arrivals)
    assert np.all(np.diff(hist.t_update) >= 0.0)  # applied in timestamp order


def test_the_same_pair_without_a_buffer_discards() -> None:
    """The pairing ADR-0002 5.2 asks for: with `buffer_horizon=0`, non-zero discarded.

    Without this, the test above proves only that the runner works on sorted
    input -- which is what `run_imu_ekf` already did.
    """
    runner = FilterRunner(make_filter(nz=2), t0=0.0, buffer_horizon=0.0)
    runner.run(two_streams(0.0, 0.06))
    assert runner.discarded > 0


def test_equal_latencies_produce_no_reordering_at_all() -> None:
    """ADR-0002 4.8's warning, pinned so the test above cannot rot into a tautology.

    Equal latencies shift every timestamp by the same amount, so arrival order
    IS sample order and nothing is ever out of order. A reordering test built on
    equal latencies passes without testing anything -- the deleted finger viewer
    learned this the hard way.
    """
    arrivals = two_streams(0.06, 0.06)
    stamps = [m.timestamp for m in arrivals]
    assert stamps == sorted(stamps), "equal latencies cannot reorder anything"

    runner = FilterRunner(make_filter(nz=2), t0=0.0, buffer_horizon=0.0)
    runner.run(arrivals)
    assert runner.discarded == 0


def test_flush_releases_the_tail_the_horizon_is_still_holding() -> None:
    """Without it, a `buffer_horizon` silently eats the end of every run."""
    ms = imu_like(5)
    runner = FilterRunner(make_filter(), t0=0.0, buffer_horizon=0.2)
    for m in ms:
        runner.ingest(m)
    assert runner.pending > 0

    released = runner.flush()
    assert runner.pending == 0
    assert len(runner.history().nis) == len(ms)
    assert len(released) > 0


def test_a_clock_releases_held_samples_once_wall_time_has_passed_them() -> None:
    """What the optional `clock` buys, stated against the case that lacks it.

    The watermark is otherwise the newest timestamp *ingested*, which is
    deterministic -- a run is a pure function of its input, which is what the
    offline path wants. The price is that a sensor going quiet strands whatever
    the horizon is holding, because nothing moves the watermark. A clock moves
    it with wall time instead.
    """
    ms = imu_like(3)
    late = imu_like(1, t0=ms[-1].timestamp + IMU_PERIOD)[0]

    clock = VirtualClock(0.0)
    with_clock = FilterRunner(make_filter(), t0=0.0, buffer_horizon=0.1, clock=clock)
    without = FilterRunner(make_filter(), t0=0.0, buffer_horizon=0.1)
    for m in ms:
        with_clock.ingest(m)
        without.ingest(m)
    assert with_clock.pending > 0 and without.pending > 0

    clock.advance(10.0)  # wall time moves well past every held sample
    with_clock.ingest(late)
    without.ingest(late)

    assert with_clock.pending == 0
    assert without.pending > 0  # the falsification: no clock, nothing released


# ------------------------------- advancing an online loop without eating its input


N_LATENT = 40
CMD_HZ = 25.0  # the arm's setpoint rate; note 25 and 20 are not integer multiples
PHASES = (0.050, 0.057, 0.063, 0.070)

# Samples of N_LATENT lost by a loop that advances the filter to `now`, at the
# SimSensor's 5 ms latency. The shape is the point: zero at three phases out of
# four, a quarter of the stream at the fourth.
#
# Only the 5 ms row is pinned by value. The counts at larger latencies are not,
# and the reason is worth knowing before anyone "tightens" this file: whether a
# given sample is lost turns on whether a command tick falls inside the latency
# window after its timestamp, so a sample sitting within one ULP of that
# boundary flips with the arithmetic used to BUILD the fixture -- writing the
# sample instants as `i / 20.0` rather than `i * 0.05` moves the 20 ms counts by
# three. That makes exact counts there a property of this file, not of the
# runner, and asserting them would be measuring the test. What is a property of
# the runner is the monotonicity, which the next test asserts instead.
LOSS_AT_5MS = {0.050: 0, 0.057: 10, 0.063: 0, 0.070: 0}
LATENCIES = (0.005, 0.020)


def drive_loop(runner: FilterRunner, latency: float, phase: float, *, safe: bool, clock=None):
    """A command loop at CMD_HZ over a sensor that hands samples over `latency` late.

    The order is the real one: send, drain what has ARRIVED, then advance. A
    sample stamped `s` is handed to `ingest` at the first tick at or after
    `s + latency`.
    """
    rows, R = np.asarray([0], dtype=np.intp), np.asarray([[0.25]])
    stream = [
        (phase + i * IMU_PERIOD + latency, meas(phase + i * IMU_PERIOD, [0.1], rows, R))
        for i in range(N_LATENT)
    ]
    period, handed = 1.0 / CMD_HZ, 0
    for i in range(int(stream[-1][0] / period) + 3):
        now = i * period
        if clock is not None:
            clock.sleep_until(now)
        while handed < len(stream) and stream[handed][0] <= now:
            runner.ingest(stream[handed][1])
            handed += 1
        runner.advance_to_safe(now) if safe else runner.advance_to(now)
    runner.flush()
    return runner


@pytest.mark.parametrize("phase", PHASES)
def test_advancing_to_now_silently_loses_latent_samples(phase) -> None:
    """The trap, found by running the dry-run loop rather than by reading the ADR.

    A sensor stamps the *sample* instant and hands it over later, so a filter
    already advanced to `now` is ahead of whatever arrives next. Whether that
    costs anything turns on where the command ticks fall relative to the sample
    instants -- and since 25 Hz and 20 Hz are not integer multiples, that phase
    drifts through a run.

    Precisely what this is and is not: an intermittent, phase-dependent, SILENT
    loss, not a filter that stops working. The loop runs, the plots are drawn,
    and the estimate is merely worse. The notebook's own dry run lost **1 of
    61** this way, and one of the four phases here loses **10 of 40** -- which
    is the real hazard, because at the other three it costs exactly nothing and
    looks fine.
    """
    runner = drive_loop(FilterRunner(make_filter(), t0=0.0), 0.005, phase, safe=False)
    lost = LOSS_AT_5MS[phase]
    assert runner.discarded == lost
    assert len(runner.history().nis) == N_LATENT - lost


def test_more_transport_latency_loses_more_samples() -> None:
    """The monotonicity, which IS a property of the runner rather than of the fixture.

    Four times the latency cannot lose fewer samples at any phase, and must lose
    strictly more somewhere. This is what the un-pinnable exact counts were
    trying to say.
    """
    def lost(latency, phase):
        return drive_loop(FilterRunner(make_filter(), t0=0.0), latency, phase, safe=False).discarded

    small = [lost(0.005, p) for p in PHASES]
    large = [lost(0.020, p) for p in PHASES]

    assert all(b >= a for a, b in zip(small, large, strict=True))
    assert sum(large) > sum(small)
    assert all(n > 0 for n in large)  # at 20 ms no phase escapes
    assert min(small) == 0  # at 5 ms most do, which is what makes it a trap


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("latency", LATENCIES)
@pytest.mark.parametrize("with_clock", [False, True])
def test_advance_to_safe_keeps_every_sample_at_every_phase(latency, phase, with_clock) -> None:
    """The pairing: same streams, same loop, horizon covering the latency.

    Run both with and without a clock because they fail differently when they
    fail. Without one the release watermark is the newest timestamp ingested, so
    a held sample waits for its successor while `now` marches on -- which
    discarded the entire stream until `advance_to_safe` learned to clamp to the
    oldest pending sample. The clamp costs an extra sample period of lag and is
    the reason this parametrisation exists.
    """
    clock = VirtualClock(0.0) if with_clock else None
    runner = drive_loop(
        FilterRunner(make_filter(), t0=0.0, buffer_horizon=latency, clock=clock),
        latency,
        phase,
        safe=True,
        clock=clock,
    )
    assert runner.discarded == 0
    assert len(runner.history().nis) == N_LATENT


def test_advance_to_safe_is_advance_to_when_there_is_no_horizon() -> None:
    """A replayed log has no transport latency, so the two coincide -- and must."""
    plain = FilterRunner(make_filter(), t0=0.0)
    assert plain.advance_to_safe(IMU_PERIOD) == 25
    assert plain.t_filter == pytest.approx(IMU_PERIOD, abs=1e-15)


def test_advance_to_safe_never_steps_past_a_held_sample() -> None:
    """The clamp, on its own, without a whole loop around it."""
    runner = FilterRunner(make_filter(), t0=0.0, buffer_horizon=0.01)
    runner.ingest(meas(0.05, [0.1], np.asarray([0], dtype=np.intp), [[0.25]]))
    assert runner.pending == 1

    runner.advance_to_safe(10.0)  # "now" is far in the future
    assert runner.t_filter <= 0.05
    assert runner.discarded == 0
    runner.flush()
    assert runner.discarded == 0


# ------------------------------------------------- what the measurement carries


def test_the_measurements_r_reaches_the_filter() -> None:
    """The other half of ADR-0002 3.2, and P6's precondition.

    Until P5 the runner unpacked measurements into bare arrays, so a calibrated
    `R` stopped at `MeasurementLog` and changed nothing. Two runs differing only
    in `Measurement.R` must now differ in their output; if they do not, the
    parameter is still decorative.
    """
    rows = np.asarray([0], dtype=np.intp)
    zs = RNG.normal(0.0, 0.5, size=(6, 1))

    tight = FilterRunner(make_filter(), t0=0.0)
    tight.run([meas(0.05 + i * IMU_PERIOD, z, rows, [[1e-4]]) for i, z in enumerate(zs)])

    loose = FilterRunner(make_filter(), t0=0.0)
    loose.run([meas(0.05 + i * IMU_PERIOD, z, rows, [[1e4]]) for i, z in enumerate(zs)])

    assert not np.allclose(tight.est.x, loose.est.x)
    # A tighter R must trust the measurement more, so the posterior variance falls.
    assert tight.est.P[0, 0] < loose.est.P[0, 0]


def test_a_measurement_r_of_the_wrong_shape_is_refused() -> None:
    rows = np.asarray([0, 1], dtype=np.intp)
    runner = FilterRunner(make_filter(nz=2), t0=0.0)
    with pytest.raises(ValueError, match="R tiene que ser"):
        runner.ingest(meas(0.05, [0.1, 0.1], rows, [[0.25]]))


# ------------------------------------------------------------- what it records


def test_the_history_carries_the_prior_and_the_posterior_of_every_update() -> None:
    """The duplicated-timestamp convention the golden fixture froze.

    One row per predict, plus one more at the SAME time for each update, plus
    the initial state. A reader who assumes `t` is strictly increasing will
    double-count every update instant.
    """
    ms = imu_like(4)
    runner = FilterRunner(make_filter(), t0=0.0)
    hist = runner.run(ms)

    assert len(hist.t) == 1 + runner.steps + len(ms)
    assert hist.x.shape == (len(hist.t), 2)
    assert hist.P.shape == (len(hist.t), 2, 2)
    assert np.all(np.diff(hist.t) >= 0.0)
    assert (np.diff(hist.t) == 0.0).sum() == len(ms)  # exactly one repeat per update


def test_record_false_drops_the_state_trace_but_keeps_the_nis() -> None:
    """The flag gates the part that grows with the run, never the diagnostic."""
    ms = imu_like(4)
    runner = FilterRunner(make_filter(), t0=0.0, record=False)
    hist = runner.run(ms)

    assert hist.x.shape == (0, 2)
    assert hist.P.shape == (0, 2, 2)
    assert len(hist.nis) == len(ms)
    assert len(hist.t_update) == len(ms)

    recorded = FilterRunner(make_filter(), t0=0.0).run(ms)
    assert np.array_equal(hist.nis, recorded.nis)  # the estimate itself is unaffected


def test_the_belief_is_a_snapshot_not_a_live_reference() -> None:
    """Held across ticks by the online loop, so it must not change under the holder."""
    runner = FilterRunner(make_filter(), t0=0.0)
    before = runner.belief
    P_seen = before.P.copy()

    # Not the estimator's own arrays: an estimator is free to write through
    # them, and `EKF` only happens to rebind instead.
    assert before.x is not runner.est.x
    assert before.P is not runner.est.P

    runner.advance_to(IMU_PERIOD)

    assert before.t == 0.0
    assert np.array_equal(before.P, P_seen)  # unchanged by 25 predicts
    assert runner.belief.t == pytest.approx(IMU_PERIOD, abs=1e-15)
    assert not np.array_equal(before.P, runner.belief.P)


def test_filter_time_is_quantised_to_the_step() -> None:
    """The filter exists only at step boundaries, so `t_filter` is not a sample stamp."""
    runner = FilterRunner(make_filter(), t0=0.0)
    runner.ingest(meas(0.0511, [0.1], np.asarray([0], dtype=np.intp), [[0.25]]))
    assert runner.t_filter == pytest.approx(26 * DT, abs=1e-15)


def test_t0_shifts_the_whole_axis() -> None:
    """The online path hands over `run_trajectory`'s t0 so stamps share one origin."""
    t0 = 1234.5
    runner = FilterRunner(make_filter(), t0=t0)
    assert runner.advance_to(t0 + IMU_PERIOD) == 25
    assert runner.t_filter == pytest.approx(t0 + IMU_PERIOD, abs=1e-9)


# ------------------------------------------------------------------- contract


def test_the_runner_is_blind_to_the_control_by_signature() -> None:
    """Same shape as P4's test on the EKF: the guarantee is that there is nowhere to put one.

    ADR-0002 4.7 rule 4 -- commands never touch the filter. This is the layer
    that would have been tempted, because it is the one that can see the command
    loop.
    """
    for name in ("__init__", "advance_to", "ingest", "run", "flush"):
        params = inspect.signature(getattr(FilterRunner, name)).parameters
        assert "u" not in params, f"{name} grew a control parameter"
        assert "ctrl" not in params


def test_advance_to_never_rewinds() -> None:
    runner = FilterRunner(make_filter(), t0=0.0)
    runner.advance_to(IMU_PERIOD)
    assert runner.advance_to(0.0) == 0
    assert runner.advance_to(-5.0) == 0
    assert runner.steps == 25


def test_an_empty_log_is_a_run_of_nothing_not_an_error() -> None:
    hist = FilterRunner(make_filter(), t0=0.0).run([])
    assert hist.t.shape == (1,)  # the initial state, nothing else
    assert hist.nis.shape == (0,)
    assert hist.discarded == 0


@pytest.mark.parametrize("horizon", [-1e-9, -1.0])
def test_a_negative_buffer_horizon_is_refused(horizon) -> None:
    with pytest.raises(ValueError, match="buffer_horizon"):
        FilterRunner(make_filter(), t0=0.0, buffer_horizon=horizon)
