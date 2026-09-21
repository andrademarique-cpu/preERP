"""ADR-0002 P3: one trajectory generator, and proof it is the notebook's.

The phase was reshaped before it was built (see 5.2). Its original obligation
-- "peak velocity scales as 1/T, the chain-rule bug" -- guards `qd`, which
nothing downstream reads: the loop commands positions, the plant is driven by
positions, and the range guard differentiates `q` numerically. That test is
still here, and still worth having, but it protects a *plot*.

The obligation that carries the phase is the first one below: the package
generator reproduces the notebook's formula bit for bit. Five hand-written
copies of `sin(w*t - pi/2)*A + A` existed, one of them inside
`make_golden_run.py`, where a divergence would have been invisible -- the
fixture regenerates the trajectory rather than comparing against the notebook,
so `--check` would keep passing while the two drifted apart.
"""

from __future__ import annotations

import numpy as np
import pytest

from erp.trajectory import resample, sine_sweep

# The notebook's knobs (cell 3) and the golden run's (make_golden_run.py).
PERIODO_S = 3
DT = 0.002
AMPS_DEG = (30.0, 10.0, 20.0)


def notebook_formula(amps_deg, period_s, dt):
    """Verbatim from notebook cell 3, kept as the thing under comparison.

    This is the one copy that is allowed to remain, because its whole job is
    to disagree if `sine_sweep` ever stops matching it.
    """
    w = 2 * np.pi / period_s
    n_samples = int(period_s / dt)
    t = np.linspace(0, period_s, n_samples)
    amplitudes = np.deg2rad(np.array(amps_deg))
    q = np.sin(w * t - np.pi / 2)[:, None] * amplitudes + amplitudes
    qd = amplitudes * w * np.cos(w * t - np.pi / 2)[:, None]
    return t, q, qd


# --------------------------------------------- the obligation that matters

@pytest.mark.parametrize("period_s", [3, 1.0, 6.283, 2.0])
@pytest.mark.parametrize("amps", [AMPS_DEG, (30, 10, 20), (5.0, 5.0, 5.0)])
def test_sine_sweep_is_bit_identical_to_the_notebook_formula(period_s, amps) -> None:
    """Not `allclose` -- equal. A moved digit here moves the golden fixture."""
    t_nb, q_nb, qd_nb = notebook_formula(amps, period_s, DT)
    t, q, qd = sine_sweep(amps, period_s, DT)
    assert np.array_equal(t, t_nb)
    assert np.array_equal(q, q_nb)
    assert np.array_equal(qd, qd_nb)


def test_the_golden_pipeline_uses_the_package_generator() -> None:
    """The drift channel P3 closed, asserted rather than assumed.

    `make_golden_run.py` regenerates the trajectory to build `sensor_log_sim`,
    which is what `estimate_lag` measures the arm's lag against. While the
    formula was copied there by hand, editing the notebook's period and not
    the script's left the fixture describing a run the notebook no longer
    performed, with every check still green.
    """
    import erp.trajectory
    from erp.io.paths import resolve_repo_path

    source = resolve_repo_path("scripts/make_golden_run.py").read_text(encoding="utf-8")
    assert "from erp.trajectory import sine_sweep" in source
    assert "sine_sweep(AMPLITUDES_DEG, PERIODO_S, dt)" in source
    # And no second copy of the formula survives anywhere in it.
    assert "np.sin(w * t - np.pi / 2)" not in source
    assert erp.trajectory.sine_sweep is sine_sweep


# ----------------------------------------------------- shape and endpoints

def test_the_sweep_starts_and_ends_at_zero() -> None:
    """Commanded from the `home` keyframe, so it must not begin with a step."""
    _, q, _ = sine_sweep(AMPS_DEG, PERIODO_S, DT)
    assert np.allclose(q[0], 0.0, atol=1e-12)
    assert np.allclose(q[-1], 0.0, atol=1e-12)
    assert np.allclose(np.rad2deg(q.max(axis=0)), 2 * np.array(AMPS_DEG), atol=1e-9)


def test_the_sample_spacing_is_not_dt() -> None:
    """The carried-over off-by-one, pinned so a later repair is deliberate.

    `int(period_s / dt)` points spanning [0, period_s] inclusive are spaced
    `period_s / (n - 1)`. At 3 s and 2 ms that is 2.0013 ms against a 2.0000
    ms physics step: the commanded profile runs 0.067% slow and finishes one
    timestep behind the simulation clock. Small beside the arm's own ~395 ms
    transport lag, which is presumably why it went unnoticed -- and the golden
    fixture was generated with it, so correcting it moves the fixture.
    """
    t, _, _ = sine_sweep(AMPS_DEG, PERIODO_S, DT)
    assert t.size == 1500
    spacing = float(np.diff(t)[0])
    assert spacing == pytest.approx(PERIODO_S / 1499, rel=1e-15)
    assert spacing / DT == pytest.approx(1.000667, abs=1e-6)
    # Where the two clocks end up after the same number of steps.
    assert float(t[-1]) - (t.size - 1) * DT == pytest.approx(DT, abs=1e-9)


# -------------------------------------------- the original P3 obligation

def test_peak_velocity_scales_as_one_over_the_period() -> None:
    """`A * 2*pi / T`, the notebook's cost table, for real."""
    peaks = {}
    for period_s in (6.283, 4.0, 3.0, 2.0, 1.571):
        _, _, qd = sine_sweep(AMPS_DEG, period_s, DT)
        peaks[period_s] = np.rad2deg(np.abs(qd).max(axis=0))
    # The table in notebook cell 3, to 1 deg/s.
    assert peaks[6.283] == pytest.approx([30, 10, 20], abs=1.0)
    assert peaks[3.0] == pytest.approx([63, 21, 42], abs=1.0)
    assert peaks[1.571] == pytest.approx([120, 40, 80], abs=1.0)
    # 1.571 s is the hard floor: J1 hits the arm's 120 deg/s spec there.
    assert peaks[1.571][0] == pytest.approx(120.0, abs=0.5)

    # Halving the period doubles the peak -- but assert that on the analytic
    # peak `A*w`, not on the sampled one. `|qd|.max()` is a maximum over a
    # grid, and the grid does not land on the cosine's crest: it comes out
    # ~1e-6 low, which is sampling error, not a scaling error.
    for period_s, peak in peaks.items():
        analytic = np.array(AMPS_DEG) * (2 * np.pi / period_s)  # deg * rad/s = deg/s
        assert peak == pytest.approx(analytic, rel=1e-5)
        assert peak.max() <= analytic.max() + 1e-9  # sampled peak never exceeds it
    assert peaks[2.0] / peaks[4.0] == pytest.approx([2.0, 2.0, 2.0], rel=1e-5)


def test_dropping_the_chain_rule_factor_keeps_the_shape_and_breaks_the_scale() -> None:
    """The bug the notebook comment warns about, and exactly how bad it is.

    Without `w` the curve is still a cosine of the right shape -- correlation
    1.0 -- and only its scale is wrong, by the factor `w`. That is why no plot
    reveals it. Be precise about the consequence though: `qd` is a diagnostic.
    Nothing commands it, the plant is driven by `q`, and `JointMap.validate`
    differentiates `q` numerically. A missing `w` produces a wrong velocity
    plot and nothing else.
    """
    _, _, qd = sine_sweep(AMPS_DEG, PERIODO_S, DT)
    t = np.linspace(0, PERIODO_S, int(PERIODO_S / DT))
    amps = np.deg2rad(AMPS_DEG)
    qd_buggy = amps * np.cos(2 * np.pi / PERIODO_S * t - np.pi / 2)[:, None]  # no w

    w = 2 * np.pi / PERIODO_S
    assert np.allclose(qd, qd_buggy * w, rtol=0, atol=1e-15)
    assert w == pytest.approx(2.0944, abs=1e-4)
    for k in range(3):
        corr = np.corrcoef(qd[:, k], qd_buggy[:, k])[0, 1]
        assert corr == pytest.approx(1.0, abs=1e-12), "shape is unchanged, only scale"
    # The guard does NOT depend on qd: it differentiates q instead, and gets
    # the right answer either way.
    _, q, _ = sine_sweep(AMPS_DEG, PERIODO_S, DT)
    numeric = np.abs(np.gradient(np.rad2deg(q), t, axis=0)).max(axis=0)
    assert numeric == pytest.approx(np.rad2deg(np.abs(qd).max(axis=0)), rel=1e-3)


# ---------------------------------------------------------------- resample

def test_resample_preserves_the_endpoints() -> None:
    """The arm's final pose must be the commanded one, not the last kept sample."""
    t, q, _ = sine_sweep(AMPS_DEG, PERIODO_S, DT)
    t_r, q_r = resample(t, q, 25.0)
    assert t_r[0] == t[0]
    assert t_r[-1] == t[-1]
    assert np.allclose(q_r[0], q[0], rtol=0, atol=1e-12)
    assert np.allclose(q_r[-1], q[-1], rtol=0, atol=1e-12)


def test_resample_hits_the_requested_rate() -> None:
    """1500 points at 2 ms do not fit 115200 baud with request/response."""
    t, q, _ = sine_sweep(AMPS_DEG, PERIODO_S, DT)
    t_r, q_r = resample(t, q, 25.0)
    assert len(t_r) == 76  # 3 s * 25 Hz + 1
    assert q_r.shape == (76, 3)
    assert float(np.diff(t_r)[0]) == pytest.approx(1 / 25.0, rel=1e-3)
    assert len(t) / len(t_r) == pytest.approx(19.7, abs=0.1)


def test_resample_never_returns_fewer_than_two_points() -> None:
    """A single point is not a trajectory; `max(2, ...)` is load-bearing."""
    t, q, _ = sine_sweep(AMPS_DEG, PERIODO_S, DT)
    t_r, q_r = resample(t, q, 0.01)  # 0.03 setpoints over 3 s
    assert len(t_r) == 2
    assert q_r.shape == (2, 3)
    assert t_r[0] == t[0] and t_r[-1] == t[-1]


def test_resample_is_linear_between_samples() -> None:
    ramp = np.linspace(0.0, 1.0, 101)[:, None] * np.array([1.0, 2.0, 3.0])
    t = np.linspace(0.0, 1.0, 101)
    _, q_r = resample(t, ramp, 10.0)
    assert np.allclose(q_r[:, 0], np.linspace(0.0, 1.0, 11), rtol=0, atol=1e-12)


# ------------------------------------------------------------- bad inputs

@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"period_s": 0.0}, "period_s must be"),
        ({"period_s": -1.0}, "period_s must be"),
        ({"dt": 0.0}, "dt must be"),
        ({"dt": 4.0}, "need at least 2"),
    ],
)
def test_sine_sweep_refuses_nonsense(kwargs, match) -> None:
    args = {"amplitudes_deg": AMPS_DEG, "period_s": PERIODO_S, "dt": DT} | kwargs
    with pytest.raises(ValueError, match=match):
        sine_sweep(**args)


def test_sine_sweep_refuses_a_2d_amplitude() -> None:
    with pytest.raises(ValueError, match="1-D and non-empty"):
        sine_sweep([[30.0, 10.0]], PERIODO_S, DT)


def test_resample_refuses_mismatched_shapes() -> None:
    t, q, _ = sine_sweep(AMPS_DEG, PERIODO_S, DT)
    with pytest.raises(ValueError, match=r"q must be"):
        resample(t[:-1], q, 25.0)
    with pytest.raises(ValueError, match=r"t must be 1-D"):
        resample(np.column_stack([t, t]), q, 25.0)
    with pytest.raises(ValueError, match="rate_hz must be"):
        resample(t, q, 0.0)
