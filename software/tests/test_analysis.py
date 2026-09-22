"""`erp.analysis` — the lifted helpers, bit for bit, and the consistency verdict.

ADR-0002 P8. Two kinds of test here, and they answer different questions.

**Did the move change anything?** `estimate_lag` and `propagate_to_site` came out
of `scripts/make_golden_run.py` under § 5.3's "move it unchanged" rule. The
copies below are verbatim from the pre-move script (`git show 0b53942:` ...) and
the package version has to agree with them exactly — the same oracle trick
`test_fusion_runner.py` and `test_calibration.py` use. Saying "moved unchanged"
in a commit message is a claim; this is the check.

**Does the verdict mean anything?** `consistency_report` is new, so there is
nothing to be bit-identical to. It is tested the other way round: every metric is
paired with a deliberately-broken run that it has to catch. A diagnostic that
returns plausible numbers for a broken filter is worse than no diagnostic, since
it launders the failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from erp.analysis import ConsistencyReport, consistency_report, estimate_lag, propagate_to_site
from erp.analysis.consistency import NOMINAL_2SIGMA_COVERAGE


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("no pyproject.toml above this file")


REPO_ROOT = _repo_root()
XML_PATH = REPO_ROOT / "mechanical" / "mujoco_assets" / "MyPalletizer260" / "MyPalletizer260.xml"


# --------------------------------------------------------------------------
# Verbatim copies of the pre-P8 script. Do not tidy these: their value is that
# they are the old text. If one has to change to keep passing, the move was not
# a move.
# --------------------------------------------------------------------------


def _oracle_estimate_lag(t_ref, q_ref_deg, t_meas, q_meas_deg, max_lag_s=0.6, n=241):  # type: ignore[no-untyped-def]
    if len(t_meas) < 4:
        return float("nan")
    inside = (t_meas >= t_ref[0] + max_lag_s) & (t_meas <= t_ref[-1])
    if inside.sum() < 4:
        return float("nan")
    tm, qm = t_meas[inside], q_meas_deg[inside]
    best, best_lag = np.inf, float("nan")
    for lag in np.linspace(0.0, max_lag_s, n):
        ref = np.column_stack([np.interp(tm - lag, t_ref, q_ref_deg[:, k])
                               for k in range(q_ref_deg.shape[1])])
        rms = float(np.sqrt(np.mean((qm - ref) ** 2)))
        if rms < best:
            best, best_lag = rms, lag
    return best_lag


def _oracle_site_position_cov(XH, PP, model, data, rows, h_dyn, H_dyn):  # type: ignore[no-untyped-def]
    u = np.zeros(model.nu)
    p = np.empty((len(XH), 3))
    C = np.empty((len(XH), 3, 3))
    for i, (x, P) in enumerate(zip(XH, PP)):  # noqa: B905  (verbatim: see above)
        p[i] = h_dyn(x, u, model, data)[rows]
        H = H_dyn(x, u, model, data)[rows]
        C[i] = H @ P @ H.T
    return p, C


def _lag_inputs(lag_s: float = 0.4) -> tuple[Any, Any, Any, Any]:
    """A 3 s, 3-joint sine sweep and a delayed, noisy, 20 Hz sampling of it."""
    rng = np.random.default_rng(0)
    t_ref = np.arange(0.0, 3.0, 0.002)
    q_ref = np.column_stack([
        30.0 * np.sin(2 * np.pi * t_ref / 3.0),
        10.0 * np.sin(2 * np.pi * t_ref / 3.0 + 0.5),
        20.0 * np.sin(2 * np.pi * t_ref / 3.0 + 1.0),
    ])
    t_meas = np.arange(0.0, 3.0, 0.05)
    q_meas = np.column_stack([
        np.interp(t_meas - lag_s, t_ref, q_ref[:, k]) for k in range(3)
    ]) + rng.normal(0.0, 0.01, size=(t_meas.size, 3))
    return t_ref, q_ref, t_meas, q_meas


# ---------------------------------------------------------------- estimate_lag


def test_estimate_lag_is_bit_identical_to_the_pre_move_script() -> None:
    t_ref, q_ref, t_meas, q_meas = _lag_inputs()
    mine = estimate_lag(t_ref, q_ref, t_meas, q_meas)
    theirs = _oracle_estimate_lag(t_ref, q_ref, t_meas, q_meas)
    assert mine == theirs, "the move changed the arithmetic"


def test_estimate_lag_recovers_an_injected_delay() -> None:
    """Paired with the oracle test: bit-identical to something wrong is no good."""
    t_ref, q_ref, t_meas, q_meas = _lag_inputs(lag_s=0.4)
    # 241 points over [0, 0.6] is a 2.5 ms grid, so 0.4 is exactly on it.
    assert estimate_lag(t_ref, q_ref, t_meas, q_meas) == pytest.approx(0.4, abs=2.5e-3)


def test_estimate_lag_returns_nan_rather_than_raising_on_a_short_log() -> None:
    """A plot asking for a lag on three samples should get a hole, not a crash."""
    t = np.linspace(0.0, 0.1, 3)
    assert np.isnan(estimate_lag(t, np.zeros((3, 1)), t, np.zeros((3, 1))))


def test_estimate_lag_ignores_samples_outside_the_valid_window() -> None:
    """The window is why a run's first 0.6 s cannot invent correlation.

    Feeding it a measurement series that exists ONLY inside the excluded window
    must give NaN, not a lag fitted to the startup transient.
    """
    t_ref, q_ref, _, _ = _lag_inputs()
    t_early = np.linspace(0.0, 0.3, 8)          # entirely below t_ref[0] + max_lag_s
    q_early = np.zeros((t_early.size, 3))
    assert np.isnan(estimate_lag(t_ref, q_ref, t_early, q_early))


# ------------------------------------------------------------ consistency_report


def _chi2_nis(rng: np.random.Generator, n: int, nz: int, inflate: float = 1.0) -> Any:
    """NIS of `n` innovations drawn from N(0, S_true) but scored against S_true/inflate.

    `inflate = 1` is a filter whose `R` is right; `inflate = 10` is one that
    believes itself ten times more precise than it is, which is the failure NIS
    exists to expose and the one a trajectory plot cannot show.
    """
    y = rng.normal(size=(n, nz))
    return (y**2).sum(axis=1) * inflate


def test_report_lands_on_target_for_a_well_matched_filter() -> None:
    rng = np.random.default_rng(1)
    rep = consistency_report(_chi2_nis(rng, 4000, 12), nz=12, warmup_fraction=0.0)
    assert rep.nis_target == 12
    assert rep.nis_median == pytest.approx(11.34, rel=0.05)  # chi2(12) median


def test_report_catches_an_overconfident_filter() -> None:
    """The falsification. Same innovations, a covariance 10x too small.

    This is the pairing that makes the metric evidence: the well-matched run
    above sits on 12, this one sits an order of magnitude above it, and nothing
    about the two runs differs except what the filter believes.
    """
    rng = np.random.default_rng(1)
    good = consistency_report(_chi2_nis(rng, 4000, 12), nz=12, warmup_fraction=0.0)
    rng = np.random.default_rng(1)
    bad = consistency_report(_chi2_nis(rng, 4000, 12, inflate=10.0), nz=12, warmup_fraction=0.0)
    assert bad.nis_median > 8 * good.nis_median
    assert bad.nis_median > 5 * bad.nis_target


def test_median_survives_a_heavy_tail_that_wrecks_the_mean() -> None:
    """Why the report leads with the median.

    NIS has a heavy right tail from brief moments where P is small and the
    linearisation is poor. Twenty spikes in four thousand samples move the mean
    by more than a factor of two and the median not at all — so a mean-only
    report would call a healthy filter broken.
    """
    rng = np.random.default_rng(2)
    nis = _chi2_nis(rng, 4000, 12)
    nis[rng.choice(4000, 20, replace=False)] = 1.0e4
    rep = consistency_report(nis, nz=12, warmup_fraction=0.0)
    assert rep.nis_median == pytest.approx(11.34, rel=0.05)
    assert rep.nis_mean > 2 * rep.nis_median


def test_the_window_discards_the_p0_transient() -> None:
    """First half huge, second half on target: the report must see only the second."""
    nis = np.concatenate([np.full(500, 900.0), np.full(500, 12.0)])
    assert consistency_report(nis, nz=12).nis_median == pytest.approx(12.0)
    assert consistency_report(nis, nz=12, warmup_fraction=0.0).nis_median > 100
    assert consistency_report(nis, nz=12).window_start == 500


def test_missing_truth_reports_none_and_not_zero() -> None:
    """A 0.0 in a NEES field reads as 'perfect' and means the opposite."""
    rep = consistency_report(np.full(100, 12.0), nz=12)
    assert rep.nees_median is None
    assert rep.site_nees_median is None
    assert rep.coverage_2sigma is None
    assert "no calculable" in rep.summary()


def test_state_nees_and_site_nees_land_on_their_targets() -> None:
    rng = np.random.default_rng(3)
    n, nx = 2000, 11
    P = np.tile(np.eye(nx), (n, 1, 1))
    err = rng.normal(size=(n, nx))
    C = np.tile(np.eye(3) * 1e-6, (n, 1, 1))
    p_true = rng.normal(scale=1e-3, size=(n, 3))
    p_est = p_true - rng.normal(scale=1e-3, size=(n, 3))
    rep = consistency_report(
        np.full(n, 12.0), nz=12, err=err, P=P,
        p_true=p_true, p_est=p_est, C_site=C, warmup_fraction=0.0,
    )
    assert rep.nees_target == nx
    assert rep.nees_median == pytest.approx(10.34, rel=0.08)   # chi2(11) median
    assert rep.site_nees_median == pytest.approx(2.37, rel=0.15)  # chi2(3) median
    assert rep.coverage_2sigma is not None
    np.testing.assert_allclose(rep.coverage_2sigma, NOMINAL_2SIGMA_COVERAGE, atol=0.03)


def test_coverage_is_reported_per_axis_so_one_bad_axis_shows() -> None:
    """Averaging the three would hide exactly the case the fixture exhibits."""
    n = 2000
    rng = np.random.default_rng(4)
    C = np.tile(np.diag([1.0, 1.0, 1.0]), (n, 1, 1))
    err = rng.normal(size=(n, 3))
    err[:, 2] *= 3.0                       # z is three times noisier than believed
    rep = consistency_report(
        np.full(n, 12.0), nz=12,
        p_true=err, p_est=np.zeros((n, 3)), C_site=C, warmup_fraction=0.0,
    )
    assert rep.coverage_2sigma is not None
    assert rep.coverage_2sigma[0] > 0.93 and rep.coverage_2sigma[1] > 0.93
    assert rep.coverage_2sigma[2] < 0.65, "the bad axis must be visible on its own"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"err": np.zeros((4, 2))}, "err y P van juntos"),
        ({"p_true": np.zeros((4, 3))}, "van los tres o ninguno"),
        ({"err": np.zeros((4, 2)), "P": np.zeros((5, 2, 2))}, "incompatibles"),
        ({"err": np.zeros((4, 2)), "P": np.zeros((4, 3, 3))}, "deberia ser"),
        ({"warmup_fraction": 1.0}, r"\[0, 1\)"),
    ],
)
def test_inconsistent_inputs_are_refused(kwargs: dict[str, Any], match: str) -> None:
    """Shapes that almost line up are the dangerous ones: they produce a number."""
    with pytest.raises(ValueError, match=match):
        consistency_report(np.full(8, 12.0), nz=12, **kwargs)


def test_empty_nis_is_refused() -> None:
    with pytest.raises(ValueError, match="no vacio"):
        consistency_report(np.array([]), nz=12)


def test_report_is_frozen() -> None:
    rep = consistency_report(np.full(10, 12.0), nz=12)
    assert isinstance(rep, ConsistencyReport)
    with pytest.raises(AttributeError):
        rep.nis_median = 0.0  # type: ignore[misc]


# ------------------------------------------------------- against the real model


@pytest.mark.mujoco
def test_propagate_to_site_is_bit_identical_to_the_pre_move_script() -> None:
    pytest.importorskip("mujoco")
    if not XML_PATH.is_file():
        pytest.skip(f"{XML_PATH.name} is missing (Git LFS?)")
    from erp.sim.mujoco import H_dyn, h_dyn, state_dim
    from erp.sim.plant import blind_variant

    model, data = blind_variant(XML_PATH)
    nx = state_dim(model)
    rng = np.random.default_rng(0)
    XH = rng.normal(scale=0.05, size=(6, nx))
    PP = np.tile(np.eye(nx) * 1e-4, (6, 1, 1))
    rows = np.arange(3)

    p_new, C_new = propagate_to_site(XH, PP, model, data, rows)
    p_old, C_old = _oracle_site_position_cov(XH, PP, model, data, rows, h_dyn, H_dyn)
    np.testing.assert_array_equal(p_new, p_old)
    np.testing.assert_array_equal(C_new, C_old)


@pytest.mark.mujoco
def test_propagate_to_site_keeps_the_full_block_the_fixture_drops() -> None:
    """The reason the function returns C and not just its diagonal.

    `sig_ef` in the frozen fixture is `sqrt(diag(C))`, and the off-diagonal
    terms were computed and thrown away. They are not negligible: on a real
    posterior the cross-covariance between world axes is the same order as the
    diagonal, which is what makes the error ellipsoid tilt.
    """
    pytest.importorskip("mujoco")
    if not XML_PATH.is_file():
        pytest.skip(f"{XML_PATH.name} is missing (Git LFS?)")
    from erp.sim.mujoco import state_dim
    from erp.sim.plant import blind_variant, warmup_to_rest

    model, data = blind_variant(XML_PATH)
    nx = state_dim(model)
    # Same start the golden run uses: the servoed equilibrium, never zeros. From
    # the `home` keyframe with act = 0 the rest reading is off by 114 sigma.
    q_cmd = np.zeros(3)
    x_rest = np.r_[warmup_to_rest(model, data, q_cmd), q_cmd]
    P = np.eye(nx) * 1e-4
    _, C = propagate_to_site(x_rest[None, :], P[None, :, :], model, data, np.arange(3))
    off = np.abs(C[0] - np.diag(np.diag(C[0]))).max()
    assert off > 0.05 * np.abs(np.diag(C[0])).max(), (
        "off-diagonal terms are not negligible; dropping them changes the band"
    )
