"""`erp.viz` — the ellipse arithmetic in the fast suite, the figures behind a skip.

ADR-0002 P8. The split in the package is mirrored here on purpose: the geometry
is pure numpy and gets real assertions, the figures get smoke tests and nothing
more.

That is not laziness about the figures. A plot is checked by looking at it, and
a test that asserts a line exists at some coordinate passes just as happily when
the line is the wrong signal. What *can* be wrong in a way a reader would not
notice is the covariance arithmetic underneath — an ellipse drawn from a
mis-read block looks perfectly plausible — so that is what is pinned here.

matplotlib lives in the `[viz]` extra and CI installs `.[dev]`, so the figure
tests skip there rather than fail. The geometry tests never skip.
"""

from __future__ import annotations

import numpy as np
import pytest

from erp.viz import ellipsoid_axes, principal_tilt_deg, sigma_per_axis


def _rot_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ------------------------------------------------------------------- geometry


def test_sigma_per_axis_is_the_square_root_of_the_diagonal() -> None:
    C = np.diag([4.0, 9.0, 16.0])[None, ...]
    np.testing.assert_allclose(sigma_per_axis(C), [[2.0, 3.0, 4.0]])


def test_a_single_covariance_is_accepted_without_a_leading_axis() -> None:
    np.testing.assert_allclose(sigma_per_axis(np.diag([4.0, 9.0, 16.0])), [[2.0, 3.0, 4.0]])


def test_ellipsoid_axes_of_a_diagonal_block_are_its_sigmas_ascending() -> None:
    lengths, _ = ellipsoid_axes(np.diag([16.0, 4.0, 9.0])[None, ...])
    np.testing.assert_allclose(lengths, [[2.0, 3.0, 4.0]])


def test_ellipsoid_axes_symmetrises_first() -> None:
    """The falsification of the symmetrisation step.

    `eigh` reads one triangle and ignores the other, so handing it a block that
    is asymmetric — which is what a covariance out of the filter is, to
    round-off — silently uses half the matrix. Here the asymmetry is large
    enough to be visible: without symmetrising, the answer is the one for a
    different covariance entirely, and it still looks like a valid ellipse.
    """
    S = np.array([[4.0, 1.0, 0.0], [1.0, 9.0, 0.0], [0.0, 0.0, 1.0]])
    A = np.array([[0.0, 3.0, 0.0], [-3.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    lengths, _ = ellipsoid_axes(S + A)
    np.testing.assert_allclose(lengths[0], np.sqrt(np.linalg.eigvalsh(S)), atol=1e-12)
    raw = np.sqrt(np.maximum(np.linalg.eigvalsh(S + A), 0.0))
    assert not np.allclose(lengths[0], raw), "symmetrising must change the answer here"


def test_a_tiny_negative_eigenvalue_is_clamped_not_raised() -> None:
    """Round-off produces these; refusing to draw over -1e-19 helps nobody."""
    lengths, _ = ellipsoid_axes(np.diag([1.0, 1.0, -1e-19])[None, ...])
    assert lengths[0, 0] == 0.0
    assert np.all(np.isfinite(lengths))


def test_an_axis_aligned_ellipsoid_has_zero_tilt() -> None:
    assert principal_tilt_deg(np.diag([1.0, 4.0, 9.0])[None, ...])[0] == pytest.approx(0.0)


def test_tilt_recovers_a_known_rotation() -> None:
    R = _rot_z(30.0)
    C = R @ np.diag([9.0, 1.0, 0.25]) @ R.T
    assert principal_tilt_deg(C[None, ...])[0] == pytest.approx(30.0, abs=1e-6)


def test_tilt_is_what_makes_per_axis_sigmas_understate_the_worst_direction() -> None:
    """The claim behind "propagate the full block", made checkable.

    At 45 degrees the two world-axis sigmas are equal and both are a factor
    sqrt(2) below the true major semi-axis. The fixture stores only those two
    numbers, so anyone reading them alone is 29% optimistic about the worst
    direction — the same effect measured on the golden run as a median 9.6% and
    a maximum of 39.7%.
    """
    R = _rot_z(45.0)
    C = R @ np.diag([1.0, 0.0, 0.0]) @ R.T + np.diag([0.0, 0.0, 1e-12])
    assert principal_tilt_deg(C[None, ...])[0] == pytest.approx(45.0, abs=1e-6)
    major = ellipsoid_axes(C[None, ...])[0][0, -1]
    per_axis_worst = sigma_per_axis(C[None, ...])[0].max()
    assert per_axis_worst == pytest.approx(major / np.sqrt(2.0), rel=1e-6)


def test_non_square_input_is_refused() -> None:
    with pytest.raises(ValueError, match="covariances"):
        sigma_per_axis(np.zeros((4, 3)))


# -------------------------------------------------------------------- figures


@pytest.fixture(scope="module")
def figures() -> object:
    """The figure module, with a headless backend forced before pyplot loads."""
    mpl = pytest.importorskip("matplotlib", reason="figures need the [viz] extra")
    mpl.use("Agg")
    import erp.viz.figures as f

    return f


def _three_way_inputs() -> dict[str, np.ndarray]:
    t = np.linspace(0.0, 3.0, 200)
    tm = np.linspace(0.0, 3.0, 60)
    return {
        "t_truth": t, "y_truth": np.column_stack([np.sin(t), np.cos(t)]),
        "t_meas": tm, "y_meas": np.column_stack([np.sin(tm), np.cos(tm)]),
        "t_est": t, "y_est": np.column_stack([np.sin(t), np.cos(t)]),
    }


def test_three_way_draws_one_row_per_channel(figures) -> None:  # type: ignore[no-untyped-def]
    fig = figures.three_way(
        **_three_way_inputs(), labels=["link1_acc_x", "link1_gyro_z"],
        ylabel="m/s^2, site frame",
    )
    assert len(fig.axes) == 2
    figures.plt.close(fig)


def test_three_way_works_without_truth(figures) -> None:  # type: ignore[no-untyped-def]
    """The real-hardware case: there is no plant to be right about."""
    kw = _three_way_inputs()
    kw["y_truth"] = np.empty((0, 2))
    fig = figures.three_way(**kw, labels=["a", "b"], ylabel="m/s^2")
    lines = [ln.get_label() for ln in fig.axes[0].get_lines()]
    assert "plant (truth)" not in lines
    assert "EKF estimate" in lines
    figures.plt.close(fig)


def test_sensor_compare_draws_the_target_line(figures) -> None:  # type: ignore[no-untyped-def]
    """A NIS trace without its target is unreadable: the whole question is the ratio."""
    t = np.linspace(0.0, 3.0, 60)
    fig = figures.sensor_compare(t, np.full(60, 12.0), nis_target=12.0)
    assert len(fig.axes) == 1
    assert any("target" in (ln.get_label() or "") for ln in fig.axes[0].get_lines())
    figures.plt.close(fig)

    # With a residual it grows a second panel, labelled by the caller so a
    # posterior residual is never silently presented as the innovation.
    fig = figures.sensor_compare(
        t, np.full(60, 12.0), nis_target=12.0,
        residual=np.zeros((60, 12)), residual_label="posterior residual",
    )
    assert len(fig.axes) == 2
    assert fig.axes[0].get_ylabel() == "posterior residual"
    figures.plt.close(fig)


def test_effector_band_draws_three_axes_and_a_band(figures) -> None:  # type: ignore[no-untyped-def]
    t = np.linspace(0.0, 3.0, 100)
    fig = figures.effector_band(
        t, np.zeros((100, 3)), np.tile(np.eye(3) * 1e-6, (100, 1, 1)),
        p_true=np.zeros((100, 3)),
    )
    assert len(fig.axes) == 3
    assert fig.axes[0].collections, "the sigma band should be a filled collection"
    figures.plt.close(fig)
