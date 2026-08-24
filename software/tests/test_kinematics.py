"""Fingertip kinematics and error propagation to the tip."""

from __future__ import annotations

import numpy as np
import pytest

from erp.models.kinematics import FingerGeometry, fk_tip, fk_tip_jacobian, tip_covariance
from erp.viz.ellipse import covariance_ellipse, ellipse_axes

GEOM = FingerGeometry(l0=0.030, l1=0.035, l2=0.030)


def test_straight_finger_stacks_the_links() -> None:
    p = fk_tip(np.zeros(2), GEOM)
    assert p[0] == 0.0
    assert p[1] == pytest.approx(0.0)
    assert p[2] == pytest.approx(GEOM.l0 + GEOM.l1 + GEOM.l2)


def test_motion_stays_in_the_yz_plane() -> None:
    """Both hinges turn about world x, so x is identically zero."""
    rng = np.random.default_rng(0)
    for q in rng.uniform(-np.pi, np.pi, size=(20, 2)):
        assert fk_tip(q, GEOM)[0] == 0.0


def test_tip_stays_within_reach() -> None:
    rng = np.random.default_rng(1)
    reach = GEOM.l1 + GEOM.l2
    for q in rng.uniform(-np.pi, np.pi, size=(50, 2)):
        p = fk_tip(q, GEOM)
        radial = np.hypot(p[1], p[2] - GEOM.l0)
        assert radial <= reach + 1e-12


@pytest.mark.parametrize("q", [[0.0, 0.0], [0.3, -0.5], [-1.1, 0.9], [1.5, 1.2]])
def test_jacobian_matches_finite_differences(q: list[float]) -> None:
    """The analytic Jacobian is the derivative, checked against the definition."""
    qa = np.array(q)
    J = fk_tip_jacobian(qa, GEOM)
    eps = 1e-7
    for j in range(2):
        step = np.zeros(2)
        step[j] = eps
        fd = (fk_tip(qa + step, GEOM) - fk_tip(qa - step, GEOM))[1:] / (2 * eps)
        assert np.allclose(J[:, j], fd, rtol=1e-6, atol=1e-9)


def test_tip_covariance_is_a_change_of_coordinates() -> None:
    """Propagation adds no uncertainty of its own.

    J is invertible away from singular poses, so the Mahalanobis distance of a
    given angular error must be identical in tip space. This is the sharpest
    check available here: if it holds, a coverage shortfall is a diagnosis of
    the *filter*, not of the kinematics.
    """
    q = np.array([0.4, -0.7])
    P_qq = np.array([[4e-6, -1.2e-6], [-1.2e-6, 9e-6]])
    dq = np.array([1.3e-3, -0.8e-3])

    J = fk_tip_jacobian(q, GEOM)
    Sigma = tip_covariance(q, P_qq, GEOM)
    d_angle = dq @ np.linalg.solve(P_qq, dq)
    d_tip = (J @ dq) @ np.linalg.solve(Sigma, J @ dq)
    assert d_tip == pytest.approx(d_angle, rel=1e-9)


def test_dropping_the_cross_term_hides_the_error_in_the_area() -> None:
    """Why the whole P_qq is propagated, not its diagonal.

    The area barely moves -- it is off by 1/sqrt(1 - rho^2) -- while the
    worst-direction sigma is materially wrong. Anyone checking the area alone
    would conclude the diagonal was fine.
    """
    q = np.array([0.4, -0.7])
    sd = np.array([2e-3, 3e-3])
    rho = -0.30
    P_full = np.array([
        [sd[0] ** 2, rho * sd[0] * sd[1]],
        [rho * sd[0] * sd[1], sd[1] ** 2],
    ])
    P_diag = np.diag(np.diag(P_full))

    full, diag = tip_covariance(q, P_full, GEOM), tip_covariance(q, P_diag, GEOM)
    area_ratio = np.sqrt(np.linalg.det(diag) / np.linalg.det(full))
    assert area_ratio == pytest.approx(1.0 / np.sqrt(1 - rho**2), rel=1e-9)
    assert area_ratio < 1.06, "area is nearly unchanged -- which is the trap"

    worst_ratio = np.sqrt(np.linalg.eigvalsh(diag).max() / np.linalg.eigvalsh(full).max())
    assert worst_ratio > 1.1, "worst-direction sigma should be visibly wrong"


def test_tip_covariance_rejects_a_full_state_covariance() -> None:
    """Only the angle block may be propagated; the tip does not depend on rate."""
    with pytest.raises(ValueError, match=r"\(2, 2\) angle block"):
        tip_covariance(np.zeros(2), np.eye(6), GEOM)


def test_ellipse_axes_recover_a_known_covariance() -> None:
    cov = np.diag([4e-6, 1e-6])                 # sigma 2 mm and 1 mm, axis-aligned
    major, minor, angle = ellipse_axes(cov, n_sigma=1.0)
    assert major == pytest.approx(2e-3)
    assert minor == pytest.approx(1e-3)
    assert np.cos(angle) ** 2 == pytest.approx(1.0)   # major axis along x


def test_ellipse_points_lie_on_the_contour() -> None:
    """Every returned point must sit at Mahalanobis distance n_sigma."""
    cov = np.array([[4e-6, 1.5e-6], [1.5e-6, 9e-6]])
    center = np.array([0.01, -0.02])
    pts = covariance_ellipse(cov, center, n_sigma=2.0, n_points=32)
    assert pts.shape == (32, 2)
    d = np.array([(p - center) @ np.linalg.solve(cov, p - center) for p in pts])
    assert np.allclose(np.sqrt(d), 2.0, rtol=1e-9)
    assert np.allclose(pts[0], pts[-1])          # closed path


def test_ellipse_survives_a_near_degenerate_covariance() -> None:
    """Near a stretched pose the ellipse collapses onto a line; it must not crash."""
    q = np.array([1e-4, 1e-4])                   # both columns of J nearly parallel
    P_qq = np.eye(2) * 1e-8
    Sigma = tip_covariance(q, P_qq, GEOM)
    major, minor, _ = ellipse_axes(Sigma, n_sigma=2.0)
    assert major >= minor >= 0.0
    assert np.all(np.isfinite(covariance_ellipse(Sigma, (0.0, 0.0))))
