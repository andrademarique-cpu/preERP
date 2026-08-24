"""Forward kinematics of the two-phalanx finger, and error propagation to the tip.

The estimator works in joint angles; what the rest of the system wants is where
the fingertip is, *with its uncertainty*. A position without a trustworthy
covariance is not an estimate, it is a guess.

Ported from ``notebooks/finger_imu_practice.ipynb`` section 5, where the
kinematics were validated against MuJoCo's own ``framepos`` sensor and
``mj_jacSite`` to 2.8e-17 m. No hardware imports: this must stay runnable with
no simulator attached.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from erp.core.types import Array

__all__ = ["FingerGeometry", "fk_tip", "fk_tip_jacobian", "tip_covariance"]


@dataclass(frozen=True)
class FingerGeometry:
    """Link lengths of the planar two-phalanx finger, metres.

    Attributes
    ----------
    l0:
        Base segment, from the world origin to joint 1, along +z.
    l1:
        Proximal phalanx, joint 1 to joint 2.
    l2:
        Distal phalanx, joint 2 to the fingertip.
    """

    l0: float = 0.030
    l1: float = 0.035
    l2: float = 0.030

    def __post_init__(self) -> None:
        if min(self.l0, self.l1, self.l2) <= 0.0:
            raise ValueError("link lengths must be positive")


def fk_tip(q: Array, geom: FingerGeometry) -> Array:
    """Fingertip position in the world frame, metres.

    Both joints are hinges about the world x axis, so the finger moves in the
    y-z plane and ``x`` is identically zero:

    .. code-block:: text

        y = -(l1 sin q1 + l2 sin(q1 + q2))
        z =  l0 + l1 cos q1 + l2 cos(q1 + q2)

    Parameters
    ----------
    q:
        ``[q1, q2]`` joint angles in radians. Only the first two entries are
        read, so a full state vector's angle block can be passed directly.
    geom:
        Link lengths.

    Returns
    -------
    ``[x, y, z]`` in metres, world frame. ``x`` is exactly 0 by construction.
    """
    q1, q2 = float(q[0]), float(q[1])
    return np.array(
        [
            0.0,
            -(geom.l1 * np.sin(q1) + geom.l2 * np.sin(q1 + q2)),
            geom.l0 + geom.l1 * np.cos(q1) + geom.l2 * np.cos(q1 + q2),
        ]
    )


def fk_tip_jacobian(q: Array, geom: FingerGeometry) -> Array:
    """Tip Jacobian restricted to the plane of motion, shape ``(2, 2)``, m/rad.

    Rows are the ``(y, z)`` components of the tip, columns are ``(q1, q2)``.
    The ``x`` row is omitted because it is identically zero, not because it is
    being neglected.
    """
    q1, q2 = float(q[0]), float(q[1])
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    return np.array(
        [
            [-geom.l1 * c1 - geom.l2 * c12, -geom.l2 * c12],
            [-geom.l1 * s1 - geom.l2 * s12, -geom.l2 * s12],
        ]
    )


def tip_covariance(q: Array, P_qq: Array, geom: FingerGeometry) -> Array:
    """Propagate angular uncertainty to the tip: ``Sigma = J P_qq J^T``.

    Parameters
    ----------
    q:
        Estimated joint angles, radians.
    P_qq:
        The ``(2, 2)`` **angle block** of the estimator covariance, rad^2.
    geom:
        Link lengths.

    Returns
    -------
    ``(2, 2)`` tip position covariance in the ``(y, z)`` plane, m^2.

    Notes
    -----
    Two things this gets right that are easy to get wrong.

    *Only the angle block enters.* The tip is a function of ``q`` alone, so
    propagating the full state covariance would be incorrect, not merely
    wasteful.

    *The whole block, never just its diagonal.* The same sensors inform both
    angles, so their errors are correlated. Discarding the cross-term barely
    moves the ellipse *area* -- a factor of 1/sqrt(1 - rho^2), about 1.05 at
    the measured rho of -0.30 -- while rotating the ellipse and inflating the
    worst-direction sigma by ~1.19x. Looking at the area therefore hides the
    error. Keeping ``sqrt(diag(P))`` and calling it the uncertainty is exactly
    this mistake.
    """
    J = fk_tip_jacobian(q, geom)
    P = np.asarray(P_qq, dtype=np.float64)
    if P.shape != (2, 2):
        raise ValueError(f"P_qq must be the (2, 2) angle block, got {P.shape}")
    return np.asarray(J @ P @ J.T, dtype=np.float64)
