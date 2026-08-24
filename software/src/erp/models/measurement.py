"""Linear measurement models for the servoed-finger state layout.

All of these assume the blocked state ``[q (rad), v (rad/s), a (rad)]`` built
by :func:`erp.models.linear.servoed_finger_model`.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from erp.core.types import Array
from erp.models.base import MeasurementModel
from erp.models.linear import JointParams

__all__ = ["LinearMeasurementModel", "joint_accel_model", "joint_block_model"]

Block = Literal["angle", "rate", "activation"]


class LinearMeasurementModel(MeasurementModel):
    """``z = H x + D u``, with optional direct feedthrough.

    ``D`` is the term that makes the ``h(x, u)`` signature necessary. It is
    zero for any sensor reading a quantity that is itself a state -- encoders,
    gyros -- and non-zero for a sensor reading something the input drives
    within the same instant, which is the accelerometer case whenever the
    actuator's own dynamics are *not* carried in the state.
    """

    def __init__(self, H: Array, R: Array, D: Array | None = None) -> None:
        """
        Parameters
        ----------
        H:
            Observation matrix, shape ``(m, n)``.
        R:
            Noise covariance, shape ``(m, m)``, units of the reading squared.
        D:
            Direct feedthrough, shape ``(m, p)``. ``None`` means no
            instantaneous dependence on the input.
        """
        self._H = np.asarray(H, dtype=np.float64)
        self._R = np.asarray(R, dtype=np.float64)
        self._D = None if D is None else np.asarray(D, dtype=np.float64)
        m = self._H.shape[0]
        if self._R.shape != (m, m):
            raise ValueError(f"R must be ({m}, {m}) to match H, got {self._R.shape}")
        if self._D is not None and self._D.shape[0] != m:
            raise ValueError(f"D must have {m} rows to match H, got {self._D.shape}")

    @property
    def dim(self) -> int:
        """Number of measurement channels."""
        return int(self._H.shape[0])

    def h(self, x: Array, u: Array) -> Array:
        """Return the expected reading in state ``x`` under input ``u``."""
        z = self._H @ x
        if self._D is not None:
            z = z + self._D @ u
        return np.asarray(z, dtype=np.float64)

    def jacobian(self, x: Array, u: Array) -> Array:
        """Return ``H``; constant for a linear model."""
        return self._H

    @property
    def R(self) -> Array:
        """Noise covariance, units of the reading squared."""
        return self._R


def joint_block_model(n_joints: int, block: Block, sigma: float) -> LinearMeasurementModel:
    """Observe one whole block of the servoed-finger state directly.

    Parameters
    ----------
    n_joints:
        Number of joints, so the state has ``3 * n_joints`` entries.
    block:
        Which block to observe. ``"angle"`` models a joint encoder (rad),
        ``"rate"`` a rate gyro measuring joint angular velocity (rad/s), and
        ``"activation"`` a servo position feedback channel (rad).
    sigma:
        Per-channel noise standard deviation, in the block's units.

    Notes
    -----
    ``D`` is zero for all three: each reads a quantity that is already a state,
    so the input has no instantaneous path into the measurement.
    """
    if n_joints < 1:
        raise ValueError(f"n_joints must be >= 1, got {n_joints}")
    offset = {"angle": 0, "rate": 1, "activation": 2}[block]
    n = n_joints
    H = np.zeros((n, 3 * n), dtype=np.float64)
    H[:, offset * n : (offset + 1) * n] = np.eye(n)
    return LinearMeasurementModel(H, sigma**2 * np.eye(n))


def joint_accel_model(joints: Sequence[JointParams], sigma: float) -> LinearMeasurementModel:
    """Observe joint angular acceleration, as an IMU on the link would.

    The reading is the second row-block of the continuous dynamics::

        qddot = (kp (a - q) - (kv + damping) v) / inertia

    Parameters
    ----------
    joints:
        Same sequence used to build the process model, ordered proximal to
        distal.
    sigma:
        Per-channel noise standard deviation, rad/s^2.

    Notes
    -----
    ``D`` is zero **because the servo activation is carried in the state**: the
    input reaches acceleration only through ``a``, never within the same
    instant. Drop the activation from the state and this same sensor acquires a
    non-zero ``D``, along with the deterministic bias that motivates carrying
    it -- which is precisely the matched/mismatched contrast measured in
    ``docs/theory/finger_imu_ekf.md`` section 4.
    """
    n = len(joints)
    if n == 0:
        raise ValueError("need at least one joint")
    H = np.zeros((n, 3 * n), dtype=np.float64)
    for i, jp in enumerate(joints):
        H[i, i] = -jp.kp / jp.inertia
        H[i, n + i] = -(jp.kv + jp.damping) / jp.inertia
        H[i, 2 * n + i] = jp.kp / jp.inertia
    return LinearMeasurementModel(H, sigma**2 * np.eye(n))
