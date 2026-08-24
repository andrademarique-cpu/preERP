"""Linear-Gaussian process models with exact variable-step discretisation.

State layout throughout is **blocked, not interleaved** -- ``[q..., v..., a...]``
-- matching the convention used in ``notebooks/finger_imu_practice.ipynb`` so
that state vectors stay comparable between the notebook and the package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from erp.core.types import Array
from erp.models.base import ProcessModel
from erp.models.discretize import van_loan, zoh_input

__all__ = ["ConstantAccelModel", "JointParams", "LinearTimeInvariantModel", "servoed_finger_model"]

_CACHE_LIMIT = 64


class LinearTimeInvariantModel(ProcessModel):
    """``dx/dt = A_c x + B_c u + G w``, discretised exactly for any ``dt``.

    ``w`` is continuous white noise with power spectral density ``q``. Both the
    input response and the process noise are computed per interval, so the
    model is correct at whatever step the sensor schedule produces and its
    ``Q`` composes exactly across a split interval.
    """

    def __init__(self, A_c: Array, B_c: Array, G: Array, q: Array, *, cache: bool = True) -> None:
        """
        Parameters
        ----------
        A_c:
            Continuous system matrix, shape ``(n, n)``, units 1/s.
        B_c:
            Continuous input matrix, shape ``(n, m)``.
        G:
            Noise input matrix, shape ``(n, k)``.
        q:
            Disturbance power spectral density, shape ``(k, k)``. For an
            angular-acceleration disturbance, units rad^2/s^3.
        cache:
            Memoise the discretisation per distinct ``dt``. Sensor rates are
            regular, so in a realtime loop this collapses to a handful of
            matrix exponentials regardless of how long the session runs.
        """
        self._A_c = np.asarray(A_c, dtype=np.float64)
        self._B_c = np.asarray(B_c, dtype=np.float64)
        self._G = np.asarray(G, dtype=np.float64)
        self._q = np.asarray(q, dtype=np.float64)
        self._n = int(self._A_c.shape[0])
        if self._B_c.shape[0] != self._n:
            raise ValueError(f"B_c must have {self._n} rows, got {self._B_c.shape}")
        self._cache: dict[float, tuple[Array, Array, Array]] | None = {} if cache else None

    @property
    def dim(self) -> int:
        """Number of state dimensions."""
        return self._n

    @property
    def input_dim(self) -> int:
        """Number of input dimensions."""
        return int(self._B_c.shape[1])

    def _discretise(self, dt: float) -> tuple[Array, Array, Array]:
        """Return ``(F, B_d, Q)`` for an interval of ``dt`` seconds."""
        if self._cache is not None and dt in self._cache:
            return self._cache[dt]
        F, Q = van_loan(self._A_c, self._G, self._q, dt)
        B_d = zoh_input(self._A_c, self._B_c, dt)
        for M in (F, B_d, Q):
            M.flags.writeable = False  # cached matrices are handed out by reference
        if self._cache is not None:
            if len(self._cache) >= _CACHE_LIMIT:
                self._cache.clear()
            self._cache[dt] = (F, B_d, Q)
        return F, B_d, Q

    def predict(self, x: Array, u: Array, dt: float) -> Array:
        """Propagate the mean, exact for ``u`` held constant over ``dt``."""
        F, B_d, _ = self._discretise(dt)
        return np.asarray(F @ x + B_d @ u, dtype=np.float64)

    def jacobian(self, x: Array, u: Array, dt: float) -> Array:
        """Return the discrete transition matrix; independent of ``x`` and ``u``."""
        F, _, _ = self._discretise(dt)
        return F

    def Q(self, dt: float) -> Array:
        """Return process noise accumulated over ``dt``; composes exactly."""
        _, _, Q = self._discretise(dt)
        return Q


@dataclass(frozen=True)
class JointParams:
    """Servoed revolute joint, as modelled in ``notebooks/assets/finger_2link.xml``.

    The servo closes a PD loop internally on a commanded *angle*, and its
    output passes through a first-order lag before reaching the link:
    ``force = kp (a - q) - kv v``, with ``a`` the servo activation.

    Attributes
    ----------
    kp:
        Servo proportional gain, N*m/rad.
    kv:
        Servo velocity gain, N*m*s/rad.
    damping:
        Viscous damping seen by the joint, N*m*s/rad: structural damping plus
        gearbox friction, summed.
    inertia:
        Effective rotational inertia about the joint axis, kg*m^2. Includes
        reflected rotor inertia, which in a high-ratio servo dominates the
        link's own inertia.
    tau:
        Servo activation time constant, seconds. The command does not step; it
        passes through a first-order lag of this width.
    psd_alpha:
        Angular-acceleration disturbance PSD, rad^2/s^3. This is the
        continuous-time counterpart of the notebook's per-step ``SIG_ALPHA``
        and must be re-derived rather than copied across.
    psd_act:
        Activation jitter PSD, rad^2/s. Models servo command noise.
    """

    kp: float
    kv: float
    damping: float
    inertia: float
    tau: float
    psd_alpha: float
    psd_act: float = 0.0


def servoed_finger_model(joints: Sequence[JointParams]) -> LinearTimeInvariantModel:
    """Build the process model for a finger of independently servoed joints.

    State is ``[q (rad), v (rad/s), a (rad)]`` in blocks, of size ``3 * n``.

    Carrying the servo activation ``a`` as state is not optional. With the
    actuator lag left out, the accelerometer channels pick up a deterministic,
    time-correlated bias that no amount of ``Q`` or ``R`` inflation removes:
    NEES median 344 281 against a target of 4, versus 5.96 against 6 once the
    activation is modelled (``docs/theory/finger_imu_ekf.md`` section 4).

    Continuous dynamics, per joint::

        q' = v
        v' = (kp (a - q) - (kv + damping) v) / inertia
        a' = (u - a) / tau

    The ``a`` block is a first-order lag, exactly integrable at any step, so
    the augmented state costs nothing in variable-step accuracy.

    Parameters
    ----------
    joints:
        One :class:`JointParams` per joint, ordered proximal to distal.
    """
    n = len(joints)
    if n == 0:
        raise ValueError("need at least one joint")

    A_c = np.zeros((3 * n, 3 * n), dtype=np.float64)
    B_c = np.zeros((3 * n, n), dtype=np.float64)
    G = np.zeros((3 * n, 2 * n), dtype=np.float64)
    q = np.zeros((2 * n, 2 * n), dtype=np.float64)

    for i, jp in enumerate(joints):
        if jp.inertia <= 0.0 or jp.tau <= 0.0:
            raise ValueError(f"joint {i}: inertia and tau must be > 0")
        iq, iv, ia = i, n + i, 2 * n + i
        A_c[iq, iv] = 1.0
        A_c[iv, iq] = -jp.kp / jp.inertia
        A_c[iv, iv] = -(jp.kv + jp.damping) / jp.inertia
        A_c[iv, ia] = jp.kp / jp.inertia
        A_c[ia, ia] = -1.0 / jp.tau
        B_c[ia, i] = 1.0 / jp.tau
        G[iv, i] = 1.0           # angular-acceleration disturbance on the rate
        G[ia, n + i] = 1.0       # activation jitter
        q[i, i] = jp.psd_alpha
        q[n + i, n + i] = jp.psd_act

    return LinearTimeInvariantModel(A_c, B_c, G, q)


class ConstantAccelModel(LinearTimeInvariantModel):
    """Per-axis double integrator driven by white acceleration noise.

    State is ``[p, v]`` in blocks, size ``2 * n``. The input matrix is zero:
    this model describes motion that is *not* commanded, so ``u`` is accepted
    and ignored. Useful as the minimal well-understood stub for exercising the
    :class:`~erp.models.base.ProcessModel` interface.
    """

    def __init__(self, n_axes: int, psd: float) -> None:
        """
        Parameters
        ----------
        n_axes:
            Number of independent axes.
        psd:
            Acceleration disturbance PSD, units of (position units)^2/s^3.
        """
        if n_axes < 1:
            raise ValueError(f"n_axes must be >= 1, got {n_axes}")
        n = n_axes
        A_c = np.zeros((2 * n, 2 * n), dtype=np.float64)
        A_c[:n, n:] = np.eye(n)
        G = np.zeros((2 * n, n), dtype=np.float64)
        G[n:, :] = np.eye(n)
        super().__init__(A_c, np.zeros((2 * n, 1)), G, psd * np.eye(n))
