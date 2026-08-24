"""Exact discretisation of continuous-time linear models over arbitrary steps.

Under multi-rate fusion the prediction interval is dictated by the sensor
schedule, so both the input response and the process noise must be computed
for whatever ``dt`` comes up. Everything here is exact for a zero-order-hold
input, which is what the actuators actually apply.

See ``docs/adr/0001-multi-rate-fusion.md`` decision D3.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from erp.core.types import Array

__all__ = ["van_loan", "zoh_input"]


def _as2d(name: str, M: Array) -> Array:
    A = np.asarray(M, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {A.shape}")
    return A


def van_loan(A_c: Array, G: Array, q: Array, dt: float) -> tuple[Array, Array]:
    """Discretise ``dx = A_c x dt + G dbeta`` over ``dt``.

    Van Loan's method: build the block matrix

    .. code-block:: text

        M = [[-A_c, G q G^T],
             [    0,   A_c^T]] * dt

    exponentiate once, then read ``F = E[1,1]^T`` and ``Q = F @ E[0,1]``.

    The resulting ``Q`` is **schedule invariant**: propagating across ``dt1``
    then ``dt2`` gives exactly the covariance of one ``dt1 + dt2`` step. The
    discrete white-noise alternative does not, which under multi-rate operation
    would make the effective process noise a function of the sensor schedule
    rather than of the physics.

    Parameters
    ----------
    A_c:
        Continuous-time system matrix, shape ``(n, n)``, units 1/s.
    G:
        Noise input matrix, shape ``(n, k)``, mapping disturbance channels onto
        state derivatives.
    q:
        Power spectral density of the disturbance, shape ``(k, k)``. For an
        angular-acceleration disturbance the units are rad^2/s^3 -- *not* the
        rad/s^2 of a per-step DWNA sigma. Converting between the two is a
        re-derivation, not a unit change.
    dt:
        Interval length in seconds, strictly positive.

    Returns
    -------
    ``(F, Q)``: discrete state transition matrix and accumulated process-noise
    covariance over ``dt``.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    A_c, G, q = _as2d("A_c", A_c), _as2d("G", G), _as2d("q", q)
    n = A_c.shape[0]
    if A_c.shape != (n, n):
        raise ValueError(f"A_c must be square, got {A_c.shape}")
    if G.shape[0] != n:
        raise ValueError(f"G must have {n} rows to match A_c, got {G.shape}")
    if q.shape != (G.shape[1], G.shape[1]):
        raise ValueError(f"q must be ({G.shape[1]}, {G.shape[1]}), got {q.shape}")

    M = np.zeros((2 * n, 2 * n), dtype=np.float64)
    M[:n, :n] = -A_c
    M[:n, n:] = G @ q @ G.T
    M[n:, n:] = A_c.T
    E = np.asarray(expm(M * dt), dtype=np.float64)

    F = np.ascontiguousarray(E[n:, n:].T)
    Q = F @ E[:n, n:]
    return F, np.asarray(0.5 * (Q + Q.T), dtype=np.float64)


def zoh_input(A_c: Array, B_c: Array, dt: float) -> Array:
    """Return the discrete input matrix for a zero-order-held ``u``.

    Computes ``B_d = integral_0^dt exp(A_c s) ds @ B_c`` via the standard
    augmented-matrix trick, so that ``x[k+1] = F x[k] + B_d u`` is exact for
    ``u`` constant across the interval.

    That constancy is guaranteed by the fusion layer, which splits every
    prediction at input breakpoints -- which is what makes this exact rather
    than a first-order approximation.

    Parameters
    ----------
    A_c:
        Continuous-time system matrix, shape ``(n, n)``, units 1/s.
    B_c:
        Continuous-time input matrix, shape ``(n, m)``.
    dt:
        Interval length in seconds, strictly positive.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    A_c, B_c = _as2d("A_c", A_c), _as2d("B_c", B_c)
    n, m = A_c.shape[0], B_c.shape[1]
    if A_c.shape != (n, n):
        raise ValueError(f"A_c must be square, got {A_c.shape}")
    if B_c.shape[0] != n:
        raise ValueError(f"B_c must have {n} rows to match A_c, got {B_c.shape}")

    M = np.zeros((n + m, n + m), dtype=np.float64)
    M[:n, :n] = A_c
    M[:n, n:] = B_c
    E = np.asarray(expm(M * dt), dtype=np.float64)
    return np.ascontiguousarray(E[:n, n:])
