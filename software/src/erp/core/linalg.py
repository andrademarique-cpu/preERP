"""Numerical helpers shared by estimators and consistency tests.

Pure linear algebra: no hardware imports, no model assumptions.
"""

from __future__ import annotations

import numpy as np

from erp.core.types import Array

__all__ = ["mahalanobis", "make_spd", "nees", "nis"]


def make_spd(P: Array, rel: float = 1e-12) -> Array:
    """Symmetrise ``P`` and floor its eigenvalues at ``rel`` times the largest.

    This is **not** cosmetic. A servoed-joint Jacobian carries entries of order
    700, and ``P`` spans roughly ten orders of magnitude between its position
    and velocity blocks, so ``F P F^T`` loses symmetry and positive
    definiteness in floating point. Without the eigenvalue floor the NEES comes
    out *negative* and the consistency diagnostic silently stops meaning
    anything -- see ``docs/theory/finger_imu_ekf.md`` section 3.6.

    Parameters
    ----------
    P:
        Covariance candidate, shape ``(n, n)``.
    rel:
        Eigenvalue floor relative to the largest eigenvalue.

    Returns
    -------
    A symmetric positive-definite matrix of the same shape.
    """
    P = 0.5 * (P + P.T)
    w, V = np.linalg.eigh(P)
    floor = max(float(w.max()), 0.0) * rel + 1e-300
    w = np.maximum(w, floor)
    return np.asarray((V * w) @ V.T, dtype=np.float64)


def mahalanobis(e: Array, S: Array) -> float:
    """Return ``e^T S^-1 e``.

    Uses :func:`numpy.linalg.solve` rather than forming ``S^-1``: innovation
    covariances here reach condition numbers around 1e11 near a stretched
    finger pose, where an explicit inverse loses most of its digits.

    Parameters
    ----------
    e:
        Error or innovation vector, shape ``(m,)``.
    S:
        Its covariance, shape ``(m, m)``, same units squared.
    """
    return float(e @ np.linalg.solve(S, e))


def nees(x_true: Array, x_est: Array, P: Array) -> float:
    """Normalised Estimation Error Squared, ``(x_true - x_est)^T P^-1 (...)``.

    For a consistent filter this is chi-squared distributed with ``dim(x)``
    degrees of freedom, so its expectation equals ``dim(x)``. Report the
    *median* for a single run: the distribution is heavy-tailed, and a serious
    ANEES needs Monte Carlo over independent seeds.

    Requires ground truth, so it is a simulation- and replay-only diagnostic.
    """
    return mahalanobis(x_true - x_est, P)


def nis(innovation: Array, S: Array) -> float:
    """Normalised Innovation Squared, ``y^T S^-1 y``.

    Chi-squared with ``dim(y)`` degrees of freedom when the filter is
    consistent. Unlike :func:`nees` this needs no ground truth, so it is the
    diagnostic that survives onto real hardware -- but note that it is
    insensitive to a whole class of failures: inflating ``R`` flattens NIS
    while leaving NEES untouched.
    """
    return mahalanobis(innovation, S)
