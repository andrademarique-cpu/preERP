# software/src/erp/core/linalg.py
"""Numerical helpers shared by every estimator. Pure numpy, no hardware imports."""
import numpy as np


def make_spd(P: np.ndarray, rel: float = 1e-12) -> np.ndarray:
    """Symmetrizes and clamps eigenvalues to ensure positive semi-definiteness.

    Call after every operation on P: round-off makes it asymmetric and pushes
    small eigenvalues below zero, and then NEES comes out negative.
    """
    P = 0.5 * (P + P.T)
    w, V = np.linalg.eigh(P)
    w = np.maximum(w, max(w.max(), 0.0) * rel + 1e-300)
    return (V * w) @ V.T


def nees_of(error: np.ndarray, covariance: np.ndarray) -> float:
    """e^T P^-1 e for a single sample. solve, not inv: P is ill-conditioned."""
    return float(error @ np.linalg.solve(covariance, error))
