"""Error-ellipse geometry. Pure numpy, no plotting backend.

Split out from the figure code so the part that can be wrong *numerically* can
be tested in the fast suite, with no matplotlib and no display. A figure module
is hard to test and easy to eyeball; this is the opposite, and the arithmetic
here is where a covariance actually gets misread.

All inputs are full covariance blocks, never per-axis sigmas. Over the golden
run the effector ellipsoid's major axis sits a median 24.9 deg (max 45.0) off
the nearest world axis, so the axis-aligned view understates the worst direction
by a median 9.6% and up to 39.7% -- :func:`principal_tilt_deg` is what
reproduces that number rather than asking anyone to trust it.
"""

from __future__ import annotations

import numpy as np

from erp.core.types import Array

__all__ = ["ellipsoid_axes", "principal_tilt_deg", "sigma_per_axis"]


def _as_cov_stack(C: Array) -> Array:
    Cs = np.asarray(C, dtype=np.float64)
    if Cs.ndim == 2:
        Cs = Cs[None, ...]
    if Cs.ndim != 3 or Cs.shape[1] != Cs.shape[2]:
        raise ValueError(f"expected (n, d, d) or (d, d) covariances, got {Cs.shape}")
    return Cs


def sigma_per_axis(C: Array) -> Array:
    """(n, d) one-sigma along each world axis, i.e. ``sqrt(diag(C))``.

    Units: the square root of ``C``'s, so m for an m^2 position covariance.

    This is the *axis-aligned* view and it is the one the frozen fixture stores.
    It is correct for a band on a single axis and wrong for "how far off can the
    estimate be", because it drops the cross terms -- see
    :func:`principal_tilt_deg`.
    """
    return np.asarray(np.sqrt(np.einsum("nii->ni", _as_cov_stack(C))), dtype=np.float64)


def ellipsoid_axes(C: Array) -> tuple[Array, Array]:
    """Semi-axis lengths and orientation of the one-sigma error ellipsoid.

    ``C`` is symmetrised before the eigendecomposition, because a covariance
    that has been through the filter is symmetric only to round-off and
    ``eigh`` reads the lower triangle alone -- feeding it an unsymmetrised block
    silently uses half the matrix.

    Negative eigenvalues are clamped to zero rather than raising: a covariance
    can come back with a tiny negative eigenvalue from round-off, and refusing
    to draw an ellipse over -1e-19 would be worse than drawing a flat one. A
    *large* negative eigenvalue means the caller skipped ``make_spd``, which is
    the real bug and shows up as a degenerate ellipse.

    -> (n, d) semi-axis lengths ascending, (n, d, d) whose COLUMNS are the
    corresponding unit axis directions in world coordinates.
    """
    Cs = _as_cov_stack(C)
    sym = 0.5 * (Cs + np.transpose(Cs, (0, 2, 1)))
    w, V = np.linalg.eigh(sym)
    return np.asarray(np.sqrt(np.maximum(w, 0.0)), dtype=np.float64), np.asarray(V, np.float64)


def principal_tilt_deg(C: Array) -> Array:
    """(n,) angle in degrees between the major axis and the nearest world axis.

    Zero means the ellipsoid is axis-aligned and per-axis sigmas tell the whole
    story. The maximum for three dimensions is 54.7 deg, the body diagonal.

    This is the diagnostic behind "propagate the full covariance block": if this
    is not near zero, ``sqrt(diag(C))`` is not a description of the error.
    """
    _, V = ellipsoid_axes(C)
    major = V[..., -1]                     # eigenvector of the largest eigenvalue
    best = np.max(np.abs(major), axis=-1)  # cosine to the closest world axis
    return np.asarray(np.rad2deg(np.arccos(np.clip(best, 0.0, 1.0))), dtype=np.float64)
