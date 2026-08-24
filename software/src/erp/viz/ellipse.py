"""Confidence-ellipse geometry for 2-D covariances.

Pure geometry, no plotting backend: returns points and axes so the caller can
draw them with whatever it already has open.
"""

from __future__ import annotations

import numpy as np

from erp.core.types import Array

__all__ = ["covariance_ellipse", "ellipse_axes"]


def ellipse_axes(cov: Array, n_sigma: float = 2.0) -> tuple[float, float, float]:
    """Return ``(semi_major, semi_minor, angle_rad)`` of a confidence ellipse.

    Parameters
    ----------
    cov:
        ``(2, 2)`` covariance, units of length squared.
    n_sigma:
        Scale factor. 1.0 gives the 39 % contour in 2-D, 2.0 gives 86 %; the
        familiar 95 % contour is ``n_sigma = 2.4477``. Two dimensions do not
        inherit the 1-D 68/95 numbers, and quoting them here is a common way
        to overstate coverage.

    Returns
    -------
    Semi-axis lengths in the same units as ``sqrt(cov)``, and the angle of the
    major axis measured from the +x axis, radians.

    Notes
    -----
    Reported separately from the point list because the *aspect ratio* is the
    diagnostic worth reading. Near a stretched finger the two Jacobian columns
    become parallel, both joints push the tip the same way, and the ellipse
    collapses onto a line -- measured at 7.9:1 median and 7663:1 worst case.
    A scalar "+/- 3 um" throws that away; the ellipse says *which direction*.
    """
    C = np.asarray(cov, dtype=np.float64)
    if C.shape != (2, 2):
        raise ValueError(f"cov must be (2, 2), got {C.shape}")
    w, V = np.linalg.eigh(0.5 * (C + C.T))
    w = np.maximum(w, 0.0)                      # guard round-off below zero
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    angle = float(np.arctan2(V[1, 0], V[0, 0]))
    return float(n_sigma * np.sqrt(w[0])), float(n_sigma * np.sqrt(w[1])), angle


def covariance_ellipse(
    cov: Array,
    center: Array | tuple[float, float] = (0.0, 0.0),
    n_sigma: float = 2.0,
    n_points: int = 64,
) -> Array:
    """Return points tracing a confidence ellipse, shape ``(n_points, 2)``.

    The path is closed: the last point repeats the first, so it can be drawn
    as a polyline without special-casing the seam.

    Parameters
    ----------
    cov:
        ``(2, 2)`` covariance, units of length squared.
    center:
        Ellipse centre, same units as ``sqrt(cov)``.
    n_sigma:
        Scale factor; see :func:`ellipse_axes`.
    n_points:
        Number of points on the path, including the repeated closing point.
    """
    if n_points < 3:
        raise ValueError(f"n_points must be >= 3, got {n_points}")
    major, minor, angle = ellipse_axes(cov, n_sigma)
    t = np.linspace(0.0, 2.0 * np.pi, n_points)
    unit = np.stack([major * np.cos(t), minor * np.sin(t)])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.asarray((rot @ unit).T + np.asarray(center, dtype=np.float64))
