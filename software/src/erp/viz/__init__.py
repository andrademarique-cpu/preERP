"""Plotting and visualisation helpers.

Landed at ADR-0002 P8. Until then this package held one docstring and nothing
else, and it said "geometry only, no backend imports" -- a rule § 4.2 explicitly
overturns, because it is what produced an empty package while the real plotting
accumulated in notebook cells. matplotlib is allowed here, gated behind the
``[viz]`` extra.

**Only the numpy half is re-exported.** :mod:`erp.viz.geometry` is pure numpy
and comes through; :mod:`erp.viz.theme` and :mod:`erp.viz.figures` import
matplotlib at module scope and are imported by name::

    from erp.viz.figures import three_way, sensor_compare, effector_band

The reason is CI: it installs ``.[dev]`` without ``[viz]``, so an ``__init__``
that reached matplotlib would make ``import erp.viz`` fail there. Same shape as
:mod:`erp.io`, which withholds ``erp.io.config`` for the analogous reason, and
the split also keeps the part that can be wrong *numerically* testable in the
fast suite with no display attached.

English, matching this file's original docstring. :mod:`erp.analysis`, which
landed in the same phase, is Spanish -- see its ``__init__`` for why the two
differ.
"""

from erp.viz.geometry import ellipsoid_axes, principal_tilt_deg, sigma_per_axis

__all__ = ["ellipsoid_axes", "principal_tilt_deg", "sigma_per_axis"]
