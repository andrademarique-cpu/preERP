"""Turning still data into a bias and a noise covariance.

Landed at ADR-0002 P6, which moved the rest-window bias out of a notebook cell
and out of ``scripts/make_golden_run.py`` and gave it the ``valid`` / ``note``
shape of :class:`~erp.core.types.CalibrationResult`, so an unusable
calibration is *refused* instead of silently applied as a zero.

**This package imports `erp.core` and numpy, and nothing else in `erp`.**
``sensors/`` imports this; this never imports ``sensors/``. That direction is
the point: a ``calibration`` module reaching back into ``sensors`` is one of
the indirect violations the CI import grep cannot see.
"""

from erp.calibration.rest import calibration_from_samples, rest_bias

__all__ = ["calibration_from_samples", "rest_bias"]
