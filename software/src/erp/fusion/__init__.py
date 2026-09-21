"""Online and offline fusion: the layer that turns timestamps into filter steps.

Empty until P5. The package the old ``FusionEngine`` occupied was deleted in
``1987f00`` along with the finger stack; nothing of it survives here, and the
``InputHistory`` / event-timeline design it came with is **not** the current
plan -- see ADR-0002 4.7 before assuming otherwise.

One rule this package exists to hold: it does not import :mod:`erp.sensors` or
:mod:`erp.robot`. :meth:`~erp.fusion.runner.FilterRunner.ingest` takes a
:class:`~erp.core.types.Measurement`, which is a ``core`` type, so the filter
never learns which class produced the sample. That is the whole reason the same
runner works against a replayed CSV, a simulated IMU and a Teensy.
"""

from erp.fusion.runner import Estimator, FilterRunner, History

__all__ = ["Estimator", "FilterRunner", "History"]
