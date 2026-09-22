"""Logging and dataset I/O.

Importing this package needs numpy only. :mod:`erp.io.config` is deliberately
**not** re-exported here, although it landed as part of the same package at
ADR-0002 P7: it imports :class:`~erp.sensors.IMUDecoder`, and re-exporting it
would mean ``import erp.io`` pulled ``erp.sensors`` in with it. :mod:`erp.fusion`
is allowed to depend on ``erp.io`` and is not allowed to reach ``erp.sensors``,
so that one line would put a hardware-facing package one hop from the filter --
the indirect reach-through the CI import grep cannot see. ``test_config.py``
checks it in a subprocess rather than trusting this paragraph.

Import the loader by name::

    from erp.io.config import load_config, build_decoder
"""

from erp.io.log import MeasurementLog

__all__ = ["MeasurementLog"]
