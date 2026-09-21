"""Arm command sinks: the transport contract, the joint map, and two arms.

Importing this package needs numpy only. ``pymycobot`` is imported when a
:class:`~erp.robot.mypalletizer.MyPalletizerArm` is constructed without a
``transport``, and mujoco is never imported here -- :meth:`JointMap.validate`
takes an already-loaded model rather than loading one.

An arm is a command sink, not a measurement source. The estimation stack
(``core``, ``models``, ``sim``, ``estimators``, ``fusion``) must never import
this package; it sits beside :mod:`erp.sensors`, not under it.
"""

from erp.robot.base import ArmInterface, JointMap
from erp.robot.dry_run import DryRunArm
from erp.robot.mypalletizer import MyPalletizerArm

__all__ = ["ArmInterface", "DryRunArm", "JointMap", "MyPalletizerArm"]
