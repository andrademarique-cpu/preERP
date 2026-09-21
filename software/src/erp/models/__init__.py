# software/src/erp/models/__init__.py
"""Contratos de modelo y la dinamica lineal. Sin hardware, y sin mujoco.

Este paquete se importa con numpy y nada mas. La implementacion de
`DiscreteDynamics` sobre MuJoCo vive en `erp.sim.dynamics`, del lado que si
puede importarlo: es lo que permite testear el EKF sin mujoco cargado.
"""

from erp.models.base import DiscreteDynamics
from erp.models.linear import LinearDynamics, affine_predict

__all__ = ["DiscreteDynamics", "LinearDynamics", "affine_predict"]
