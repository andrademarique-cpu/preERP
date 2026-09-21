"""`MujocoDynamics`: MuJoCo detras del contrato `DiscreteDynamics`.

Es la pieza que saca el `MjModel` de adentro del EKF. El filtro pasa a hablar
con un protocolo de numpy, y este modulo es el unico lugar donde ese protocolo
se cumple con fisica de verdad.

No toca la API de mujoco directamente: llama a `erp.sim.mujoco`, que sigue
siendo el unico modulo que lo hace, y por eso no se escapa ningun `Any` hacia
el estimador.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from erp.core.types import Array
from erp.sim.mujoco import observe_with_jacobian, state_dim, step_with_jacobian

MjModel = Any
MjData = Any

__all__ = ["MujocoDynamics"]


class MujocoDynamics:
    """`DiscreteDynamics` sobre un `MjModel` + `MjData`.

    El modelo que se le pasa decide si el filtro que lo use es sano o no, y esa
    eleccion no es de este modulo: `blind_variant` deja las activaciones como
    random walk (que es lo que el EKF ciego estima en lugar del comando que
    nunca recibe), mientras que el modelo de planta las decae a cero. La planta
    es la que GENERA la verdad; nunca es sobre la que corre el filtro.

    Guarda su propio `MjData`. No es reentrante ni thread-safe: dos hilos
    llamando `step` sobre la misma instancia se pisan el `MjData`. En este
    proyecto el filtro corre en el hilo principal y los sensores en los suyos,
    asi que alcanza; si alguna vez hay dos filtros, van dos `MjData`.

    Parameters
    ----------
    model, data:
        Lo que devuelve `erp.sim.plant.blind_variant`.
    eps:
        Paso de la diferencia finita de `mjd_transitionFD`.
    """

    def __init__(self, model: MjModel, data: MjData, *, eps: float = 1e-6) -> None:
        if eps <= 0.0:
            raise ValueError(f"eps tiene que ser > 0, es {eps}")
        self.model = model
        self.data = data
        self.eps = float(eps)
        self.nx: int = state_dim(model)
        self.nz: int = int(model.nsensordata)
        self.nu: int = int(model.nu)
        self.dt: float = float(model.opt.timestep)

    def step(self, x: npt.ArrayLike, u: npt.ArrayLike) -> tuple[Array, Array]:
        """`(x_next, F)`, las dos desde una sola carga del `MjData`."""
        x_next, F = step_with_jacobian(
            np.asarray(x, dtype=np.float64),
            np.asarray(u, dtype=np.float64),
            self.model,
            self.data,
            self.eps,
        )
        return np.asarray(x_next, dtype=np.float64), np.asarray(F, dtype=np.float64)

    def observe(self, x: npt.ArrayLike, u: npt.ArrayLike) -> tuple[Array, Array]:
        """`(z, H)` completas, las dos desde una sola carga del `MjData`."""
        z, H = observe_with_jacobian(
            np.asarray(x, dtype=np.float64),
            np.asarray(u, dtype=np.float64),
            self.model,
            self.data,
            self.eps,
        )
        return np.asarray(z, dtype=np.float64), np.asarray(H, dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"MujocoDynamics(nx={self.nx}, nz={self.nz}, nu={self.nu}, "
            f"dt={self.dt * 1e3:.0f} ms)"
        )
