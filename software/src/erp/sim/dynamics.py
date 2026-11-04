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
from erp.sim.plant import distancia_con_jacobiana, geoms_por_clase

MjModel = Any
MjData = Any

__all__ = ["MujocoContactGeometry", "MujocoDynamics"]


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


class MujocoContactGeometry:
    """`ContactGeometry` sobre el MAPA de la escena: herramienta vs solidos conocidos.

    El modelo que se le pasa es el de la PLANTA cargado con `load_model` (por
    ejemplo `MyPalletizer260_fsr.xml` o `MyPalletizer260_fsr_escaleras.xml`),
    NO el modelo ciego de la dinamica. Son dos cosas a proposito: el mapa
    necesita los solidos (piso y props con conaffinity = 2) y la dinamica del
    filtro no tiene que tenerlos -- con contacto en la dinamica el estimado se
    pega al piso al levantar (`scripts/medir_penetracion.py`). Las dos comparten
    el brazo: mismo nq = nv y mismo orden, asi que `x[:nv]` vale para las dos.

    `herramienta` es la geometria SENSIBLE, por nombre: `fsr_tip`, la cupula que
    cubre el `fsr_pad` del `<touch>`. ON quiere decir que toco la cupula, no el
    cilindro del adaptador ni el `ee_collider`.

    `nx` es el del FILTRO, no el del mapa: el mapa es la planta (na = 0, nx = 8)
    y el filtro corre sobre el modelo ciego (na = 3, nx = 11). Se exige que el
    mapa tenga el mismo nv y que entre en `nx`; las activaciones no mueven la
    distancia y su bloque de `D` es cero.

    `data` es un `MjData` PROPIO del mapa (el de `load_model`); se escribe su
    qpos en cada llamada.
    """

    def __init__(
        self,
        model: MjModel,
        data: MjData,
        nx: int,
        *,
        herramienta: str = "fsr_tip",
        dist_max: float = 0.5,
    ) -> None:
        nombres = [str(model.geom(g).name) for g in range(int(model.ngeom))]
        if herramienta not in nombres:
            con_nombre = sorted(n for n in nombres if n)
            raise KeyError(f"el mapa no tiene el geom {herramienta!r}; tiene {con_nombre}")
        _, obstaculos = geoms_por_clase(model)
        if obstaculos.size == 0:
            raise ValueError("el mapa no tiene solidos (ningun geom con conaffinity = 2)")
        if int(model.nq) != int(model.nv) or nx < 2 * int(model.nv):
            raise ValueError(
                f"el mapa (nq={model.nq}, nv={model.nv}) no encaja en un estado de nx={nx}"
            )
        self.model = model
        self.data = data
        self.herramienta = np.array([nombres.index(herramienta)], dtype=np.intp)
        self.obstaculos = obstaculos
        self.dist_max = float(dist_max)
        self.nx: int = int(nx)

    def distance(self, x: npt.ArrayLike) -> tuple[float, Array]:
        xa = np.asarray(x, dtype=np.float64)
        if xa.size != self.nx:
            raise ValueError(f"x tiene {xa.size} elementos, el mapa pide {self.nx}")
        return distancia_con_jacobiana(
            self.model, self.data, xa, self.herramienta, self.obstaculos, self.dist_max
        )

    def __repr__(self) -> str:
        nombre = self.model.geom(int(self.herramienta[0])).name
        return (
            f"MujocoContactGeometry(herramienta={nombre!r}, "
            f"obstaculos={len(self.obstaculos)}, nx={self.nx})"
        )
