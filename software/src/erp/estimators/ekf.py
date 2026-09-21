# software/src/erp/estimators/ekf.py
"""EKF sobre `DiscreteDynamics`. Update en forma de Joseph. CIEGO al control."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from erp.core.linalg import make_spd
from erp.core.types import Array
from erp.models.base import DiscreteDynamics

__all__ = ["EKF"]


class EKF:
    """EKF con f/F/h/H de una `DiscreteDynamics`. Joseph. CIEGO al control.

    YA NO TIENE UN MjModel ADENTRO. Habla con el protocolo de `erp.models.base`,
    asi que el mismo filtro corre sobre MuJoCo (`erp.sim.dynamics`) o sobre una
    dinamica lineal (`erp.models.linear`). Eso ultimo no es un adorno: es lo que
    permite compararlo contra un KF con solucion cerrada, que es la unica
    evidencia real de que la aritmetica esta bien.

    CIEGO, Y ASI SE QUEDA -- decision del proyecto, no un estado transitorio.
    `self.u_blind` es el UNICO ctrl que escribe: se aloca una vez, en cero, y va
    a `step` y a `observe` por igual, asi que las dos dependen solo del estado.
    Ningun metodo toma un parametro `u`, y eso es la garantia: no es que no se
    le pase control, es que no hay por donde. Un test lo verifica por firma.

    Que eso sea sano depende enteramente del modelo que se le da: el modelo
    ciego deja la activacion como random walk -- que es lo que el filtro estima
    en lugar del comando que nunca recibe -- mientras que el modelo de planta la
    decae a cero. La planta genera la verdad; nunca es sobre la que corre el
    filtro.

    Parameters
    ----------
    x0: (nx,)        estado inicial
    P0: (nx, nx)     covarianza inicial
    Q:  (nx, nx)     ruido de proceso, valido al `dyn.dt` con que se armo
    R:  (nz, nz)     ruido de medicion sobre la observacion COMPLETA. Es el
                     valor por defecto: `update` acepta una `R` por medicion y
                     la prefiere si se la dan.
    dyn:             la dinamica. `MujocoDynamics` o `LinearDynamics`.
    """

    def __init__(
        self,
        x0: npt.ArrayLike,
        P0: npt.ArrayLike,
        Q: npt.ArrayLike,
        R: npt.ArrayLike,
        dyn: DiscreteDynamics,
    ) -> None:
        self.x: Array = np.asarray(x0, dtype=np.float64).copy()
        self.P: Array = make_spd(np.asarray(P0, dtype=np.float64))
        self.Q: Array = np.asarray(Q, dtype=np.float64)
        self.R: Array = np.asarray(R, dtype=np.float64)
        self.dyn = dyn
        if self.x.size != dyn.nx:
            raise ValueError(f"x0 tiene {self.x.size} elementos, la dinamica pide {dyn.nx}")
        # El unico ctrl que ve el filtro. Alocado una vez, en cero, para siempre.
        self.u_blind: Array = np.zeros(dyn.nu)

    def predict(self) -> None:
        """t_k -> t_{k+1}, un paso de `dyn.dt`.

        `F` sale del MISMO punto que `x_next`, que es el x de ANTES del paso.
        Por eso vienen juntas del `step`: pedirlas por separado deja abierta la
        posibilidad de evaluarlas en x distintos, y ese bug no se ve.
        """
        x_next, F = self.dyn.step(self.x, self.u_blind)
        self.x = x_next
        self.P = make_spd(F @ self.P @ F.T + self.Q)

    def update(
        self,
        z: npt.ArrayLike,
        rows: npt.ArrayLike,
        R: npt.ArrayLike | None = None,
    ) -> tuple[Array, float]:
        """Corrige con los canales `rows` de `z` (observacion COMPLETA).

        -> `(innovacion, NIS)`.

        `solve` y no `inv`: S se pone mal condicionada cuando el brazo se
        estira. Joseph y no `(I-KH)P`: sobrevive el redondeo que la forma corta
        no.

        `R` cierra el callejon sin salida de ADR-0002 3.2. Hasta P4 la
        `Measurement` cargaba una `R` que el filtro nunca miraba, asi que
        `SerialIMUSensor.calibrate()` no cambiaba absolutamente nada del
        resultado. Ahora quien llama puede pasar la `R` (k, k) de la medicion;
        sin ella se usa el bloque `self.R[rows, rows]`, que es lo que se hacia
        siempre. Hoy los dos caminos dan el mismo numero -- el `R` del sensor se
        construye como ese mismo bloque -- y eso es a proposito: la fase mueve
        el cableado sin mover la corrida congelada. Empieza a importar en P6,
        cuando `calibrate()` corra sobre el brazo de verdad.
        """
        rows_i = np.asarray(rows, dtype=np.intp)
        z_arr = np.asarray(z, dtype=np.float64)
        z_pred, H_full = self.dyn.observe(self.x, self.u_blind)
        H = H_full[rows_i]
        y = z_arr[rows_i] - z_pred[rows_i]
        R_k = self.R[np.ix_(rows_i, rows_i)] if R is None else np.asarray(R, dtype=np.float64)
        if R_k.shape != (rows_i.size, rows_i.size):
            raise ValueError(
                f"R tiene que ser ({rows_i.size}, {rows_i.size}) para estas rows, "
                f"es {R_k.shape}"
            )
        S = make_spd(H @ self.P @ H.T + R_k)
        K = np.linalg.solve(S, H @ self.P).T  # = P H^T S^-1
        self.x = self.x + K @ y
        I_KH = np.eye(self.x.size) - K @ H
        self.P = make_spd(I_KH @ self.P @ I_KH.T + K @ R_k @ K.T)
        return y, float(y @ np.linalg.solve(S, y))
