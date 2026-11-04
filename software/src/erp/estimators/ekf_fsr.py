# software/src/erp/estimators/ekf_fsr.py
"""EKF ciego + el FSR de la punta como contacto ON/OFF. Joseph, como `EKF`.

DE DONDE SALE. La tesis de Nipote (PoliMi 2024-25) mete la fuerza del FSR en el
ESTADO, x = [q, qdot, f], y la recorta al rango del sensor en un UKF restringido
(CCUKF): f = min(max(f_pred, 0), f_max). Eso no se traslada aca, por tres
razones:

1. Este EKF es CIEGO por decision del proyecto y no tiene estado de fuerza.
2. El FSR de este banco se usa BINARIO: toca / no toca. Por debajo de ~2 N el
   FSR de la tesis es ruidoso y con umbral, asi que su valor continuo no se le
   cree; el cruce de un umbral si.
3. Lo que importa medir no es la fuerza sino la GEOMETRIA: el sintoma a
   corregir es el estimado atravesando solidos (`scripts/medir_penetracion.py`).

QUE HACE. El contacto entra como MEDICION de la distancia con signo de la cupula
al solido conocido mas cercano, d(x) (negativa adentro), con su jacobiana D que
da un `ContactGeometry`. La dinamica sigue SIN contacto, y eso no es descuido:
con el piso solido adentro del modelo ciego el estimado se queda pegado al piso
al levantar -- 5.97 mm adentro a 69.5 sigma, NIS 103 027 y J1 corrida 6.6 grados
para siempre -- mientras que sin contacto en la dinamica se porta bien.

- ON  -> igualdad: "la cupula esta en la superficie", z = -pen_on, h(x) = d(x).
         `pen_on` es lo que el contacto se hunde de verdad: 0 en un banco
         rigido; ~0.8 mm en la simulacion con el contacto rigido del adaptador
         (`MyPalletizer260_fsr.xml`, solref -1e5 -1e3).
- OFF -> DESIGUALDAD: d(x) >= 0. Solo se aplica si el estimado esta ADENTRO
         (d < -tol_off), como pseudo-medicion z = 0 que lo empuja hacia afuera.
         Si esta afuera no se toca nada -- ni x ni P --, porque OFF no dice
         donde esta la herramienta, solo que no esta apretando: un toque suave
         por debajo del umbral tambien lee OFF. Tratarlo como igualdad tiraria
         un estimado libre contra la superficie; `test_ekf_fsr.py` lo falsea.

Es esta mitad, OFF, la que ataca el sintoma medido: al levantar, el FSR pasa a
OFF y un estimado que se quedo adentro es empujado afuera en la siguiente
lectura.

CIEGO, IGUAL QUE `EKF`: `update_contacto` toma un bool y nada mas. No hay `u`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from erp.core.types import Array
from erp.estimators.ekf import EKF
from erp.models.base import ContactGeometry, DiscreteDynamics

__all__ = ["EKFFSR"]


class EKFFSR(EKF):
    """`EKF` + `update_contacto(en_contacto)`.

    Parameters
    ----------
    x0, P0, Q, R, dyn:
        Como en `EKF`. `dyn` tiene que ser una dinamica SIN contacto (ver el
        docstring del modulo).
    geometria:
        `ContactGeometry` del mapa de la escena, con el mismo `nx` que `dyn`.
    sig_contacto:
        Desvio de la pseudo-medicion de contacto, m. Chico (1 mm por defecto):
        cuando el FSR dice que toca, la cupula esta en la superficie con la
        precision de la geometria del mapa, no con la de una IMU.
    pen_on:
        Penetracion esperada con el FSR en ON, m (>= 0).
    tol_off:
        Cuanto puede estar adentro el estimado con el FSR en OFF antes de
        corregirlo, m (>= 0). 0 por defecto.
    """

    def __init__(
        self,
        x0: npt.ArrayLike,
        P0: npt.ArrayLike,
        Q: npt.ArrayLike,
        R: npt.ArrayLike,
        dyn: DiscreteDynamics,
        geometria: ContactGeometry,
        *,
        sig_contacto: float = 1e-3,
        pen_on: float = 0.0,
        tol_off: float = 0.0,
    ) -> None:
        super().__init__(x0, P0, Q, R, dyn)
        if geometria.nx != dyn.nx:
            raise ValueError(f"la geometria tiene nx={geometria.nx}, la dinamica {dyn.nx}")
        if sig_contacto <= 0.0:
            raise ValueError(f"sig_contacto tiene que ser > 0, es {sig_contacto}")
        if pen_on < 0.0 or tol_off < 0.0:
            raise ValueError(f"pen_on y tol_off son >= 0: {pen_on}, {tol_off}")
        self.geometria = geometria
        self.R_contacto: Array = np.array([[float(sig_contacto) ** 2]])
        self.pen_on = float(pen_on)
        self.tol_off = float(tol_off)

    def update_contacto(self, en_contacto: bool) -> tuple[float, float] | None:
        """Corrige con una lectura binaria del FSR. -> `(innovacion m, NIS)` o None.

        None quiere decir que no hubo correccion (OFF con el estimado afuera), y
        en ese caso `x` y `P` quedan identicos bit a bit.
        """
        d, D = self.geometria.distance(self.x)
        if en_contacto:
            z = -self.pen_on
        elif d < -self.tol_off:
            z = 0.0
        else:
            return None
        y = np.array([z - d])
        nis = self._corregir(y, np.asarray(D, dtype=np.float64)[None, :], self.R_contacto)
        return float(y[0]), nis
