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
         rigido; ~0.4 mm en la simulacion con la clase de contacto
         "herramienta" (`mechanical/mujoco_assets/contacto.xml`).
- OFF -> DESIGUALDAD: d(x) >= 0. Solo se aplica si el estimado esta ADENTRO
         (d < -tol_off), como pseudo-medicion z = 0 que lo empuja hacia afuera.
         Si esta afuera no se toca nada -- ni x ni P --, porque OFF no dice
         donde esta la herramienta, solo que no esta apretando: un toque suave
         por debajo del umbral tambien lee OFF. Tratarlo como igualdad tiraria
         un estimado libre contra la superficie; `test_ekf_fsr.py` lo falsea.

Es esta mitad, OFF, la que ataca el sintoma medido: al levantar, el FSR pasa a
OFF y un estimado que se quedo adentro es empujado afuera en la siguiente
lectura.

COMO LLEGA LA LECTURA. El FSR es una fila mas de la `Measurement`: la del
`<touch name="fsr_force">` del modelo con adaptador, en NEWTONS, en la misma
linea serial que las IMUs y con el mismo sello. Con `fila_fsr` dado, `update`
la separa: las demas filas van a `EKF.update` y la del FSR a
`update_contacto(F > umbral_n)`. Asi el contacto pasa por `FilterRunner` como
cualquier medicion -- buffer, orden por sello, paso correcto -- y no hace
falta tocar el runner ni el contrato de `Measurement`.

Lo que NO puede pasar es que esa fila llegue a `EKF.update` como un canal
comun: el modelo ciego no tiene contacto, su `<touch>` lee 0 siempre, y la
fuerza entera se volveria innovacion de un `h` que no la puede explicar.
`test_ekf_fsr.py` lo falsea.

CIEGO, IGUAL QUE `EKF`: `update_contacto` toma un bool y nada mas. No hay `u`.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from erp.core.types import Array
from erp.estimators.ekf import EKF
from erp.models.base import ContactGeometry, DiscreteDynamics

__all__ = ["EKFFSR", "Contacto"]


class Contacto(NamedTuple):
    """Una lectura del FSR tal como la uso el filtro. Para graficar y medir.

    `paso` es el numero de predicts dados al aplicarla, asi que su instante es
    `t0 + paso * dt` del `FilterRunner`: el filtro no conoce el tiempo, pero el
    runner avanza hasta el paso de la muestra ANTES de corregir.
    """

    paso: int
    fuerza_n: float          # lo que leyo el FSR, N
    en_contacto: bool        # fuerza_n > umbral_n
    d_antes: float           # d(x) del estimado ANTES de corregir, m (negativa adentro)
    innovacion: float | None  # m; None si no hubo correccion (OFF con el estimado afuera)
    nis: float | None


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
    fila_fsr:
        Indice en `sensordata` de la fila del FSR (`rows_of(modelo, "fsr_force")`).
        Con `None`, `update` es exactamente el de `EKF`.
    umbral_n:
        Fuerza que cuenta como ON, N. 1 N por defecto: medido en la planta,
        0 N a 16 mm del piso y 24-32 N apoyado, asi que 1 N separa limpio
        (`scripts/medir_penetracion.py`). El FSR de la tesis de Nipote es
        ruidoso y con umbral por debajo de ~2 N; en el banco real hay que
        volver a medirlo.
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
        fila_fsr: int | None = None,
        umbral_n: float = 1.0,
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
        if fila_fsr is not None and fila_fsr < 0:
            raise ValueError(f"fila_fsr tiene que ser >= 0, es {fila_fsr}")
        self.fila_fsr = None if fila_fsr is None else int(fila_fsr)
        self.umbral_n = float(umbral_n)
        self.pasos = 0
        self.contactos: list[Contacto] = []
        """Cada lectura del FSR que paso por `update`, en orden. Crece sin
        limite: un consumidor vivo la vacia con `tomar_contactos`."""

    def predict(self) -> None:
        """`EKF.predict`, contando pasos para sellar los `Contacto`."""
        super().predict()
        self.pasos += 1

    def update(
        self,
        z: npt.ArrayLike,
        rows: npt.ArrayLike,
        R: npt.ArrayLike | None = None,
    ) -> tuple[Array, float]:
        """`EKF.update`, con la fila del FSR desviada a `update_contacto`.

        Primero las demas filas (IMUs), despues el contacto: las dos son la
        misma muestra, y corregir con las IMUs primero hace que el contacto se
        linealice en el estimado ya corregido. -> `(innovacion, NIS)` de las
        IMUs, para que la `History` del runner no mezcle un NIS en m^2/sigma
        de contacto con el de 12 canales. Si la medicion trae SOLO el FSR,
        devuelve el del contacto (o `(vacio, nan)` si no hubo correccion).
        """
        rows_i = np.asarray(rows, dtype=np.intp)
        if self.fila_fsr is None or self.fila_fsr not in rows_i:
            return super().update(z, rows_i, R)
        z_arr = np.asarray(z, dtype=np.float64)
        resto = rows_i != self.fila_fsr
        y: Array = np.zeros(0)
        nis = float("nan")
        if resto.any():
            R_k = None if R is None else np.asarray(R, dtype=np.float64)[np.ix_(resto, resto)]
            y, nis = super().update(z_arr, rows_i[resto], R_k)
        fuerza = float(z_arr[self.fila_fsr])
        on = fuerza > self.umbral_n
        d, res = self._contacto(on)
        self.contactos.append(Contacto(
            self.pasos, fuerza, on, d,
            None if res is None else res[0], None if res is None else res[1],
        ))
        if not resto.any() and res is not None:
            return np.array([res[0]]), res[1]
        return y, nis

    def tomar_contactos(self) -> list[Contacto]:
        """Devuelve los `Contacto` acumulados y vacia la lista."""
        out, self.contactos = self.contactos, []
        return out

    def update_contacto(self, en_contacto: bool) -> tuple[float, float] | None:
        """Corrige con una lectura binaria del FSR. -> `(innovacion m, NIS)` o None.

        None quiere decir que no hubo correccion (OFF con el estimado afuera), y
        en ese caso `x` y `P` quedan identicos bit a bit.
        """
        return self._contacto(en_contacto)[1]

    def _contacto(self, en_contacto: bool) -> tuple[float, tuple[float, float] | None]:
        """`update_contacto`, devolviendo tambien `d(x)` de ANTES de corregir, m."""
        d, D = self.geometria.distance(self.x)
        if en_contacto:
            z = -self.pen_on
        elif d < -self.tol_off:
            z = 0.0
        else:
            return float(d), None
        y = np.array([z - d])
        nis = self._corregir(y, np.asarray(D, dtype=np.float64)[None, :], self.R_contacto)
        return float(d), (float(y[0]), nis)
