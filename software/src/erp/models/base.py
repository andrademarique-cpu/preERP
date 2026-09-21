"""Contrato de dinamica discreta: un paso de transicion y una observacion.

El desacople que mas importa de ADR-0002: **el EKF no puede tener un MjModel
adentro**. Mientras lo tenga, no hay forma de correrlo contra un modelo lineal
con solucion cerrada, y sin eso la unica evidencia de que el filtro esta bien
es que sus graficos parecen razonables.

El protocolo no sabe de sensores, ni de reloj de pared, ni de donde salio `z`.
Solo numpy: este modulo tiene que poder importarse sin mujoco. La
implementacion sobre MuJoCo vive en `erp.sim.dynamics`, del lado que si puede
importarlo.

POR QUE DEVUELVEN PARES y no cuatro metodos f / F / h / H sueltos:

1. Garantiza que el valor y su jacobiana salen del MISMO punto de
   linealizacion. Con metodos separados nada impide evaluarlos en x distintos,
   y ese bug no se ve: el filtro sigue corriendo y diverge de a poco.
2. Sobre MuJoCo ademas comparte la carga del MjData -- cuatro cargas por ciclo
   pasan a dos. El ahorro medido es ~8%, no el 50% que sugiere el conteo: lo
   caro es `mjd_transitionFD`, no la carga.

Es tambien la razon por la que NO se revive la division de ADR-0001 entre
`ProcessModel` y `MeasurementModel`: MuJoCo calcula las dos cosas desde una
sola carga, asi que separarlas duplica el costo e invita a dos puntos de
linealizacion distintos.
"""

from __future__ import annotations

from typing import Protocol

from erp.core.types import Array

__all__ = ["DiscreteDynamics"]


class DiscreteDynamics(Protocol):
    """Una transicion de paso fijo y una observacion, ambas con su jacobiana.

    `nx` es la dimension del estado en el espacio TANGENTE (`2*nv + na` sobre
    MuJoCo: `nv`, no `nq`), que es lo que hace valida una jacobiana por
    diferencias finitas.
    """

    nx: int
    """Dimension del estado."""

    nz: int
    """Dimension de la observacion COMPLETA, antes de elegir filas."""

    nu: int
    """Dimension del control. El EKF de este proyecto le pasa siempre ceros."""

    dt: float
    """Paso nativo, en segundos. No es invariante de escala: ver `make_Q`."""

    def step(self, x: Array, u: Array) -> tuple[Array, Array]:
        """`(x_next, F)` -- x_{k+1} y df/dx, las dos evaluadas en `(x, u)`.

        `F` es la jacobiana en el x de ENTRADA, no en el de salida.
        """
        ...

    def observe(self, x: Array, u: Array) -> tuple[Array, Array]:
        """`(z, H)` -- observacion completa `(nz,)` y dh/dx `(nz, nx)` en `(x, u)`.

        Devuelve TODOS los canales. Elegir cuales se usan es trabajo de quien
        llama, con las `rows` de la `Measurement`.
        """
        ...
