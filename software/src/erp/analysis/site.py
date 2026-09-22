"""Del estado del filtro a la posicion de un sitio, con su covarianza entera.

Levantado de `site_position_cov` en `scripts/make_golden_run.py` en la fase P8
de ADR-0002, con el cuerpo intacto y anotaciones de tipo agregadas. El nombre
cambio a `propagate_to_site` porque es el que ADR-0002 4.1 le reserva; la
funcion es la misma y `test_analysis.py` lo comprueba contra una copia textual.

Este modulo toca mujoco a traves de `erp.sim.mujoco`, nunca la API directa: la
regla de que `erp.sim.mujoco` es el unico que llama `mj.*` para la dinamica vale
igual para un consumidor de solo lectura.

Una sola desviacion respecto del original, y es forzada: ruff B905 exige
`strict=` explicito en `zip`, regla que el script nunca vio porque CI no lintea
`scripts/`. Se puso `strict=True`, que es la semantica correcta -- `XH` y `PP`
salen de la misma corrida y una diferencia de largo es un bug, no algo que haya
que truncar en silencio. Sobre entradas del mismo largo, que es el caso de la
corrida congelada, no cambia ningun numero.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from erp.core.types import Array, IntArray
from erp.sim.mujoco import H_dyn, h_dyn

__all__ = ["propagate_to_site"]


def propagate_to_site(
    XH: Array,
    PP: Array,
    model: Any,
    data: Any,
    rows: IntArray,
) -> tuple[Array, Array]:
    """Posicion de un sensor `framepos` y su covarianza, propagada desde el estado.

    p = h(x)[rows],   Sigma_p = H P H^T   con   H = dh/dx [rows]   (3 x nx)

    Covarianza COMPLETA, no sqrt(diag(P)) junta por junta: el efector depende de
    las tres juntas a la vez y las correlaciones que el filtro tiene entre ellas
    cambian la banda. H es la misma diferencia finita que usa el filtro; su
    bloque de q coincide con mj_jacSite a 3e-8.

    Que la covarianza salga entera y no achatada a su diagonal no es prolijidad.
    Medido sobre la corrida congelada: el eje mayor del elipsoide de error del
    efector esta a una mediana de 24.9 grados (maximo 45.0) del eje del mundo mas
    cercano, asi que los sigmas por eje subestiman la peor direccion una mediana
    de 9.6% y hasta 39.7%. Quien solo necesite la banda por eje que haga
    `np.sqrt(np.einsum("nii->ni", C))`; el que dibuje una elipse necesita C.

    Unidades: `p` en m, marco mundo; `C` en m^2.

    -> p (N, 3) m, marco mundo;  C (N, 3, 3) m^2
    """
    u = np.zeros(model.nu)
    p = np.empty((len(XH), 3))
    C = np.empty((len(XH), 3, 3))
    for i, (x, P) in enumerate(zip(XH, PP, strict=True)):
        p[i] = h_dyn(x, u, model, data)[rows]
        H = H_dyn(x, u, model, data)[rows]
        C[i] = H @ P @ H.T
    return p, C
