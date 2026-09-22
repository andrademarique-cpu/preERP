"""Alineacion temporal entre una consigna y lo que se midio.

Levantado de `scripts/make_golden_run.py` en la fase P8 de ADR-0002, que a su
vez lo habia levantado del notebook. El cuerpo es el mismo: lo unico que se
agrego son las anotaciones de tipo que `mypy --strict` exige dentro del paquete
y que el script no necesitaba. `test_analysis.py` lleva una copia textual de la
version del script y exige que las dos den el MISMO float, bit a bit, sobre el
log grabado -- el mismo truco de oraculo que usan `test_fusion_runner.py` y
`test_calibration.py`, y la unica forma de que "se movio sin cambiarlo" sea una
afirmacion verificable y no una promesa.
"""

from __future__ import annotations

import numpy as np

from erp.core.types import Array

__all__ = ["estimate_lag"]


def estimate_lag(
    t_ref: Array,
    q_ref_deg: Array,
    t_meas: Array,
    q_meas_deg: Array,
    max_lag_s: float = 0.6,
    n: int = 241,
) -> float:
    """Retardo, en s, que mejor alinea la medida con la consigna (RMS minimo).

    Positivo = la medida va ATRASADA respecto de la consigna. Se barre el
    retardo y se interpola la consigna corrida sobre los sellos de tiempo
    REALES de la medida, no sobre los nominales. Solo se usan las muestras que
    caen dentro de la ventana valida, para que el arranque no invente
    correlacion donde no la hay.

    Unidades: `t_ref` y `t_meas` en s (base monotona del host, absoluta);
    `q_ref_deg` y `q_meas_deg` en grados, (N, k), una columna por junta.

    El barrido es una grilla de `n` puntos sobre [0, max_lag_s], o sea 2.5 ms de
    paso con los valores por defecto. Dos grabaciones que reporten 400 y 405 ms
    son puntos adyacentes de esa grilla, no una discrepancia: tres grabaciones
    del brazo dieron 400, 405 y 400 ms, asi que la dispersion es entre
    grabaciones y no del estimador.

    Devuelve NaN -- no levanta -- cuando no hay suficientes muestras dentro de
    la ventana, porque el llamador tipico es un plot y abortar la corrida por un
    log corto seria peor que dibujar un hueco.
    """
    if len(t_meas) < 4:
        return float("nan")
    inside = (t_meas >= t_ref[0] + max_lag_s) & (t_meas <= t_ref[-1])
    if inside.sum() < 4:
        return float("nan")
    tm, qm = t_meas[inside], q_meas_deg[inside]
    best, best_lag = np.inf, float("nan")
    for lag in np.linspace(0.0, max_lag_s, n):
        ref = np.column_stack([np.interp(tm - lag, t_ref, q_ref_deg[:, k])
                               for k in range(q_ref_deg.shape[1])])
        rms = float(np.sqrt(np.mean((qm - ref) ** 2)))
        if rms < best:
            best, best_lag = rms, lag
    return best_lag
