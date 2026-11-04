"""Cuanto se mete el estimado del filtro adentro de un solido.

El sintoma a medir: cuando el brazo toca algo que el modelo del filtro no tiene,
el estimado deriva y la pose estimada atraviesa el solido. El NIS no lo mide --
mide la innovacion, no la geometria -- y el fantasma de `demo_teleop.py` lo
muestra pero no lo cuenta. Esto lo cuenta.

Entrada: dos series de distancia CON SIGNO herramienta-obstaculo (negativa =
adentro), una con la pose verdadera y otra con la estimada, ambas medidas en la
geometria de la PLANTA (`erp.sim.plant.distancias`). Solo numpy.

Por que contra la verdad y no contra cero: el contacto de MuJoCo es blando, y la
planta misma se hunde unos milimetros en lo que toca. Esa penetracion no es del
filtro y no se le puede cobrar; lo que se le cobra es el EXCESO sobre ella.
`test_penetracion.py` empareja esto con la variante rota que mide contra cero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from erp.core.types import Array

__all__ = ["ResumenPenetracion", "penetracion", "resumen_penetracion"]

UMBRAL_M = 1e-3


def penetracion(dist: npt.ArrayLike) -> Array:
    """Profundidad adentro del solido, m: `max(0, -dist)`. Afuera es 0, nunca negativa.

    No es `abs(dist)`: eso contaria la holgura de una herramienta que esta
    AFUERA como si estuviera adentro.
    """
    d = np.asarray(dist, dtype=np.float64)
    return np.asarray(np.maximum(0.0, -d), dtype=np.float64)


@dataclass(frozen=True)
class ResumenPenetracion:
    """Resumen de una corrida. Longitudes en mm, tiempos en s.

    max_pen_est_mm, max_pen_true_mm:
        La penetracion mas honda de la pose estimada y de la verdadera. La
        segunda es el piso alcanzable (contacto blando), no un error.
    max_exceso_mm:
        max(p_est - p_true): lo que el FILTRO agrega. Es la cifra principal.
    integral_mm_s:
        Integral (trapecios) de max(0, exceso): profundidad por duracion. Un
        recorte corto y hondo y uno largo y somero pueden dar el mismo maximo;
        esto los separa.
    frac_recorte:
        Fraccion de las muestras con exceso > umbral. Por muestra, no pesada por
        dt: asume la grilla uniforme del tick de la demo.
    recuperacion_s:
        Por cada contacto verdadero (d_true <= 0), el tiempo desde que TERMINA
        hasta que el exceso vuelve a <= umbral; el peor de todos. `inf` si no
        vuelve antes del fin de la corrida -- la deriva que persiste --, `0.0`
        si no hubo contacto.
    max_exceso_sigma:
        max(exceso / sigma_n) sobre las muestras con exceso > umbral, donde
        sigma_n es el desvio del efector a lo largo de la normal del obstaculo.
        Hasta ~2 el filtro admite que no sabe; mucho mas es un recorte
        SOBRECONFIADO, que es el error de verdad. `nan` si no se paso sigma o
        no hubo exceso.
    n_contactos:
        Episodios de contacto verdadero.
    """

    max_pen_est_mm: float
    max_pen_true_mm: float
    max_exceso_mm: float
    integral_mm_s: float
    frac_recorte: float
    recuperacion_s: float
    max_exceso_sigma: float
    n_contactos: int


def _episodios(contacto: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """[(inicio, fin)) de cada tramo contiguo en True."""
    c = np.r_[False, contacto, False].astype(np.int8)
    bordes = np.flatnonzero(np.diff(c))
    return [(int(a), int(b)) for a, b in zip(bordes[::2], bordes[1::2], strict=True)]


def resumen_penetracion(
    t: npt.ArrayLike,
    d_est: npt.ArrayLike,
    d_true: npt.ArrayLike,
    sigma_n: npt.ArrayLike | None = None,
    umbral_m: float = UMBRAL_M,
) -> ResumenPenetracion:
    """Resume dos series 1-D de distancia con signo (m) sobre la grilla `t` (s).

    `sigma_n` (m), si se da, es el desvio del efector a lo largo de la normal del
    obstaculo, `sqrt(n^T C n)` con C de `erp.analysis.propagate_to_site`.
    """
    tt = np.asarray(t, dtype=np.float64)
    de = np.asarray(d_est, dtype=np.float64)
    dv = np.asarray(d_true, dtype=np.float64)
    if not (tt.shape == de.shape == dv.shape) or tt.ndim != 1:
        raise ValueError(f"t, d_est y d_true tienen que ser 1-D y iguales: {tt.shape}, "
                         f"{de.shape}, {dv.shape}")
    # Antes de saber si hay exceso: si no, una sigma mal formada pasa callada
    # justo en las corridas limpias, y revienta recien en la primera que recorta.
    s = None if sigma_n is None else np.asarray(sigma_n, dtype=np.float64)
    if s is not None and s.shape != tt.shape:
        raise ValueError(f"sigma_n tiene forma {s.shape}, se esperaba {tt.shape}")
    p_est, p_true = penetracion(de), penetracion(dv)
    exceso = p_est - p_true
    pos = np.maximum(0.0, exceso)
    integral = float(np.sum(0.5 * (pos[1:] + pos[:-1]) * np.diff(tt))) if len(tt) > 1 else 0.0
    recorta = exceso > umbral_m

    episodios = _episodios(dv <= 0.0)
    recuperacion = 0.0
    for _, fin in episodios:
        if fin >= len(tt):
            continue            # el contacto dura hasta el final: no hay "despues"
        ok = np.flatnonzero(exceso[fin:] <= umbral_m)
        # `fin` es la primera muestra sin contacto: 0 s si ya ahi el exceso bajo.
        r = float(tt[fin + ok[0]] - tt[fin]) if len(ok) else np.inf
        recuperacion = max(recuperacion, r)

    exceso_sigma = np.nan
    if s is not None and recorta.any():
        exceso_sigma = float(np.max(exceso[recorta] / s[recorta]))

    return ResumenPenetracion(
        max_pen_est_mm=float(p_est.max(initial=0.0)) * 1e3,
        max_pen_true_mm=float(p_true.max(initial=0.0)) * 1e3,
        max_exceso_mm=float(exceso.max(initial=0.0)) * 1e3,
        integral_mm_s=integral * 1e3,
        frac_recorte=float(recorta.mean()) if len(tt) else 0.0,
        recuperacion_s=recuperacion,
        max_exceso_sigma=exceso_sigma,
        n_contactos=len(episodios),
    )
