"""El veredicto de una corrida: NEES, NIS y cobertura.

Fase P8 de ADR-0002. La logica viene de ``report()`` en
``notebooks/finger_imu_toolkit.ipynb``, que es donde se decidieron las tres
cosas que hacen que el numero signifique algo. Las tres estan abajo con su
razon, porque cada una se puede "simplificar" y el resultado sigue pareciendo
un diagnostico.

Por que existe: la correccion de un filtro se juzga por NEES y NIS, nunca
mirando una trayectoria. Una estimacion que sigue bien la verdad y se cree diez
veces mas precisa de lo que es dibuja un grafico perfecto.

Sin verdad no hay NEES. Sobre el brazo real solo se puede calcular NIS, porque
el error de estado no es observable -- de ahi que este modulo tome la verdad
como opcional y devuelva ``None`` en vez de inventar un cero. Un ``0.0`` en un
campo de NEES se lee como "perfecto" y es exactamente al reves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from erp.core.linalg import nees_of
from erp.core.types import Array

__all__ = ["ConsistencyReport", "consistency_report"]

# Fraccion de la corrida que se descarta por delante. La primera mitad esta
# dominada por el transitorio de convergencia de P0: el filtro arranca con una
# covarianza inventada y tarda en olvidarla, asi que puntear ahi mide la
# condicion inicial y no el modelo.
DEFAULT_WARMUP_FRACTION = 0.5

# Cobertura nominal de una banda de 2 sigma para un gaussiano escalar. 0.95 es
# la cifra que se cita siempre; el valor exacto es erf(sqrt(2)) = 0.9545.
NOMINAL_2SIGMA_COVERAGE = 0.9545


@dataclass(frozen=True, slots=True, eq=False)
class ConsistencyReport:
    """Numeros de consistencia de una corrida, ya recortados a la ventana.

    ``eq=False`` porque :attr:`coverage_2sigma` es un array: el ``__eq__``
    generado lo compararia elemento a elemento y despues llamaria ``bool()``
    sobre el resultado, que levanta. Es la convencion de :mod:`erp.core.types`.

    Todo es mediana, no media, salvo :attr:`nis_mean`, que se conserva porque la
    corrida congelada lo guarda y porque la distancia entre media y mediana es
    en si misma un dato: si difieren por ordenes de magnitud, hay una cola.

    Attributes
    ----------
    nis_median, nis_mean:
        NIS sobre la ventana. Adimensional.
    nis_target:
        Grados de libertad de la medicion, ``len(rows)``. Un filtro consistente
        da NIS ~ ``nis_target``.
    n_nis:
        Muestras que entraron en el NIS.
    nees_median:
        NEES del estado, o ``None`` si no se paso verdad. Adimensional.
    nees_target:
        ``nx``, o ``None`` sin verdad.
    n_nees:
        Pasos que entraron en el NEES de estado.
    site_nees_median:
        NEES del efector (3 grados de libertad), o ``None``. Es el numero que
        importa para M1: es donde la estimacion se usa, y depende de las tres
        juntas a la vez.
    coverage_2sigma:
        (3,) fraccion de pasos en que ``|error| <= 2 sigma``, por eje del mundo,
        o ``None``. Nominal 0.9545. Por eje y no agregada porque la degradacion
        suele ser de un solo eje -- sobre la corrida congelada da
        1.00 / 1.00 / 0.93-0.96, y promediarlas esconderia el tercero.
    window_start:
        Indice donde arranco la ventana. Se guarda para que un reporte se pueda
        reproducir sin adivinar que fraccion se uso.
    """

    nis_median: float
    nis_mean: float
    nis_target: int
    n_nis: int
    nees_median: float | None
    nees_target: int | None
    n_nees: int
    site_nees_median: float | None
    coverage_2sigma: Array | None
    window_start: int

    def summary(self) -> str:
        """Una linea por metrica, con su objetivo al lado.

        El objetivo va pegado al numero a proposito: un NIS de 29 no dice nada
        sin el 12, y separar los dos es como se llega a citar una cifra sin su
        escala.
        """
        lines = [
            f"NIS      mediana {self.nis_median:8.2f}  media {self.nis_mean:8.2f}"
            f"   (objetivo {self.nis_target}, n = {self.n_nis})"
        ]
        if self.nees_median is None:
            lines.append("NEES     no calculable: la corrida no trae verdad de estado")
        else:
            lines.append(
                f"NEES     mediana {self.nees_median:8.2f}"
                f"                      (objetivo {self.nees_target}, n = {self.n_nees})"
            )
        if self.site_nees_median is None:
            lines.append("NEES ef. no calculable: la corrida no trae verdad del efector")
        else:
            lines.append(
                f"NEES ef. mediana {self.site_nees_median:8.2f}"
                "                      (objetivo 3)"
            )
        if self.coverage_2sigma is not None:
            cov = " / ".join(f"{c:.2f}" for c in self.coverage_2sigma)
            lines.append(
                f"cobertura 2 sigma por eje  {cov}"
                f"        (nominal {NOMINAL_2SIGMA_COVERAGE:.2f})"
            )
        return "\n".join(lines)


def _window(n: int, warmup_fraction: float) -> int:
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError(f"warmup_fraction debe estar en [0, 1), no {warmup_fraction}")
    return int(n * warmup_fraction)


def _nees_series(err: Array, P: Array) -> Array:
    """``e^T P^-1 e`` por muestra, reusando :func:`erp.core.linalg.nees_of`.

    Se llama al primitivo del core en un bucle en vez de vectorizar la misma
    formula aca: duplicarla seria una segunda definicion de la aritmetica que
    todo el resto del paquete evita, y con ~1800 pasos el bucle no se nota.
    """
    return np.asarray([nees_of(e, S) for e, S in zip(err, P, strict=True)], dtype=np.float64)


def consistency_report(
    nis: Array,
    *,
    nz: int,
    err: Array | None = None,
    P: Array | None = None,
    p_true: Array | None = None,
    p_est: Array | None = None,
    C_site: Array | None = None,
    warmup_fraction: float = DEFAULT_WARMUP_FRACTION,
) -> ConsistencyReport:
    """Arma el veredicto de una corrida.

    Parameters
    ----------
    nis:
        (m,) NIS por medicion. Adimensional.
    nz:
        Grados de libertad de la medicion, ``len(rows)``: el objetivo del NIS.
    err, P:
        (n, nx) error de estado ``x_verdad - x_estimado`` y (n, nx, nx)
        covarianza del filtro, para el NEES de estado. Los dos o ninguno.
    p_true, p_est, C_site:
        (n, 3) m posicion verdadera y estimada del sitio y (n, 3, 3) m^2 su
        covarianza entera, para el NEES del efector y la cobertura. Los tres o
        ninguno. ``C_site`` entero, no su diagonal: ver
        :func:`~erp.analysis.propagate_to_site`.
    warmup_fraction:
        Fraccion inicial descartada. Por defecto la mitad -- ver
        :data:`DEFAULT_WARMUP_FRACTION`.

    Notas sobre el metodo, que son la parte que importa:

    - **Mediana, no media.** La distribucion de NEES tiene cola pesada: el
      percentil 99 queda varios ordenes sobre la mediana por momentos breves en
      que P es chica y la linealizacion es mala. Una media sobre eso reporta la
      cola, no el filtro.
    - **Segunda mitad.** La primera esta dominada por el transitorio de P0.
    - **La cobertura se mide por eje.** Ver
      :attr:`ConsistencyReport.coverage_2sigma`.

    Raises
    ------
    ValueError
        Si un grupo opcional viene a medias, si las formas no coinciden, o si
        ``nis`` esta vacio. Un reporte armado sobre entradas inconsistentes
        devuelve numeros con la forma correcta y el significado equivocado, que
        es el unico modo de falla que este modulo existe para evitar.
    """
    nis = np.asarray(nis, dtype=np.float64)
    if nis.ndim != 1 or nis.size == 0:
        raise ValueError(f"nis debe ser (m,) y no vacio, es {nis.shape}")
    if nz <= 0:
        raise ValueError(f"nz debe ser > 0, es {nz}")

    k = _window(nis.size, warmup_fraction)
    nis_w = nis[k:]

    nees_median: float | None = None
    nees_target: int | None = None
    n_nees = 0
    if (err is None) != (P is None):
        raise ValueError("err y P van juntos o no van")
    if err is not None and P is not None:
        err = np.asarray(err, dtype=np.float64)
        P = np.asarray(P, dtype=np.float64)
        if err.ndim != 2 or P.ndim != 3 or P.shape[0] != err.shape[0]:
            raise ValueError(
                f"err (n, nx) y P (n, nx, nx) incompatibles: {err.shape}, {P.shape}"
            )
        nx = err.shape[1]
        if P.shape[1:] != (nx, nx):
            raise ValueError(f"P deberia ser (n, {nx}, {nx}), es {P.shape}")
        j = _window(err.shape[0], warmup_fraction)
        series = _nees_series(err[j:], P[j:])
        nees_median = float(np.median(series))
        nees_target = int(nx)
        n_nees = int(series.size)

    site_nees_median: float | None = None
    coverage: Array | None = None
    site_given = (p_true is not None, p_est is not None, C_site is not None)
    if any(site_given) and not all(site_given):
        raise ValueError("p_true, p_est y C_site van los tres o ninguno")
    if p_true is not None and p_est is not None and C_site is not None:
        p_true = np.asarray(p_true, dtype=np.float64)
        p_est = np.asarray(p_est, dtype=np.float64)
        C_site = np.asarray(C_site, dtype=np.float64)
        if p_true.ndim != 2 or p_true.shape[1] != 3 or p_true.shape != p_est.shape:
            raise ValueError(
                f"p_true y p_est deben ser (n, 3) iguales: {p_true.shape}, {p_est.shape}"
            )
        if C_site.shape != (p_true.shape[0], 3, 3):
            raise ValueError(f"C_site deberia ser ({p_true.shape[0]}, 3, 3), es {C_site.shape}")
        j = _window(p_true.shape[0], warmup_fraction)
        e_site = p_true[j:] - p_est[j:]
        site_nees_median = float(np.median(_nees_series(e_site, C_site[j:])))
        sig = np.sqrt(np.einsum("nii->ni", C_site[j:]))
        coverage = np.asarray((np.abs(e_site) <= 2.0 * sig).mean(axis=0), dtype=np.float64)

    return ConsistencyReport(
        nis_median=float(np.median(nis_w)),
        nis_mean=float(nis_w.mean()),
        nis_target=int(nz),
        n_nis=int(nis_w.size),
        nees_median=nees_median,
        nees_target=nees_target,
        n_nees=n_nees,
        site_nees_median=site_nees_median,
        coverage_2sigma=coverage,
        window_start=k,
    )
