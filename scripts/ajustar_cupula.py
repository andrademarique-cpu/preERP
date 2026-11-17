"""Blandura de la cupula del FSR a partir del ensayo de indentacion.

    python scripts/ajustar_cupula.py --autoprueba                    # datos sinteticos + falsaciones
    python scripts/ajustar_cupula.py --sin-mujoco                    # el material (E por radio de punzon)
    python scripts/ajustar_cupula.py --radio-punzon 5 --sin-mujoco   # el material con el punzon medido
    python scripts/ajustar_cupula.py --radio-punzon 5                # + el contacto en MuJoCo

QUE RESUELVE. En el modelo la cupula (`fsr_tip`) es una esfera RIGIDA; la
silicona se imita con la blandura del contacto (`solref`/`solimp`). Este script
mide el material en el ensayo de banco y propone esa blandura. NO edita ningun
archivo: imprime los valores, y el cambio a `mechanical/mujoco_assets/contacto.xml`
se hace a mano despues de mirarlos.

EL ENSAYO (`electronics/FSR/ensayo-fsr-20261109-1931.csv`). Un punzon de punta
ESFERICA indenta un domo MACIZO de silicona, 50 ciclos. Columnas: `t_s`;
`z_mm` (posicion del punzon; z = 0 es el toque sin deformar, z > 0 indenta);
`F1_N`, `F2_N` (dos celdas de carga) y `F_N` = su suma; `V_mV`, `R_ohm` del
FSR; `fase` (0 quieto, 1 bajando = carga, 2 pausa abajo = sostenido, 3 subiendo
= descarga, 4 pausa arriba = reposo); `ciclo`.

LO QUE TIENE EL BANCO, medido aca (2026-11-17, maquina A):
  - las celdas dan una muestra nueva cada 100 ms (10 Hz, no los 100 Hz
    declarados);
  - con z quieto en el sostenido la fuerza de las celdas sigue subiendo, de
    13.3 a 15.6 N en ~0.3 s: un atraso de ~0.19 s entre z y la fuerza (o de las
    celdas, o del punzon detras de la consigna: estos datos no los separan);
  - el lazo F(z) sale INVERTIDO (la descarga encima de la carga), imposible
    para un material pasivo, y corregir solo ese atraso no lo arregla: hay
    ademas juego de z, o el FSR -- que podria desempatar -- no sirve arriba,
    porque satura cerca de 15 N y tiene histeresis propia.

POR QUE IGUAL SE PUEDE MEDIR E. En la rama de CARGA el punzon baja a velocidad
constante (0.975 mm/s) desde antes del contacto. Con eso el juego ya esta
recorrido cuando empieza el contacto, y un atraso puro de cualquier lado es
un corrimiento de v*tau. Los defectos del banco se reducen a UN corrimiento
constante de z, que el ajuste absorbe como d0 (el "toque aparente"). E sale de
la carga sin necesidad de saber cual defecto es cual. La descarga, el
sostenido y el FSR quedan como diagnostico, y el factor de perdida se reporta
como "no separable del banco". `--autoprueba` lo demuestra con los tres
defectos a la vez.

EL METODO:
  0 `limpiar`       cero de cada celda por ciclo, desde la fase 4 (sin contacto)
  1 `diagnostico`   tasa de las celdas, atraso en el sostenido, lazo invertido
  2 `ajustar_ley`   Hertz esfera-esfera sobre la CARGA: F = 4/3 E* sqrt(R_ef) (d - d0)^1.5
  3 `por_ciclo`     E en cada ciclo: ablandamiento (Mullins) o asentamiento
  4 `sostenido`     relajacion con z quieto
  5 `curva_plano`   la misma silicona contra un PLANO (lo que pasa en el robot)
  6 (`--sin-mujoco` lo saltea) el contacto de MuJoCo que reproduce esa curva

POR QUE EL PASO 5. El ensayo usa un punzon esferico; en el robot la cupula
apoya sobre superficies PLANAS (piso, escalon). La curva del ensayo no se copia:
se extrae una propiedad del material (E) y con ella se predice la otra
geometria (Johnson, Contact Mechanics, cap. 4).

Castellano, como sus hermanos en `scripts/`. No pasa por CI (ni ruff ni mypy):
correr ruff a mano.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CSV_DEFECTO = Path("electronics/FSR/ensayo-fsr-20261109-1931.csv")
R_CUPULA_MM = 16.5      # el `size` de fsr_tip en MyPalletizer260_fsr.xml, "no medido"
NU_DEFECTO = 0.49       # silicona: casi incompresible (0.48-0.50)

# Por debajo de 1 N no se ajusta. En el banco real, al arrancar la carga y ANTES
# del contacto, las celdas marcan un bulto de hasta 0.48 N: aparece apenas se
# mueve el punzon, culmina a los ~0.56 s (z ~ 0) y decae hasta 0.25 N en
# z = +0.4 mm, con el FSR plano todo ese tramo. No es la cupula (rozamiento de
# la guia o reaccion del arranque: estos datos no lo dicen), y con el umbral
# de 0.2 N que habia aca arruinaba el ajuste (rms log 0.6).
F_MIN_AJUSTE_N = 1.0
BIN_MM = 0.05
DELTA0_GRILLA_MM = np.arange(-1.0, 1.5001, 0.005)
MESETA_S = (0.3, 0.4)   # ventana del sostenido que hace de "meseta" para el atraso
FSR_LINEAL_N = 10.0     # la calibracion G(F) se ajusta por debajo: arriba el FSR satura
E_RAZONABLE_PA = (5e4, 2e6)   # siliconas blandas, Shore 00-10 a A-40 aprox.


# -- datos ------------------------------------------------------------------


@dataclass
class Celda:
    """Las muestras NUEVAS de una celda de carga: las filas donde su valor
    cambia. Entre una y otra el log repite el ultimo valor, asi que solo estas
    son mediciones. El cero ya esta restado."""

    t: np.ndarray        # s, fila en que aparecio la muestra en el log
    F: np.ndarray        # N


@dataclass
class Ensayo:
    """El ensayo limpio, en SI. Una fila por muestra del log (50 Hz)."""

    t: np.ndarray        # s
    z: np.ndarray        # m, posicion REPORTADA del punzon; 0 = toque, > 0 indenta
    G: np.ndarray        # S, conductancia del FSR = 1 / R_ohm
    fase: np.ndarray     # int
    ciclo: np.ndarray    # int
    celdas: tuple[Celda, ...]

    def fuerza(self, tau: float = 0.0) -> np.ndarray:
        """F1 + F2 (N) en cada fila, cada celda interpolada entre sus muestras
        nuevas y adelantada `tau` (s) si se la quiere corregir por atraso.

        Por celda y no sobre la suma: F1 y F2 se actualizan en instantes
        distintos, y su suma mezcla una muestra nueva con una vieja."""
        return np.sum([np.interp(self.t, c.t - tau, c.F) for c in self.celdas], axis=0)


def cargar(path: Path) -> dict[str, np.ndarray]:
    d = np.genfromtxt(path, delimiter=",", names=True)
    return {n: np.asarray(d[n], dtype=np.float64) for n in d.dtype.names}


def limpiar(d: dict[str, np.ndarray]) -> tuple[Ensayo, dict[str, float]]:
    """Paso 0. Sin el ciclo 0 (retroceso inicial). El cero de cada celda se
    interpola en el tiempo entre las fases 4 (sin contacto) que rodean cada
    ciclo: un cero global convertiria la deriva de las celdas en rigidez.
    """
    t, z = d["t_s"], d["z_mm"] * 1e-3
    fase, ciclo = d["fase"].astype(int), d["ciclo"].astype(int)
    celdas, ceros, ruido = [], [], []
    for col in ("F1_N", "F2_N"):
        F = d[col]
        t_c, f_c = [], []
        for c in np.unique(ciclo):
            m = (ciclo == c) & (fase == 4)
            if m.sum() >= 5:
                t_c.append(float(t[m].mean()))
                f_c.append(float(F[m].mean()))
        nueva = np.r_[True, np.diff(F) != 0.0]
        cero = np.interp(t[nueva], np.array(t_c), np.array(f_c))
        celdas.append(Celda(t[nueva], F[nueva] - cero))
        ceros.append(np.array(f_c))
    for c in np.unique(ciclo):
        m = (ciclo == c) & (fase == 4)
        if m.sum() >= 5:
            ruido.append(float(d["F_N"][m].std()))
    R = np.where(d["R_ohm"] > 0.0, d["R_ohm"], np.inf)
    keep = ciclo >= 1
    e = Ensayo(t[keep], z[keep], (1.0 / R)[keep], fase[keep], ciclo[keep], tuple(celdas))
    cero_suma = ceros[0] + ceros[1]
    info = {
        "cero_medio_n": float(cero_suma.mean()),
        "cero_deriva_n": float(cero_suma.max() - cero_suma.min()),
        "ruido_celdas_n": float(np.median(ruido)),
        "n_ciclos": float(np.unique(e.ciclo).size),
    }
    return e, info


# -- paso 1: diagnostico del banco ---------------------------------------------------


@dataclass
class Diagnostico:
    periodo_celdas_s: float
    v_carga: float            # m/s, velocidad del punzon en la carga
    v_carga_cv: float         # su variacion relativa (la carga tiene que ser pareja)
    tau_sostenido: float      # s: (meseta - inicio) / pendiente, con z quieto
    tau_fsr: float            # s: lo mismo para la conductancia del FSR
    lazo_invertido: float     # fraccion de ciclos con la descarga ENCIMA de la carga
    eta_banco: float          # el "factor de perdida" del lazo crudo, NO del material


def _tau_sostenido(e: Ensayo, x: np.ndarray) -> float:
    """Con z quieto, una senal atrasada sigue subiendo hasta su meseta:
    (meseta - inicio) / (pendiente al final de la carga) es el atraso. La meseta
    en MESETA_S y no el maximo: la silicona relaja y el FSR deriva. Si el
    material relaja rapido, esto da de menos."""
    taus = []
    for k in np.unique(e.ciclo):
        h = (e.ciclo == k) & (e.fase == 2)
        lo = (e.ciclo == k) & (e.fase == 1)
        if h.sum() < 50 * MESETA_S[1] or lo.sum() < 10:
            continue
        th = e.t[h] - e.t[h][0]
        tl = e.t[lo]
        w = tl > tl[-1] - 0.4
        pendiente = np.polyfit(tl[w], x[lo][w], 1)[0]
        meseta = np.median(x[h][(th >= MESETA_S[0]) & (th < MESETA_S[1])])
        if pendiente > 0:
            taus.append((meseta - x[h][0]) / pendiente)
    return float(np.median(taus)) if taus else float("nan")


def _eta_ciclos(d: np.ndarray, F: np.ndarray, fase: np.ndarray, ciclo: np.ndarray) -> np.ndarray:
    """eta = W_disipada / (2 pi W_almacenada) por ciclo (Lakes, Viscoelastic
    Materials), con W_alm = trabajo de la carga. Negativo = lazo invertido."""
    etas = []
    for k in np.unique(ciclo):
        ramas = []
        for f in (1, 3):
            m = (ciclo == k) & (fase == f) & (d > 0.0)
            if m.sum() < 5:
                break
            o = np.argsort(d[m])
            ramas.append((d[m][o], F[m][o]))
        if len(ramas) < 2:
            continue
        hi = min(ramas[0][0].max(), ramas[1][0].max())
        g = np.linspace(0.0, hi, 200)
        Fc, Fd = (np.interp(g, dd, ff) for dd, ff in ramas)
        w_alm = float(np.trapezoid(Fc, g))
        if w_alm > 0:
            etas.append((w_alm - float(np.trapezoid(Fd, g))) / (2 * np.pi * w_alm))
    return np.array(etas)


def diagnostico(e: Ensayo) -> Diagnostico:
    periodo = float(np.median([np.median(np.diff(c.t)) for c in e.celdas]))
    m = (e.fase == 1) & (e.z > 0.0)
    v = np.gradient(e.z, e.t)[m]
    tau_c = _tau_sostenido(e, e.fuerza(0.0))
    tau_g = _tau_sostenido(e, e.G)
    F = e.fuerza(tau_c if np.isfinite(tau_c) else 0.0)
    etas = _eta_ciclos(e.z, F, e.fase, e.ciclo)
    return Diagnostico(periodo, float(np.median(v)), float(np.std(v) / np.median(v)),
                       tau_c, tau_g, float(np.mean(etas < 0)), float(np.median(etas)))


# -- paso 2: la ley de contacto, sobre la carga ---------------------------------------------


@dataclass
class Ley:
    A_libre: float     # N / m^n
    n: float
    d0_libre: float    # m
    A_hertz: float     # N / m^1.5
    d0_hertz: float    # m: el toque APARENTE (cero real + corrimiento del banco)
    rms_log_hertz: float
    d_max: float       # m, el mayor hundimiento ajustado (desde d0_hertz)
    F_max: float       # N, la mayor fuerza ajustada


def _binear(d: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Medianas de F en franjas de BIN_MM de z, con al menos 5 muestras."""
    bordes = np.arange(d.min(), d.max() + BIN_MM * 1e-3, BIN_MM * 1e-3)
    idx = np.digitize(d, bordes)
    dc, Fc = [], []
    for i in range(1, bordes.size):
        s = idx == i
        if s.sum() >= 5:
            dc.append(float(np.median(d[s])))
            Fc.append(float(np.median(F[s])))
    return np.array(dc), np.array(Fc)


def _ajuste_log(dc: np.ndarray, Fc: np.ndarray, n_fijo: float | None,
                d0_fijo: float | None = None) -> tuple[float, float, float, float]:
    """min sobre d0 de || log F - log A - n log(d - d0) ||. -> (rms, A, n, d0)."""
    mejor = (np.inf, 0.0, 0.0, 0.0)
    if d0_fijo is not None:
        # d0 impuesto (la variante ingenua): los puntos detras de el no se pueden
        # ajustar con esta ley y se dejan afuera, en vez de anular el ajuste.
        ok = dc > d0_fijo
        dc, Fc = dc[ok], Fc[ok]
    grilla = DELTA0_GRILLA_MM * 1e-3 if d0_fijo is None else np.array([d0_fijo])
    for d0 in grilla:
        x = dc - d0
        if np.any(x <= 0.0):
            continue
        lx, lF = np.log(x), np.log(Fc)
        if n_fijo is None:
            X = np.c_[np.ones(x.size), lx]
            coef, *_ = np.linalg.lstsq(X, lF, rcond=None)
            logA, n = float(coef[0]), float(coef[1])
        else:
            n = n_fijo
            logA = float(np.mean(lF - n * lx))
        r = float(np.sqrt(np.mean((lF - logA - n * lx) ** 2)))
        if r < mejor[0]:
            mejor = (r, float(np.exp(logA)), n, float(d0))
    return mejor


def ajustar_ley(d: np.ndarray, F: np.ndarray, d0_fijo: float | None = None) -> Ley:
    """Ajuste en log de F = A (d - d0)^n (n libre) y con n = 1.5 (Hertz).

    En log porque el error relativo es lo que importa en las dos puntas: con
    cuadrados absolutos los pocos puntos de 15 N mandan sobre todo el arranque,
    que es justo donde vive el umbral del FSR. `d0_fijo` es para la variante
    ingenua de la autoprueba (d0 = 0: creerle al cero del banco).
    """
    dc, Fc = _binear(d, F)
    ok = Fc > F_MIN_AJUSTE_N
    dc, Fc = dc[ok], Fc[ok]
    _, A, n, d0 = _ajuste_log(dc, Fc, None, d0_fijo)
    rms_h, A_h, _, d0_h = _ajuste_log(dc, Fc, 1.5, d0_fijo)
    return Ley(A, n, d0, A_h, d0_h, rms_h, float(dc.max() - d0_h), float(Fc.max()))


def carga(e: Ensayo) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """La rama de carga: (z (m), F (N), ciclo). Sin corregir nada: a velocidad
    constante el atraso y el juego son un corrimiento de z, y d0 lo absorbe."""
    m = e.fase == 1
    return e.z[m], e.fuerza(0.0)[m], e.ciclo[m]


def radio_efectivo(r_cupula: float, r_punzon: float) -> float:
    """1/R_ef = 1/R_c + 1/R_p (dos esferas). R_p = inf es un plano."""
    return 1.0 / (1.0 / r_cupula + (0.0 if np.isinf(r_punzon) else 1.0 / r_punzon))


def modulo_E(A_hertz: float, r_ef: float, nu: float) -> float:
    """E (Pa) desde F = 4/3 E* sqrt(R_ef) d^1.5, E* = E/(1-nu^2) (punzon rigido)."""
    return 3.0 * A_hertz * (1.0 - nu**2) / (4.0 * np.sqrt(r_ef))


def factor_espesor(r_ef: float, d: float, h: float) -> float:
    """Correccion de capa pegada a base rigida (Dimitriadis et al. 2002, nu = 0.5):
    F_medida = F_hertz * f(chi), chi = sqrt(R d)/h. Para un domo macizo de altura
    h ~ R_c es una COTA: el domo no es una capa plana."""
    chi = np.sqrt(r_ef * d) / h
    return float(1 + 1.133 * chi + 1.283 * chi**2 + 0.769 * chi**3 + 0.0975 * chi**4)


# -- pasos 3 y 4 -----------------------------------------------------------------------


def por_ciclo(z: np.ndarray, F: np.ndarray, ciclo: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A de Hertz (N/m^1.5) y d0 (m) por ciclo, cada uno con su propio d0: si el
    cero del banco deriva entre ciclos, no se lee como rigidez. -> (ciclos, A, d0)."""
    ciclos, A, d0 = [], [], []
    for k in np.unique(ciclo):
        m = ciclo == k
        dc, Fc = _binear(z[m], F[m]) if m.sum() > 20 else (np.array([]), np.array([]))
        ok = Fc > F_MIN_AJUSTE_N
        if ok.sum() >= 8:
            _, a, _, d = _ajuste_log(dc[ok], Fc[ok], 1.5)
            ciclos.append(int(k))
            A.append(a)
            d0.append(d)
    return np.array(ciclos), np.array(A), np.array(d0)


def sostenido(e: Ensayo, tau: float) -> tuple[float, float]:
    """Relajacion con z quieto, de la meseta (MESETA_S) al final del sostenido
    (~0.6 s): (celdas, FSR), relativa. > 0 = la fuerza cae (relaja); < 0 = sube."""
    F = e.fuerza(tau if np.isfinite(tau) else 0.0)
    rel_F, rel_G = [], []
    for k in np.unique(e.ciclo):
        m = (e.ciclo == k) & (e.fase == 2)
        if m.sum() < 40:
            continue
        th = e.t[m] - e.t[m][0]
        a, z = (th >= MESETA_S[0]) & (th < MESETA_S[1]), th > th[-1] - 0.1
        rel_F.append(1.0 - F[m][z].mean() / F[m][a].mean())
        rel_G.append(1.0 - e.G[m][z].mean() / e.G[m][a].mean())
    return float(np.median(rel_F)), float(np.median(rel_G))


# -- paso 5 y la calibracion del FSR ---------------------------------------------------


def curva_plano(E: float, nu: float, r_cupula: float, d: np.ndarray) -> np.ndarray:
    """F (N) de la cupula contra un PLANO rigido: Hertz con R_ef = R_c."""
    return 4.0 / 3.0 * E / (1.0 - nu**2) * np.sqrt(r_cupula) * np.power(d, 1.5)


def hundimiento_plano(E: float, nu: float, r_cupula: float, F: np.ndarray) -> np.ndarray:
    """La inversa: d (m) para una fuerza F (N) contra un plano."""
    return np.power(F / (4.0 / 3.0 * E / (1.0 - nu**2) * np.sqrt(r_cupula)), 2.0 / 3.0)


@dataclass
class CalFSR:
    g0: float        # S
    g1: float        # S / N
    r2: float
    pendiente_alta: float   # g1 por encima de FSR_LINEAL_N / g1 abajo: < 1 = satura
    F_inicio: float  # N: fuerza cuando el FSR se despega del reposo, en la carga


def calibrar_fsr(e: Ensayo) -> CalFSR:
    """G = g0 + g1 F (Interlink: conductancia ~lineal con la fuerza), sobre la
    CARGA y por debajo de FSR_LINEAL_N, con el atraso relativo FSR-celdas que
    mejor las alinea. Arriba de eso se reporta cuanto cae la pendiente."""
    m = e.fase == 1
    mejor = (-np.inf, 0.0)
    for tau in np.arange(0.0, 0.501, 0.01):
        F = e.fuerza(tau)[m]
        s = (F > 0.5) & (F < FSR_LINEAL_N)
        if s.sum() > 20:
            r = float(np.corrcoef(F[s], e.G[m][s])[0, 1])
            if r > mejor[0]:
                mejor = (r, float(tau))
    F = e.fuerza(mejor[1])[m]
    G = e.G[m]
    s = (F > 0.5) & (F < FSR_LINEAL_N)
    X = np.c_[np.ones(s.sum()), F[s]]
    coef, *_ = np.linalg.lstsq(X, G[s], rcond=None)
    r2 = 1.0 - float(np.sum((G[s] - X @ coef) ** 2) / np.sum((G[s] - G[s].mean()) ** 2))
    alto = F >= FSR_LINEAL_N
    g1_alto = float(np.polyfit(F[alto], G[alto], 1)[0]) if alto.sum() > 20 else np.nan
    reposo = e.fase == 4
    umbral = float(np.median(e.G[reposo]) + 5.0 * e.G[reposo].std())
    inicios = []
    for k in np.unique(e.ciclo[m]):
        sk = (e.ciclo[m] == k) & (G > umbral)
        if sk.any():
            inicios.append(float(F[sk][0]))
    return CalFSR(float(coef[0]), float(coef[1]), r2, g1_alto / float(coef[1]),
                  float(np.median(inicios)) if inicios else np.nan)


# -- el analisis completo ------------------------------------------------------------------


@dataclass
class Resultado:
    info: dict[str, float]
    diag: Diagnostico
    ley: Ley
    ciclos: np.ndarray
    A_ciclo: np.ndarray
    d0_ciclo: np.ndarray
    relaj: tuple[float, float]
    cal: CalFSR


def analizar(d: dict[str, np.ndarray], ingenuo: bool = False) -> Resultado:
    """Pasos 0-4. `ingenuo`: la variante rota de la autoprueba -- las dos
    ramas juntas y d0 = 0, o sea creerle al cero y al z del banco."""
    e, info = limpiar(d)
    dg = diagnostico(e)
    if ingenuo:
        m = np.isin(e.fase, (1, 3))
        ley = ajustar_ley(e.z[m], e.fuerza(0.0)[m], d0_fijo=0.0)
    else:
        z, F, _ = carga(e)
        ley = ajustar_ley(z, F)
    z, F, c = carga(e)
    ciclos, A, d0 = por_ciclo(z, F, c)
    return Resultado(info, dg, ley, ciclos, A, d0, sostenido(e, dg.tau_sostenido), calibrar_fsr(e))


def reportar(res: Resultado, r_cupula: float, r_punzon: float | None, nu: float) -> float | None:
    """Imprime los pasos 0-5. -> E (Pa) si se conoce el radio del punzon y los
    chequeos pasan; None si no."""
    dg, ley, cal = res.diag, res.ley, res.cal
    print("== 0. limpieza ==")
    print(f"  ciclos: {res.info['n_ciclos']:.0f}; cero de las celdas {res.info['cero_medio_n']:+.2f} N"
          f" (deriva {res.info['cero_deriva_n']:.2f} N); ruido en reposo {res.info['ruido_celdas_n']:.3f} N")
    print("== 1. diagnostico del banco ==")
    print(f"  celdas: una muestra nueva cada {dg.periodo_celdas_s * 1e3:.0f} ms"
          f" ({1 / dg.periodo_celdas_s:.1f} Hz)")
    print(f"  carga a {dg.v_carga * 1e3:.3f} mm/s (variacion {dg.v_carga_cv:.1%}): con esto el"
          f" atraso y el juego son un corrimiento constante de z")
    print(f"  atraso z -> fuerza, por el sostenido: celdas {dg.tau_sostenido:.3f} s,"
          f" FSR {dg.tau_fsr:.3f} s  (= {dg.v_carga * dg.tau_sostenido * 1e3:.2f} mm de z)")
    print(f"  lazo crudo invertido en {dg.lazo_invertido:.0%} de los ciclos (eta del banco"
          f" {dg.eta_banco:+.3f}): la perdida del MATERIAL no se puede separar con este ensayo")
    print("== 2. ley de contacto, rama de CARGA (ensayo: punzon esferico) ==")
    print(f"  libre:  F = {ley.A_libre:.4g} (z - {ley.d0_libre * 1e3:+.3f} mm)^{ley.n:.3f}")
    print(f"  Hertz:  F = {ley.A_hertz:.4g} N/m^1.5 (z - {ley.d0_hertz * 1e3:+.3f} mm)^1.5,"
          f" rms(log) {ley.rms_log_hertz:.3f}")
    print(f"  toque aparente d0 = {ley.d0_hertz * 1e3:+.3f} mm (cero del banco + juego/2 +"
          f" v*atraso); ajustado hasta {ley.d_max * 1e3:.2f} mm y {ley.F_max:.1f} N")
    print("== 3. por ciclo ==")
    temprano = res.ciclos <= res.ciclos.min() + 4
    tarde = res.ciclos >= res.ciclos.max() - 4
    cambio = res.A_ciclo[tarde].mean() / res.A_ciclo[temprano].mean() - 1.0
    print(f"  A de Hertz: ciclos {res.ciclos.min()}-{res.ciclos.min() + 4} {res.A_ciclo[temprano].mean():.4g},"
          f" ultimos 5 {res.A_ciclo[tarde].mean():.4g} ({cambio:+.1%});"
          f" d0 {res.d0_ciclo[temprano].mean() * 1e3:+.3f} -> {res.d0_ciclo[tarde].mean() * 1e3:+.3f} mm")
    print("== 4. sostenido ==")
    print(f"  de la meseta al final (~0.6 s): celdas {res.relaj[0]:+.1%}, FSR {res.relaj[1]:+.1%}"
          f"  (> 0 = relaja)")
    print("== FSR ==")
    print(f"  G = {cal.g0 * 1e6:.3f} uS + {cal.g1 * 1e6:.3f} uS/N * F  (r2 {cal.r2:.3f},"
          f" F < {FSR_LINEAL_N:g} N); arriba la pendiente es {cal.pendiente_alta:.0%} de esa")
    print(f"  despega del reposo a F = {cal.F_inicio:.2f} N")

    chequeos = {
        "carga a velocidad pareja (variacion < 10 %)": dg.v_carga_cv < 0.10,
        "n libre cerca de 1.5 (1.2-1.9)": 1.2 <= ley.n <= 1.9,
        "toque aparente acotado (|d0| < 1 mm)": abs(ley.d0_hertz) < 1e-3,
        "Hertz ajusta la carga (rms log < 0.15)": ley.rms_log_hertz < 0.15,
    }
    E: float | None = None
    if r_punzon is None:
        print("== 2b. E segun el radio del punzon (no se paso --radio-punzon) ==")
        for rp in (2.0, 5.0, 10.0, 20.0, np.inf):
            r_ef = radio_efectivo(r_cupula, rp * 1e-3)
            print(f"  R_p {rp:>5} mm -> R_ef {r_ef * 1e3:5.2f} mm -> E = "
                  f"{modulo_E(ley.A_hertz, r_ef, nu) / 1e3:7.1f} kPa")
    else:
        r_ef = radio_efectivo(r_cupula, r_punzon)
        E = modulo_E(ley.A_hertz, r_ef, nu)
        f_esp = factor_espesor(r_ef, ley.d_max, r_cupula)
        print("== 2b. el material ==")
        print(f"  R_c {r_cupula * 1e3:.1f} mm, R_p {r_punzon * 1e3:.1f} mm -> R_ef {r_ef * 1e3:.2f} mm; nu {nu}")
        print(f"  E = {E / 1e3:.1f} kPa  (neo-Hookeano: mu = E/3 = {E / 3e3:.1f} kPa,"
              f" C10 = {E / 6e3:.1f} kPa)")
        print(f"  d_max / R_ef = {ley.d_max / r_ef:.2f} (Hertz pide ~0.1);"
              f" a/h = {np.sqrt(r_ef * ley.d_max) / r_cupula:.2f}")
        print(f"  cota por espesor finito (Dimitriadis): E podria ser hasta {E / f_esp / 1e3:.1f} kPa"
              f" (factor {f_esp:.2f} en d_max)")
        print("== 5. la misma cupula contra un PLANO (robot) ==")
        for F in (1.0, 5.0, 10.0, 15.0, 25.0, 30.0):
            dd = float(hundimiento_plano(E, nu, r_cupula, np.array([F]))[0])
            extra = "" if F <= ley.F_max else f"   <- extrapolado (el ensayo llega a {ley.F_max:.0f} N)"
            print(f"  F {F:5.1f} N -> hundimiento {dd * 1e3:5.2f} mm{extra}")
        chequeos["E en el rango de siliconas blandas"] = E_RAZONABLE_PA[0] <= E <= E_RAZONABLE_PA[1]
    print("== chequeos ==")
    for nombre, ok in chequeos.items():
        print(f"  [{'ok' if ok else 'NO'}] {nombre}")
    if E is not None and not all(chequeos.values()):
        print("  -> hay chequeos que no pasan: los numeros se muestran, pero NO se propone solref.")
        return None
    return E


# -- autoprueba ---------------------------------------------------------------------------


def sintetico(E: float, nu: float, r_ef: float, tau: float, juego: float,
              ruido: float = 0.03, eta_visc: float = 0.01, tau_relaj: float = 1.0,
              seed: int = 0) -> dict[str, np.ndarray]:
    """Un ensayo con la misma forma que el real, los tres defectos del banco y
    la verdad conocida.

    z reportado: 50 ciclos de -0.5 a 2.7 mm a 1 mm/s, 1 s abajo, sube, 1 s
    arriba. El punzon REAL sigue a z con JUEGO (operador de juego: no se mueve
    hasta que z recorre el juego, que es lo que hace un tornillo con huelgo).
    F verdadera = Hertz sobre el hundimiento real con una histeresis REAL chica
    (eta_visc) que relaja con `tau_relaj` cuando el punzon para. Las celdas
    muestrean F(t - tau) cada 100 ms y retienen el valor, con ruido y un cero
    que deriva. El FSR ve F al dia.
    """
    rng = np.random.default_rng(seed)
    dt, v = 0.02, 1e-3
    A = 4.0 / 3.0 * E / (1.0 - nu**2) * np.sqrt(r_ef)
    filas: list[tuple[float, float, int, int]] = []
    t = 0.0
    for c in range(0, 51):
        if c == 0:
            tramos = [(3, 0.0, -0.5e-3, 0.7)]
        else:
            tramos = [(1, -0.5e-3, 2.7e-3, 3.2e-3 / v), (2, 2.7e-3, 2.7e-3, 1.0),
                      (3, 2.7e-3, -0.5e-3, 3.2e-3 / v), (4, -0.5e-3, -0.5e-3, 1.0)]
        for fase, z0, z1, dur in tramos:
            n = round(dur / dt)
            for i in range(n):
                filas.append((t, z0 + (z1 - z0) * i / n, fase, c))
                t += dt
    tt = np.array([f[0] for f in filas])
    zz = np.array([f[1] for f in filas])
    fase_a = np.array([f[2] for f in filas])
    ciclo_a = np.array([f[3] for f in filas])
    d_real = np.empty_like(zz)
    d_real[0] = zz[0]
    for i in range(1, zz.size):      # operador de juego (play)
        d_real[i] = min(max(d_real[i - 1], zz[i] - juego / 2.0), zz[i] + juego / 2.0)
    vel = np.gradient(d_real, tt)
    visc = np.zeros_like(zz)          # histeresis real: sigue al signo de la velocidad, relaja
    for i in range(1, zz.size):
        objetivo = np.sign(vel[i]) if abs(vel[i]) > 1e-5 else 0.0
        visc[i] = visc[i - 1] + (objetivo - visc[i - 1]) * min(1.0, dt / tau_relaj * 5.0
                                                                 if objetivo else dt / tau_relaj)

    def f_verdad(ti: np.ndarray) -> np.ndarray:
        d = np.interp(ti, tt, d_real)
        sv = np.interp(ti, tt, visc)
        return A * np.power(np.clip(d, 0.0, None), 1.5) * (1.0 + np.pi * eta_visc * sv)

    t_muestra = np.arange(0.0, tt[-1] + 0.1, 0.1)
    medido = f_verdad(t_muestra - tau) + rng.normal(0.0, ruido, t_muestra.size)
    deriva = -0.8 + 0.05 * np.sin(t_muestra / 60.0)
    idx = np.searchsorted(t_muestra, tt + 1e-9, side="right") - 1
    F = (medido + deriva)[idx]
    F1 = 0.5 * F
    Fv = f_verdad(tt)
    G = 2e-6 + 3e-6 * Fv + rng.normal(0.0, 2e-8, tt.size)
    return {"t_s": tt, "z_mm": zz * 1e3, "F1_N": F1, "F2_N": F - F1, "F_N": F,
            "V_mV": 142.0 + 100.0 * Fv, "R_ohm": 1.0 / G, "fase": fase_a.astype(float),
            "ciclo": ciclo_a.astype(float)}


def autoprueba() -> int:
    """Los tres defectos a la vez (atraso, juego, histeresis real). El analisis
    tiene que recuperar E desde la carga y el toque aparente que predice la
    teoria (juego/2 + v*atraso); la variante ingenua (las dos ramas, d0 = 0)
    tiene que fallar, y el diagnostico tiene que ver el lazo invertido."""
    E, nu, r_c, r_p, tau, juego = 300e3, 0.49, 16.5e-3, 5e-3, 0.20, 0.6e-3
    r_ef = radio_efectivo(r_c, r_p)
    d = sintetico(E, nu, r_ef, tau, juego)
    res = analizar(d)
    rota = analizar(d, ingenuo=True)
    E_hat = modulo_E(res.ley.A_hertz, r_ef, nu)
    E_rota = modulo_E(rota.ley.A_hertz, r_ef, nu)
    d0_teoria = juego / 2.0 + 1e-3 * tau
    print(f"verdad: E {E / 1e3:.0f} kPa, atraso {tau:.2f} s, juego {juego * 1e3:.2f} mm"
          f" -> toque aparente {d0_teoria * 1e3:+.2f} mm (+ hasta 0.02 mm del log)")
    print(f"  carga: E {E_hat / 1e3:.1f} kPa ({E_hat / E - 1:+.1%}); n libre {res.ley.n:.3f};"
          f" d0 {res.ley.d0_hertz * 1e3:+.3f} mm; rms log {res.ley.rms_log_hertz:.3f}")
    print(f"  diagnostico: atraso en el sostenido {res.diag.tau_sostenido:.3f} s;"
          f" lazo invertido en {res.diag.lazo_invertido:.0%}")
    print(f"  INGENUO (las dos ramas, d0 = 0): E {E_rota / 1e3:.1f} kPa ({E_rota / E - 1:+.1%})")
    pruebas = {
        "E recuperado dentro del 5 %": abs(E_hat / E - 1.0) < 0.05,
        "n libre cerca de 1.5 (+-0.1)": abs(res.ley.n - 1.5) < 0.1,
        "toque aparente = juego/2 + v*atraso (+-0.05 mm)": abs(res.ley.d0_hertz - d0_teoria) < 0.05e-3 + 2e-5,
        "el diagnostico ve el lazo invertido": res.diag.lazo_invertido > 0.5,
        "FALSACION: el ajuste ingenuo se equivoca en mas del 10 %": abs(E_rota / E - 1.0) > 0.10,
    }
    for nombre, ok in pruebas.items():
        print(f"  [{'ok' if ok else 'FALLA'}] {nombre}")
    return 0 if all(pruebas.values()) else 1


# -- main -----------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--csv", type=Path, default=CSV_DEFECTO)
    ap.add_argument("--radio-punzon", type=float, default=None,
                    help="radio de la punta esferica del punzon, mm (sin el: E por radio)")
    ap.add_argument("--radio-cupula", type=float, default=R_CUPULA_MM,
                    help=f"radio de la cupula, mm (default {R_CUPULA_MM}, el del XML, no medido)")
    ap.add_argument("--nu", type=float, default=NU_DEFECTO, help="Poisson de la silicona")
    ap.add_argument("--sin-mujoco", action="store_true", help="solo el analisis del material")
    ap.add_argument("--autoprueba", action="store_true",
                    help="datos sinteticos con la verdad conocida + la variante ingenua")
    args = ap.parse_args(argv)
    if args.autoprueba:
        return autoprueba()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from erp.io.paths import resolve_repo_path

    path = args.csv if args.csv.is_absolute() else resolve_repo_path(*args.csv.parts)
    res = analizar(cargar(path))
    r_c = args.radio_cupula * 1e-3
    r_p = None if args.radio_punzon is None else args.radio_punzon * 1e-3
    E = reportar(res, r_c, r_p, args.nu)
    if args.sin_mujoco or E is None:
        return 0
    return ajustar_mujoco(E, args.nu, r_c)


def ajustar_mujoco(E: float, nu: float, r_cupula: float) -> Any:
    raise SystemExit("paso 6 (MuJoCo): todavia no implementado; usar --sin-mujoco")


if __name__ == "__main__":
    raise SystemExit(main())
