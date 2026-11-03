"""Recorrido por los 6 peldanos con splines cubicos, sin chocar con las escaleras.

    python scripts/planear_escaleras.py               # planea, revisa y guarda el plan
    python scripts/planear_escaleras.py --ver         # ademas lo muestra en MuJoCo
    python scripts/planear_escaleras.py --ver --ritmo 0.3   # en camara lenta
    python scripts/planear_escaleras.py --ver --cerca       # camara pegada a la goma
    python scripts/planear_escaleras.py --sin-evitar  # falsacion: sin subir, DEBE fallar

QUE HACE. Recorre home -> A1 -> A2 -> A3 -> B1 -> B2 -> B3 -> home, apoyando la
punta en cada peldano. Los peldanos son los ANGULOS MEDIDOS a mano con la punta
del FSR puesta (el `stair_poses_promedio*.csv` mas reciente), no una cinematica
inversa: lo que el brazo real hizo para tocar cada peldano es la mejor meta que
hay, y no hereda el error del largo de la punta del modelo (ver abajo).

Cada traslado entre dos peldanos es un SPLINE CUBICO EN ANGULOS DE JUNTA
(`erp.trajectory.clamped_cubic_spline`) por cuatro nudos:

    P_a  -- peldano de salida, quieto (velocidad 0)
    U_a  -- encima de P_a, a la altura segura `--altura` (sin frenar)
    U_b  -- encima de P_b, a la misma altura (sin frenar)
    P_b  -- peldano de llegada, quieto (velocidad 0)

`U` se obtiene moviendo J2/J3 con J1 fijo hasta que la punta quede a la altura
segura sobre el MISMO (x, y): subir y bajar en vertical, viajar por arriba. Los
tiempos entre nudos salen del mayor giro de junta a `--vel` grados/s.

COMO SE REVISA. El plan se muestrea a 200 Hz y en cada muestra se pone la pose
en el modelo (cinematica, `mj_forward`) y se mide con `mj_geomDistance` la
distancia de cada geom de la herramienta (`contype=2`: el cilindro viejo, el
cilindro del FSR y la cupula) a cada bloque y al tablero. La regla:

  - tablero y bloques ajenos: distancia >= `--margen` (10 mm por defecto);
  - el bloque de salida o de llegada, mientras la cupula esta ENCIMA de el
    (dentro de su huella mas `HOLGURA_XY_M` y no mas de `TOL_APOYO_M` abajo
    de su techo): la cupula puede tocarlo -- es el objetivo -- y el resto de
    la herramienta solo tiene que no atravesarlo (>= 0).

Si un traslado no cumple -- choque o limites de junta --, se reintenta FRENANDO
en U (tres tramos de reposo a reposo, mas rectos y sin pasarse de largo), y
despues subiendo la altura segura de a 1 cm.

Ademas se pasa todo el plan por `JMAP.validate` con la envolvente J3(J2): cajas
de la API, rangos del XML, envolvente medida y velocidad pico <= 120 grados/s.

POR QUE LA TOLERANCIA DE APOYO NO ES CERO. Con los angulos medidos, la cupula
del MODELO queda 2.8 a 6.2 mm adentro de cada bloque: el adaptador del modelo
(36.5 mm, medidas "PROVISORIAS" en MyPalletizer260_fsr.xml) es ~4 mm mas largo
que el real. El error va para el lado SEGURO en el resto del plan: el modelo
cree que la punta esta mas abajo de lo que esta, asi que el margen real es
mayor que el que se reporta.

LO QUE NO HACE. No manda nada al robot: solo planea, revisa y guarda
`data/processed/plan_escaleras.csv` (t, J1..J4 en grados API, a 25 Hz). Y no
sabe del retraso de ~395 ms del brazo real ni de su seguimiento: el plan es la
consigna, no lo que el brazo va a hacer exactamente. El lazo con ventana no
tiene tests; el spline si (`software/tests/test_trajectory.py`).
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco as mj
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_teleop_real import jmap_with_envelope
from espejo_brazo import (
    MAX_APRIETE_MM,
    XML_FSR_ESCALERAS,
    clasificar,
    pintar,
    poner_q,
    suma_tendon,
    ultimo_csv,
)

from erp.io.paths import resolve_repo_path
from erp.sim.plant import load_model
from erp.trajectory import clamped_cubic_spline, resample

ORDEN = ("A1", "A2", "A3", "B1", "B2", "B3")
FS_REVISION_HZ = 200.0     # muestreo para revisar choques
RATE_PLAN_HZ = 25.0        # el CSV guardado: la tasa a la que se le manda al brazo
T_MIN_S = 0.4              # ningun tramo entre nudos dura menos que esto
HOLGURA_XY_M = 0.012       # los peldanos medidos caen a 3-10 mm del centro del bloque
TOL_APOYO_M = 0.008        # la cupula del modelo entra hasta 6.2 mm (punta larga)
DIST_MAX_M = 0.05          # mj_geomDistance no busca mas lejos que esto
VERDE = np.array([0.1, 0.8, 0.2, 0.8], dtype=np.float32)
ROJO = np.array([0.95, 0.1, 0.1, 1.0], dtype=np.float32)


@dataclass
class Escena:
    """Modelo cargado y los ids que la revision necesita."""

    model: Any
    data: Any
    l0: float
    herramienta: list[int]
    cupula: int
    bloques: dict[str, int]
    piso: int

    @classmethod
    def cargar(cls, xml: Path) -> Escena:
        model, data = load_model(resolve_repo_path(*xml.parts))
        herr = [g for g in range(model.ngeom) if int(model.geom_contype[g]) == 2]
        nombres = {model.geom(g).name: g for g in range(model.ngeom)}
        cupula = nombres.get("fsr_tip", nombres["ee_collider"])
        bloques = {n[4:]: g for n, g in nombres.items() if n.startswith("esc_")}
        if set(bloques) != set(ORDEN):
            raise SystemExit(f"{xml.name} no trae los 6 bloques esc_A1..esc_B3: {sorted(bloques)}")
        return cls(model, data, suma_tendon(model), herr, cupula, bloques, nombres["floor"])

    def punta(self, q: np.ndarray) -> np.ndarray:
        """(3,) posicion del sitio `efector` (la punta de la cupula), m, marco mundo."""
        poner_q(self.model, self.data, q, self.l0)
        return np.asarray(self.data.site("efector").xpos, dtype=np.float64).copy()

    def techo(self, bloque: str) -> tuple[np.ndarray, float]:
        """((x, y) del centro, z del techo) de un bloque, m."""
        g = self.bloques[bloque]
        pos, size = self.model.geom_pos[g], self.model.geom_size[g]
        return np.array(pos[:2], dtype=np.float64), float(pos[2] + size[2])


def encima(esc: Escena, q: np.ndarray, z_obj: float) -> np.ndarray:
    """Misma J1 y mismo (x, y) de la punta que `q`, con la punta a altura `z_obj`.

    Gauss-Newton sobre (radio horizontal, z) con J2/J3 libres y jacobiano por
    diferencias finitas en el modelo: dos incognitas, converge en pocas vueltas.
    """
    p0 = esc.punta(q)
    meta = np.array([np.hypot(p0[0], p0[1]), z_obj])
    x = np.array(q, dtype=np.float64)

    def rz(v: np.ndarray) -> np.ndarray:
        p = esc.punta(v)
        return np.array([np.hypot(p[0], p[1]), p[2]])

    for _ in range(30):
        f = rz(x) - meta
        if np.max(np.abs(f)) < 1e-5:
            return x
        jac = np.zeros((2, 2))
        for k in (1, 2):
            dx = x.copy()
            dx[k] += 1e-5
            jac[:, k - 1] = (rz(dx) - rz(x)) / 1e-5
        x[1:] -= np.linalg.solve(jac, f)
    raise RuntimeError(f"no se encontro pose a z = {z_obj * 100:.1f} cm sobre q = {np.rad2deg(q)}")


@dataclass
class Tramo:
    """Un traslado entre dos paradas, con su resultado de la revision."""

    desde: str
    hasta: str
    t: np.ndarray
    q: np.ndarray
    modo: str
    altura_m: float
    holgura_mm: float = np.inf
    peor: str = ""
    malas: list[float] = field(default_factory=list)
    limites: str = ""   # vacio si el tramo respeta cajas, XML y envolvente

    @property
    def ok(self) -> bool:
        return not self.malas and not self.limites


def tiempos(nudos: list[np.ndarray], vel_rad_s: float) -> np.ndarray:
    """Tiempo de cada nudo: el mayor giro de junta entre nudos a `vel_rad_s`."""
    dur = [max(T_MIN_S, float(np.max(np.abs(b - a))) / vel_rad_s)
           for a, b in itertools.pairwise(nudos)]
    return np.concatenate([[0.0], np.cumsum(dur)])


def spline(nudos: list[np.ndarray], vel_rad_s: float, frenar: bool) -> tuple[np.ndarray, ...]:
    """(t, q) muestreado a FS_REVISION_HZ. `frenar`: velocidad 0 en cada nudo."""
    tk = tiempos(nudos, vel_rad_s)
    qk = np.array(nudos)
    if not frenar:
        t = np.arange(0.0, tk[-1], 1.0 / FS_REVISION_HZ)
        t = np.append(t, tk[-1])
        return t, clamped_cubic_spline(tk, qk, t)[0]
    ts, qs = [], []
    for i in range(len(nudos) - 1):
        t = np.arange(tk[i], tk[i + 1], 1.0 / FS_REVISION_HZ)
        ts.append(t)
        qs.append(clamped_cubic_spline(tk[i:i + 2], qk[i:i + 2], t)[0])
    ts.append(np.array([tk[-1]]))
    qs.append(qk[-1:])
    return np.concatenate(ts), np.vstack(qs)


def revisar(esc: Escena, tramo: Tramo, margen_m: float) -> None:
    """Llena `holgura_mm`, `peor` y `malas` del tramo segun la regla del docstring."""
    m, d = esc.model, esc.data
    extremos = [b for b in (tramo.desde, tramo.hasta) if b in esc.bloques]
    holgura, peor, malas = np.inf, "", []
    for t, q in zip(tramo.t, tramo.q, strict=True):
        p = esc.punta(q)
        ok = True
        for nombre, g_obs in [("tablero", esc.piso), *esc.bloques.items()]:
            apoyo = False
            if nombre in extremos:
                c, z_techo = esc.techo(nombre)
                apoyo = (np.max(np.abs(p[:2] - c)) <= 0.015 + HOLGURA_XY_M
                         and p[2] >= z_techo - TOL_APOYO_M)
            for g in esc.herramienta:
                dist = float(mj.mj_geomDistance(m, d, g, g_obs, DIST_MAX_M, None))
                if apoyo:
                    if g == esc.cupula:
                        continue          # tocar el peldano es el objetivo
                    exigido = 0.0         # el resto no lo atraviesa
                else:
                    exigido = margen_m
                    if dist < holgura:
                        holgura, peor = dist, f"{nombre} ({m.geom(g).name})"
                if dist < exigido:
                    ok = False
        if not ok:
            malas.append(float(t))
    tramo.holgura_mm, tramo.peor, tramo.malas = holgura * 1000.0, peor, malas


def revisar_limites(jmap: Any, esc: Escena, tramo: Tramo) -> None:
    """Llena `limites` con lo que `JointMap.validate` objete, o lo deja vacio.

    Por tramo y no solo al final, porque es lo que decide el reintento: un
    spline que PASA por un nudo con velocidad se puede pasar de largo. Medido:
    con los peldanos 1 a 15 cm de la base, subir la punta a 12 cm pide J2 ~1
    grado, y el spline suave baja J2 hasta -5.7 (la API llega a -2). Frenando
    en cada nudo cada pedazo es un cubico de reposo a reposo, monotono entre
    sus extremos, y no se pasa.
    """
    try:
        jmap.validate(tramo.q, esc.model, tramo.t, verbose=False)
        tramo.limites = ""
    except ValueError as err:
        tramo.limites = str(err).splitlines()[1].strip(" -")


def planear_tramo(esc: Escena, jmap: Any, desde: str, qa: np.ndarray, hasta: str,
                  qb: np.ndarray, args: argparse.Namespace) -> Tramo:
    """El primer plan que pasa la revision: suave, despues frenando, despues mas alto."""
    vel = np.deg2rad(args.vel)
    if args.sin_evitar:
        t, q = spline([qa, qb], vel, frenar=False)
        tr = Tramo(desde, hasta, t, q, "directo", 0.0)
        revisar(esc, tr, args.margen / 1000.0)
        revisar_limites(jmap, esc, tr)
        return tr
    tr = None
    for subida in range(4):
        z = args.altura / 100.0 + 0.01 * subida
        nudos = [qa]
        if desde != "home":
            nudos.append(encima(esc, qa, z))
        if hasta != "home":
            nudos.append(encima(esc, qb, z))
        nudos.append(qb)
        for frenar in (False, True):
            t, q = spline(nudos, vel, frenar)
            tr = Tramo(desde, hasta, t, q, "frenando en U" if frenar else "suave", z)
            revisar(esc, tr, args.margen / 1000.0)
            revisar_limites(jmap, esc, tr)
            if tr.ok:
                return tr
    assert tr is not None
    return tr


def unir(tramos: list[Tramo]) -> tuple[np.ndarray, np.ndarray]:
    """Concatena los tramos en un solo (t, q), sin repetir las paradas."""
    ts, qs, t0 = [], [], 0.0
    for i, tr in enumerate(tramos):
        sl = slice(0 if i == 0 else 1, None)
        ts.append(tr.t[sl] + t0)
        qs.append(tr.q[sl])
        t0 += tr.t[-1]
    return np.concatenate(ts), np.vstack(qs)


def debajo(esc: Escena, p: np.ndarray) -> tuple[str, float]:
    """(nombre, z del techo en m) de la superficie bajo la punta: un peldano o el tablero.

    Un peldano cuenta si la punta esta sobre su huella mas `HOLGURA_XY_M`, la
    misma regla con la que `revisar` decide que la cupula esta "encima".
    """
    for nombre in esc.bloques:
        c, z_techo = esc.techo(nombre)
        if np.max(np.abs(p[:2] - c)) <= 0.015 + HOLGURA_XY_M:
            return nombre, z_techo
    return "tablero", 0.0


def cupula_aplastada(r: float, c: float) -> tuple[float, float]:
    """(radio horizontal, alto) de la media cupula de radio `r` aplastada `c`, en m.

    La silicona es casi incompresible, asi que se conserva el VOLUMEN de la
    media esfera, (2/3) pi r^3, con una media elipsoide de alto h = r - c:
    a^2 h = r^3, a = r sqrt(r / h). Es geometria, no un modelo del material:
    dice cuanto tiene que aplastarse la goma para que la pose sea posible, no
    con que fuerza. Esa la mide el FSR.
    """
    h = max(r - c, 0.15 * r)   # no dibujar una cupula de alto cero
    return r * float(np.sqrt(r / h)), h


def color_deformacion(c_mm: float) -> np.ndarray:
    """Gris sin deformar -> naranja a 3 mm -> rojo desde MAX_APRIETE_MM."""
    gris = np.array([0.2, 0.2, 0.22, 1.0])
    naranja = np.array([1.0, 0.55, 0.1, 1.0])
    rojo = np.array([0.95, 0.1, 0.1, 1.0])
    if c_mm <= 3.0:
        f = c_mm / 3.0
        return np.asarray(gris + f * (naranja - gris), dtype=np.float32)
    f = min(1.0, (c_mm - 3.0) / (MAX_APRIETE_MM - 3.0))
    return np.asarray(naranja + f * (rojo - naranja), dtype=np.float32)


def mostrar(esc: Escena, t: np.ndarray, q: np.ndarray, tramos: list[Tramo],
            ritmo: float = 1.0, cerca: bool = False) -> None:
    """Reproduce el plan en bucle, con la estela de la punta y la goma aplastada.

    La cupula del modelo es una esfera RIGIDA: al apretar, se mete en el
    peldano. Aca se la esconde mientras toca y en su lugar se dibuja la media
    cupula APLASTADA contra la superficie (`cupula_aplastada`), coloreada por
    cuanto se deforma (`color_deformacion`), con la cifra en el texto. La
    deformacion es cuanto queda la punta rigida por debajo de la superficie.

    `ritmo` < 1 reproduce en camara lenta; `cerca` hace que la camara siga la
    punta de cerca.
    """
    m, d = esc.model, esc.data
    colores = {g: m.geom_rgba[g].copy() for g in esc.bloques.values()}
    r_cupula = float(m.geom_size[esc.cupula][0])
    rgba_cupula = m.geom_rgba[esc.cupula].copy()
    malas = {round(x, 6) for x in unir_malas(tramos)}
    paso = max(1, len(t) // 400)
    estela = [(esc.punta(q[i]), round(float(t[i]), 6) in malas) for i in range(0, len(t), paso)]
    with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
        scn = v.user_scn
        n_estela = min(len(estela), scn.maxgeom - 1)   # el ultimo lugar es la linea
        for i in range(n_estela):
            p, mala = estela[i]
            mj.mjv_initGeom(scn.geoms[i], mj.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.002, 0.0, 0.0]), p, np.eye(3).flatten(),
                            ROJO if mala else VERDE)
        scn.ngeom = n_estela
        if cerca:
            v.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = m.geom_bodyid[esc.cupula]
            v.cam.distance = 0.18
            v.cam.elevation = -8.0
        print("Ventana abierta: el plan se repite en bucle. Cerrala para terminar.")
        t0 = time.perf_counter()
        while v.is_running():
            ahora = ((time.perf_counter() - t0) * ritmo) % t[-1]
            i = min(int(np.searchsorted(t, ahora)), len(t) - 1)
            with v.lock():
                poner_q(m, d, q[i], esc.l0)
                estados = clasificar(m, d)
                pintar(m, colores, estados)
                p = np.array(d.site("efector").xpos, dtype=np.float64)
                sup, z_sup = debajo(esc, p)
                hueco_mm = (p[2] - z_sup) * 1000.0
                if hueco_mm < 0.0:
                    # La goma toca: esconder la esfera rigida y dibujarla aplastada.
                    c = -hueco_mm / 1000.0
                    a, h = cupula_aplastada(r_cupula, c)
                    centro = np.array(d.geom_xpos[esc.cupula], dtype=np.float64)
                    m.geom_rgba[esc.cupula][3] = 0.0
                    mj.mjv_initGeom(scn.geoms[n_estela], mj.mjtGeom.mjGEOM_ELLIPSOID,
                                    np.array([a, a, h]), centro,
                                    np.asarray(d.geom_xmat[esc.cupula], dtype=np.float64),
                                    color_deformacion(-hueco_mm))
                    scn.ngeom = n_estela + 1
                else:
                    m.geom_rgba[esc.cupula] = rgba_cupula
                    scn.ngeom = n_estela
                deg = np.rad2deg(q[i])
                lineas = [f"t {ahora:5.1f} / {t[-1]:.1f} s",
                          f"J1 {deg[0]:6.1f}  J2 {deg[1]:6.1f}  J3 {deg[2]:6.1f}",
                          f"punta a {p[2] * 1000:6.1f} mm del tablero"]
                nombre = sup if sup == "tablero" else f"peldano {sup} (techo {z_sup * 100:.0f} cm)"
                if hueco_mm >= 0.0:
                    lineas.append(f"sobre {nombre}: faltan {hueco_mm:5.1f} mm")
                else:
                    pct = -hueco_mm / (r_cupula * 1000.0) * 100.0
                    lineas.append(f"apoyada en {nombre}")
                    lineas.append(f"GOMA DEFORMADA: {-hueco_mm:4.1f} mm"
                                  f" ({pct:.0f}% del alto de la cupula)")
                lineas += [f"CHOQUE {k.removeprefix('esc_')}: {mm:.1f} mm"
                           for k, (e, mm) in sorted(estados.items()) if e == "choque"]
                v.set_texts((mj.mjtFontScale.mjFONTSCALE_150, mj.mjtGridPos.mjGRID_TOPLEFT,
                             "\n".join(lineas), None))
            v.sync()
            time.sleep(1.0 / 60.0)


def unir_malas(tramos: list[Tramo]) -> list[float]:
    """Instantes con choque, en el tiempo del plan completo."""
    out, t0 = [], 0.0
    for tr in tramos:
        out += [x + t0 for x in tr.malas]
        t0 += tr.t[-1]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=None,
                    help="angulos medidos de los peldanos (default: el mas reciente)")
    ap.add_argument("--altura", type=float, default=12.0,
                    help="altura segura de la punta para viajar, cm sobre el tablero")
    ap.add_argument("--margen", type=float, default=10.0,
                    help="distancia minima a tablero y bloques, mm")
    ap.add_argument("--vel", type=float, default=30.0,
                    help="velocidad de junta para repartir tiempos, grados/s (el brazo da 120)")
    ap.add_argument("--ver", action="store_true", help="mostrar el plan en MuJoCo")
    ap.add_argument("--ritmo", type=float, default=1.0,
                    help="velocidad de la reproduccion con --ver: 0.5 = camara lenta a la mitad")
    ap.add_argument("--cerca", action="store_true",
                    help="con --ver: la camara sigue la punta de cerca para ver la goma")
    ap.add_argument("--sin-evitar", action="store_true",
                    help="falsacion: ir directo de peldano a peldano, sin subir. Debe fallar")
    ap.add_argument("--no-guardar", action="store_true", help="no escribir el CSV del plan")
    args = ap.parse_args()

    esc = Escena.cargar(XML_FSR_ESCALERAS)
    jmap = jmap_with_envelope(real_arm=False)
    csv = args.csv or ultimo_csv()
    if csv is None:
        raise SystemExit("no hay data/raw/stair_poses_promedio*.csv: medir los peldanos primero")
    filas = np.genfromtxt(csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    metas = {f"{r['escalera']}{r['peldano']}": jmap.to_model_rad([[r["j1"], r["j2"], r["j3"]]])[0]
             for r in np.atleast_1d(filas)}
    falta = [p for p in ORDEN if p not in metas]
    if falta:
        raise SystemExit(f"{csv.name} no tiene los peldanos {falta}")
    print(f"peldanos de {csv.name}; modelo {XML_FSR_ESCALERAS.name}")

    paradas = [("home", np.zeros(3)), *[(p, metas[p]) for p in ORDEN], ("home", np.zeros(3))]
    tramos = [planear_tramo(esc, jmap, a, qa, b, qb, args)
              for (a, qa), (b, qb) in itertools.pairwise(paradas)]

    print(f"\n{'tramo':<12}{'modo':<15}{'altura':>8}{'dura':>7}{'holgura':>10}  cerca de")
    for tr in tramos:
        estado = "" if not tr.malas else f"   CHOCA en {len(tr.malas)} muestras"
        if tr.limites:
            estado += f"   LIMITES: {tr.limites}"
        alt = f"{tr.altura_m * 100:.0f} cm" if tr.altura_m else "-"
        print(f"{tr.desde + '->' + tr.hasta:<12}{tr.modo:<15}{alt:>8}{tr.t[-1]:6.1f}s"
              f"{tr.holgura_mm:8.1f}mm  {tr.peor}{estado}")
    t, q = unir(tramos)
    print(f"total {t[-1]:.1f} s, {len(t)} muestras a {FS_REVISION_HZ:.0f} Hz\n")

    fallas = [tr for tr in tramos if not tr.ok]
    try:
        jmap.validate(q, esc.model, t)
        limites_ok = True
    except ValueError as err:
        print(err)
        limites_ok = False

    if fallas or not limites_ok:
        print(f"\nPLAN RECHAZADO: {len(fallas)} tramo(s) con choque o fuera de limites"
              f"{'' if limites_ok else ', limites violados'}. No se guarda.")
    elif not args.no_guardar:
        t25, q25 = resample(t, q, RATE_PLAN_HZ)
        api = jmap.to_api_deg(q25)
        salida = resolve_repo_path("data", "processed", "plan_escaleras.csv", must_exist=False)
        np.savetxt(salida, np.column_stack([t25, api]), delimiter=",",
                   header="t,j1,j2,j3,j4", comments="", fmt="%.4f")
        print(f"\nPLAN OK. Guardado en {salida} ({len(t25)} consignas a {RATE_PLAN_HZ:.0f} Hz,"
              " grados API)")
    else:
        print("\nPLAN OK.")

    if args.ver:
        mostrar(esc, t, q, tramos, args.ritmo, args.cerca)
    return 1 if (fallas or not limites_ok) else 0


if __name__ == "__main__":
    sys.exit(main())
