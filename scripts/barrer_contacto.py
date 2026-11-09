"""Barrido de solref/solimp de la HERRAMIENTA: el contacto mas rigido que sigue estable.

    python scripts/barrer_contacto.py

Sin flags y sin visor. El resultado NO se usa desde aca: el ganador se escribe a
mano en `mechanical/mujoco_assets/contacto.xml`, que es la configuracion de
contacto de TODA simulacion. Este script es solo la herramienta para elegirlo.

POR QUE ALCANZA CON LA HERRAMIENTA. Lo unico del brazo que choca es la clase
herramienta (contype = 2: `ee_collider`, `fsr_cyl`, `fsr_tip`); piso y props son
pasivos (conaffinity = 2). Con `priority` mas alta, los parametros de la
herramienta mandan ENTEROS en cada contacto -- sin ella MuJoCo los mezcla con
los del solido (0.02 1 por defecto). Asi que un solo juego vale para piso,
escaleras, caja, y para el piso MODELADO del filtro, que usa los mismos geoms.

QUE MIDE, solo la planta (sin filtro), con los mismos guiones que
`medir_penetracion.py` y la misma rampa de consigna que `Teleop.jog`:

  pen_hold   mediana de la penetracion mas honda de la herramienta, apoyada, mm
  pen_max    pico, mm (el impacto)
  flicker    fraccion de pasos SIN fuerza en la parte ASENTADA de un apoyo
             (huecos <= 20 ms, sin los ASENTAR_S del impacto y de la salida).
             Es el rebote a dt = 2 ms: la fuerza parpadea a 0
  chatter    rms(f - media movil de 20 ms) / media(f), asentado, sin bordes de
             convolucion: oscilacion rapida
  cyl        fraccion de la fuerza normal que cae en `fsr_cyl` (fuera del
             `<touch>`); solo en las plantas con adaptador
  warn       avisos del solver (mjData.warning) + NaN

REGLA. Pasa si, en TODAS las corridas, flicker == 0, chatter < CHATTER_MAX,
warn == 0, y en el piso vertical cyl < CYL_MAX. Gana el de menor pen_hold en el
piso con adaptador (que es el `pen_on` del EKFFSR) entre los que pasan Y cuyo
vecino mas rigido (k siguiente, mismo zeta y solimp) tambien pasa: el borde de
estabilidad depende de la masa efectiva, que cambia con la pose, y un margen de
una muesca es lo que separa "estable en estos guiones" de "estable".

FALSACION. k = 1e5 con zeta 1.6 y 2.5 rebotaba en la primera pasada; se
corre igual y TIENE que fallar. Si pasa, el script sale con 2 y no da ganador.

Castellano, como sus hermanos en `scripts/`. No pasa por CI (ni ruff ni mypy).
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_teleop as base
import medir_penetracion as mp

from erp.io.paths import resolve_repo_path
from erp.sim.plant import geoms_por_clase, load_model
from erp.trajectory import slew_limit

# -- la grilla ----------------------------------------------------------------
# solref NEGATIVO = (-k, -b): rigidez y amortiguamiento directos, 1/s^2 y 1/s.
# b sale de un zeta: b = 2 zeta sqrt(k) (zeta = 1 critico). El actual, -1e5 -1e3,
# es zeta 1.58.
#
# Segunda pasada (2026-11-09), afinada alrededor del borde que encontro la
# primera (60 candidatos, k 1e5-3e5, zeta 0.7-2.5, solimp 0.9-0.999): el limite
# lo pone el AMORTIGUAMIENTO y cae en k dt^2 + 2 b dt ~ 4; todo zeta 2.5 rebota
# y el -1e5 -1e3 de antes estaba en el borde (rebota en `lateral` con el
# ee_collider rigido). solimp 0.99 0.999 baja la penetracion ~5x sin
# desestabilizar.
K = (1.0e5, 1.25e5, 1.5e5, 1.75e5, 2.0e5, 2.25e5, 2.5e5)
ZETA = (0.5, 0.7, 0.85, 1.0)
# (d0, dmax): impedancia del contacto, cuanto resiste vs cuanto se regulariza.
# Ancho 0.001 m, mid 0.5, power 2 como el defecto de MuJoCo.
SOLIMP = ((0.95, 0.99), (0.99, 0.999), (0.995, 0.9995))
# FALSACION: estos TIENEN que fallar (rebotaban en la primera pasada). Si pasan,
# la regla esta mal y el ganador no vale.
FALSOS_ZETA = (1.6, 2.5)
FALSOS_K = 1.0e5

CHATTER_MAX = 0.10
CYL_MAX = 0.02
HUECO_MAX_PASOS = 10          # 20 ms: un hueco mas largo es un apoyo nuevo, no rebote
ASENTAR_S = 0.10              # se descarta el impacto antes de medir pen_hold y chatter

# (nombre del guion, con adaptador). `piso` en las dos puntas: el ee_collider es
# un cilindro de cara plana contra un plano, el caso de varios contactos a la vez.
# `caja`: el brazo pelado apoyando el ee_collider sobre la caja (MyPalletizer260_box.xml),
# la ultima planta que el ee_collider toca.
CORRIDAS = (("piso", True), ("piso", False), ("lateral", True), ("escaleras", True),
            ("caja", False))
CORRIDA_PEN_ON = ("piso", True)
XML_CAJA = mp.XML_ARM.with_name("MyPalletizer260_box.xml")


@dataclass(frozen=True)
class Contacto:
    k: float
    zeta: float
    solimp: tuple[float, float]

    @property
    def solref(self) -> tuple[float, float]:
        return (-self.k, -2.0 * self.zeta * np.sqrt(self.k))

    def margen(self, dt: float) -> float:
        """k dt^2 + 2 b dt: la primera pasada rebota por encima de ~4 (a dt = 2 ms)."""
        return self.k * dt**2 + 2.0 * (2.0 * self.zeta * np.sqrt(self.k)) * dt

    def etiqueta(self) -> str:
        sr = self.solref
        return (f"{sr[0]:.2e} {sr[1]:5.0f} | {self.solimp[0]} {self.solimp[1]} "
                f"[{self.margen(0.002):.2f}]")


def aplicar(model: Any, c: Contacto | None) -> None:
    """Escribe el contacto en los geoms de la herramienta. None = el XML tal cual."""
    if c is None:
        return
    herr, _ = geoms_por_clase(model)
    model.geom_priority[herr] = 1
    model.geom_solref[herr] = c.solref
    model.geom_solimp[herr] = (c.solimp[0], c.solimp[1], 0.001, 0.5, 2.0)


@dataclass
class Medida:
    pen_hold_mm: float
    pen_max_mm: float
    flicker: float
    chatter: float
    cyl: float
    warn: int
    apoyado_s: float

    def pasa(self, vertical: bool) -> bool:
        ok = (self.flicker == 0.0 and self.chatter < CHATTER_MAX and self.warn == 0
              and self.apoyado_s > 0.0)
        if vertical and not np.isnan(self.cyl):
            ok = ok and self.cyl < CYL_MAX
        return ok


def _apoyos(con: np.ndarray) -> list[tuple[int, int]]:
    """Tramos [i0, i1) con contacto, uniendo huecos de <= HUECO_MAX_PASOS."""
    idx = np.flatnonzero(con)
    if idx.size == 0:
        return []
    tramos, i0, prev = [], int(idx[0]), int(idx[0])
    for i in idx[1:]:
        if i - prev > HUECO_MAX_PASOS + 1:
            tramos.append((i0, prev + 1))
            i0 = int(i)
        prev = int(i)
    tramos.append((i0, prev + 1))
    return tramos


def guion_caja(q0: np.ndarray) -> mp.Guion:
    """Bajar el ee_collider sobre la caja y empujar 1 cm por debajo de la tapa.

    La caja esta en (0, 0.160) con la tapa a z = 0.080 (prop_box.xml), o sea az
    90 grados = J1 0. El efector del brazo pelado esta en la punta del
    ee_collider. Si 1 cm abajo no tiene pose, se prueba 5 mm: igual empuja.
    """
    model = mj.MjModel.from_xml_path(str(resolve_repo_path(*XML_CAJA.parts)))
    g = model.geom("box_geom").id
    cx, cy, cz = (float(v) for v in model.geom_pos[g])
    tapa = cz + float(model.geom_size[g][2])
    r = float(np.hypot(cx, cy))
    arriba = mp.pose_para(model, 0.0, r, tapa + 0.04, q0)
    for hondo in (0.010, 0.005):
        try:
            abajo = mp.pose_para(model, 0.0, r, tapa - hondo, arriba)
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError("sin pose para empujar la caja")
    obj, dur = mp.por_paradas([(arriba, 0.3), (abajo, mp.RETENCION_S), (arriba, 0.3),
                               (q0, 0.0)], q0)
    return mp.Guion("caja", XML_CAJA, dur, obj)


def correr(nombre: str, con_fsr: bool, c: Contacto | None) -> Medida:
    model0, _ = load_model(resolve_repo_path(*mp.XML_ARM.parts))
    q0 = np.asarray(model0.key("home").ctrl[:3], dtype=np.float64)
    g = guion_caja(q0) if nombre == "caja" else mp.guion(nombre, q0, con_fsr=con_fsr)
    model, data = load_model(resolve_repo_path(*g.planta.parts))
    # Como `Teleop.set_target`: el objetivo se recorta a `ctrlrange`.
    lo, hi = np.asarray(model.actuator_ctrlrange, dtype=np.float64)[:3].T
    aplicar(model, c)
    herr, _ = geoms_por_clase(model)
    herr_set = {int(h) for h in herr}
    cyl_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "fsr_cyl")

    ctrl = np.asarray(data.ctrl[:3], dtype=np.float64).copy()
    for _ in range(base.WARMUP_STEPS):
        data.ctrl[:3] = ctrl
        mj.mj_step(model, data)
    warn0 = int(sum(w.number for w in data.warning))

    dt = float(model.opt.timestep)
    n_tick = max(1, round((1.0 / base.RENDER_HZ) / dt))
    tick_s = 1.0 / base.RENDER_HZ
    pen, f_tot, f_cyl = [], [], []
    f6 = np.zeros(6)
    t, nan = 0.0, False
    while t < g.duracion_s:
        ctrl = slew_limit(ctrl, np.clip(g.objetivo(t), lo, hi), base.SLEW_RAD_S * tick_s)
        for _ in range(n_tick):
            data.ctrl[:3] = ctrl
            mj.mj_step(model, data)
            t += dt
            p, ft, fc = 0.0, 0.0, 0.0
            for i in range(int(data.ncon)):
                ct = data.contact[i]
                g1, g2 = int(ct.geom1), int(ct.geom2)
                if g1 not in herr_set and g2 not in herr_set:
                    continue
                mj.mj_contactForce(model, data, i, f6)
                fn = max(0.0, float(f6[0]))
                p = max(p, -float(ct.dist))
                ft += fn
                if cyl_id >= 0 and cyl_id in (g1, g2):
                    fc += fn
            pen.append(p)
            f_tot.append(ft)
            f_cyl.append(fc)
        if not np.all(np.isfinite(data.qpos)):
            nan = True
            break

    pen_a, f_a, fc_a = np.array(pen), np.array(f_tot), np.array(f_cyl)
    warn = int(sum(w.number for w in data.warning)) - warn0 + int(nan)
    tramos = _apoyos(f_a > 0.0)
    asentar = round(ASENTAR_S / dt)
    # Todo se mide en la parte ASENTADA de cada apoyo (sin ASENTAR_S en cada
    # punta): la primera pasada contaba el micro-rebote del impacto como
    # flicker (0.02-0.03 en TODO candidato estable, el XML incluido) y el borde
    # de la convolucion "same" como chatter (0.12 en las escaleras para todos).
    huecos = largo = 0
    quietos: list[np.ndarray] = []
    chat: list[float] = []
    n_media = HUECO_MAX_PASOS
    for i0, i1 in tramos:
        if i1 - i0 <= 2 * asentar + n_media:
            continue
        sl = slice(i0 + asentar, i1 - asentar)
        fs = f_a[sl]
        huecos += int(np.sum(fs == 0.0))
        largo += fs.size
        quietos.append(pen_a[sl])
        media = np.convolve(fs, np.ones(n_media) / n_media, mode="valid")
        centro = fs[(n_media - 1) // 2:(n_media - 1) // 2 + media.size]
        if fs.mean() > 0.0:
            chat.append(float(np.sqrt(np.mean((centro - media) ** 2)) / fs.mean()))
    pen_q = np.concatenate(quietos) if quietos else np.array([np.nan])
    return Medida(
        pen_hold_mm=float(np.nanmedian(pen_q) * 1e3),
        pen_max_mm=float(pen_a.max(initial=0.0) * 1e3),
        flicker=huecos / largo if largo else 0.0,
        chatter=max(chat) if chat else 0.0,
        cyl=float(fc_a.sum() / f_a.sum()) if cyl_id >= 0 and f_a.sum() > 0 else float("nan"),
        warn=warn,
        apoyado_s=largo * dt,
    )


def fila(etq: str, ms: dict[tuple[str, bool], Medida], ok: bool) -> str:
    celdas = []
    for corrida in CORRIDAS:
        m = ms[corrida]
        cyl = "  -  " if np.isnan(m.cyl) else f"{m.cyl:5.2f}"
        celdas.append(f"{m.pen_hold_mm:5.2f}/{m.pen_max_mm:5.2f} {m.flicker:5.3f} "
                      f"{m.chatter:5.3f} {cyl} {m.warn:2d}")
    return f"{etq:<28}{'OK ' if ok else '-- '}" + " | ".join(celdas)


def main() -> int:
    falsos = [Contacto(FALSOS_K, z, s) for s in SOLIMP for z in FALSOS_ZETA]
    cands: list[Contacto | None] = [None] + [
        Contacto(k, z, s) for s, z, k in itertools.product(SOLIMP, ZETA, K)
    ] + list(falsos)
    enc = " | ".join(f"{n}{'+fsr' if f else ''}: pen hold/max  flick chatt  cyl  w"
                     for n, f in CORRIDAS)
    print(f"{'solref | solimp (d0 dmax)':<28}   {enc}")
    resultados: dict[Contacto | None, tuple[dict[tuple[str, bool], Medida], bool]] = {}
    for c in cands:
        ms = {corrida: correr(*corrida, c) for corrida in CORRIDAS}
        ok = all(m.pasa(vertical=corrida[0] == "piso") for corrida, m in ms.items())
        resultados[c] = (ms, ok)
        print(fila("XML actual" if c is None else c.etiqueta(), ms, ok), flush=True)

    def vecino(c: Contacto) -> Contacto | None:
        i = K.index(c.k)
        return Contacto(K[i + 1], c.zeta, c.solimp) if i + 1 < len(K) else None

    elegibles = []
    for c, (ms, ok) in resultados.items():
        if c is None or not ok or c in falsos:
            continue
        v = vecino(c)
        if v is not None and resultados[v][1]:
            elegibles.append((ms[CORRIDA_PEN_ON].pen_hold_mm, ms[CORRIDA_PEN_ON].chatter, c))

    print()
    falsos_pasan = [c for c in falsos if resultados[c][1]]
    if falsos_pasan:
        print("FALSACION ROTA: pasan " + ", ".join(c.etiqueta() for c in falsos_pasan)
              + " -- rebotaban en la primera pasada. La regla no discrimina; "
              "no creer el ganador.")
        return 2
    print(f"falsacion ok: los {len(falsos)} candidatos que rebotaban fallan.")
    if not elegibles:
        print("Ningun candidato pasa con margen.")
        return 1
    pen, chat, c = min(elegibles, key=lambda e: (e[0], e[1]))
    sr = c.solref
    print(f"GANADOR: solref=\"{sr[0]:.4g} {sr[1]:.4g}\" "
          f"solimp=\"{c.solimp[0]} {c.solimp[1]} 0.001 0.5 2\"  "
          f"(zeta {c.zeta}; pen_hold piso+fsr {pen:.2f} mm, chatter {chat:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
