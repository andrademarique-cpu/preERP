"""Cuanto se mete el estimado del EKF ciego adentro de los solidos. Sin visor.

    python scripts/medir_penetracion.py --escenario libre       # control: 0 mm
    python scripts/medir_penetracion.py --escenario piso        # piso NO modelado
    python scripts/medir_penetracion.py --escenario piso --filtro-modela piso   # A/B
    python scripts/medir_penetracion.py --escenario lateral     # J1 contra el costado de B3
    python scripts/medir_penetracion.py --escenario escaleras   # data/processed/plan_escaleras.csv
    python scripts/medir_penetracion.py --escenario piso --degrade swap        # DEBE recortar
    python scripts/medir_penetracion.py --escenario todos       # la tabla entera
    python scripts/medir_penetracion.py --escenario piso --filtro-modela nada piso --fsr --graficar
                                        # + columna `fsr`: EKFFSR, el FSR como contacto ON/OFF

EL SINTOMA: cuando el brazo toca algo que el modelo del filtro no tiene, el
estimado deriva y la pose estimada atraviesa el solido. El fantasma de
`demo_teleop.py` lo muestra; esto lo cuenta. Corre la MISMA `Teleop` que la demo
(planta + `LiveSimSensor` + EKF ciego + `FilterRunner`), con el teclado
reemplazado por un guion de objetivos, y despues de cada tick pone la pose
verdadera y la estimada en la geometria de la PLANTA y le pide a MuJoCo la
distancia con signo herramienta-obstaculo (`erp.sim.plant.distancias`).

LO QUE SE LE COBRA AL FILTRO ES EL EXCESO sobre la verdad, no la penetracion
cruda: el contacto de MuJoCo es blando y la planta misma se hunde unos mm. Solo
cuenta la herramienta (`contype = 2`): los eslabones atraviesan los solidos
tambien en la planta, y ahi no habria verdad contra la cual comparar.

`--filtro-modela` es la comparacion que importa:
  nada   el filtro sobre el brazo pelado y SIN piso solido (el defecto desde que
         `blind_variant` lo apaga): todo contacto es dinamica no modelada.
  piso   el brazo pelado CON piso solido: el piso modelado, los props no.
  todo   el filtro compilado desde el MISMO envoltorio que la planta: conoce
         cada solido. Solo tiene sentido como cota: el sistema real no lo tiene.

La sigma a lo largo de la normal sale de `propagate_to_site` sobre el sitio
`efector` del modelo del FILTRO. En el escenario con FSR ese sitio esta en la
punta del brazo pelado, no en la cupula (35.1 mm mas abajo): la covarianza es la
del mismo cuerpo rigido, pero el brazo de palanca difiere unos cm. Para ordenes
de magnitud alcanza; para un numero fino, no.

Castellano, como sus hermanos en `scripts/`. No pasa por CI (ni ruff ni mypy).
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_teleop as base
from espejo_brazo import poner_q, suma_tendon

from erp.analysis import ResumenPenetracion, propagate_to_site, resumen_penetracion
from erp.io.paths import resolve_repo_path
from erp.sensors.mujoco import rows_of
from erp.sim.plant import distancias, geoms_por_clase

XML_ARM = base.XML_RELATIVE
XML_FSR = XML_ARM.with_name("MyPalletizer260_fsr.xml")
XML_FSR_ESCALERAS = XML_ARM.with_name("MyPalletizer260_fsr_escaleras.xml")
# ON = la fuerza del <touch> pasa este umbral. El FSR de la tesis de Nipote es
# ruidoso y con umbral por debajo de ~2 N; medido aca: 0 N a 16 mm del piso y
# 24-32 N apoyado, asi que 1 N separa limpio.
FSR_UMBRAL_N = 1.0
# Lo que la cupula se hunde apoyada con el contacto rigido del adaptador
# (solref -1e5 -1e3): medido 0.75-0.88 mm a 24-32 N.
PEN_ON_M = 0.0008
PLAN_ESCALERAS = ("data", "processed", "plan_escaleras.csv")

RETENCION_S = 1.5      # cuanto empuja contra el solido
COLA_S = 2.0           # quieto en home al final: aca se ve si la deriva se queda
UMBRAL_MM = 1.0


# -- guiones ----------------------------------------------------------------


@dataclass
class Guion:
    """Un objetivo de juntas en funcion del tiempo, y la planta donde corre."""

    nombre: str
    planta: Path
    duracion_s: float
    objetivo: Callable[[float], np.ndarray]   # t (s) -> (3,) rad [rot, link1, link2]


def por_paradas(paradas: list[tuple[np.ndarray, float]], q0: np.ndarray) -> tuple[Any, float]:
    """Objetivo escalonado: ir a cada parada y quedarse `espera` s.

    El tiempo de ida es el giro mayor a SLEW_RAD_S (la rampa de `Teleop.jog`),
    con 20 % de margen, asi que cada parada se alcanza antes de pasar a la
    siguiente. Devuelve (objetivo, duracion).
    """
    tiempos, metas, q = [0.0], [q0], q0
    t = 0.0
    for meta, espera in paradas:
        t += 1.2 * float(np.max(np.abs(meta - q))) / base.SLEW_RAD_S + espera
        tiempos.append(t)
        metas.append(meta)
        q = meta
    tiempos_a = np.array(tiempos)

    def objetivo(t: float) -> np.ndarray:
        i = int(np.searchsorted(tiempos_a, t, side="right"))
        return np.asarray(metas[min(i, len(metas) - 1)], dtype=np.float64)

    return objetivo, t + COLA_S


def pose_para(model: Any, rot: float, r: float, z: float, q_ini: np.ndarray) -> np.ndarray:
    """(3,) rad con J1 = `rot` y el sitio `efector` a radio `r`, altura `z` (m).

    Gauss-Newton en (link1, link2), jacobiano por diferencias finitas sobre la
    cinematica con la igualdad del tendon (`poner_q`). Igual idea que
    `planear_escaleras.encima`, sin su `Escena`.
    """
    data = mj.MjData(model)
    l0 = suma_tendon(model)
    x = np.array([rot, q_ini[1], q_ini[2]], dtype=np.float64)

    def rz(v: np.ndarray) -> np.ndarray:
        poner_q(model, data, v, l0)
        p = data.site("efector").xpos
        return np.array([np.hypot(p[0], p[1]), p[2]])

    meta = np.array([r, z])
    for _ in range(50):
        f = rz(x) - meta
        if np.max(np.abs(f)) < 1e-5:
            return x
        jac = np.zeros((2, 2))
        for k in (1, 2):
            dx = x.copy()
            dx[k] += 1e-6
            jac[:, k - 1] = (rz(dx) - rz(x)) / 1e-6
        x[1:] -= np.linalg.solve(jac, f)
    raise RuntimeError(f"sin pose para r = {r:.3f} m, z = {z:.3f} m")


def guion(nombre: str, q0: np.ndarray, con_fsr: bool = False) -> Guion:
    # `con_fsr`: libre y piso corren sobre el brazo CON adaptador (misma dinamica,
    # otra punta y contacto rigido), para que todas las columnas compartan planta.
    brazo = XML_FSR if con_fsr else XML_ARM
    if nombre == "libre":
        # Lejos de todo: ningun punto de este recorrido baja de ~5 cm.
        obj, dur = por_paradas([(np.array([1.0, 0.6, -0.5]), RETENCION_S),
                                (q0, 0.0)], q0)
        return Guion(nombre, brazo, dur, obj)

    if nombre == "piso":
        # link1 al tope (1.57) con link2 en 0: la consigna queda ~4 cm debajo
        # del piso, asi que el servo EMPUJA contra el piso todo el tiempo.
        obj, dur = por_paradas([(np.array([0.0, 1.57, 0.0]), RETENCION_S),
                                (q0, 0.0)], q0)
        return Guion(nombre, brazo, dur, obj)

    if nombre == "lateral":
        # B3 (el mas alto y el mas afuera, az 66 grados) por el costado: bajar
        # la herramienta a media altura del bloque con J1 = 0 (az 90, nada en
        # el medio), y girar J1 hacia el centro del bloque. La planta se frena
        # contra la cara; la consigna sigue empujando.
        model = mj.MjModel.from_xml_path(str(resolve_repo_path(*XML_FSR_ESCALERAS.parts)))
        g = model.geom("esc_B3").id
        cx, cy, cz = (float(v) for v in model.geom_pos[g])
        r = float(np.hypot(cx, cy))
        az_bloque = float(np.arctan2(cy, cx))
        # J1 = 0 apunta a az 90 (medido: la punta en x ~ 0, y > 0).
        rot_bloque = az_bloque - np.pi / 2
        arriba = pose_para(model, 0.0, r, cz + 0.08, q0)
        abajo = pose_para(model, 0.0, r, cz + 0.005, arriba)
        contra = np.array([rot_bloque, abajo[1], abajo[2]])
        obj, dur = por_paradas([(arriba, 0.3), (abajo, 0.5), (contra, RETENCION_S),
                                (abajo, 0.5), (arriba, 0.3), (q0, 0.0)], q0)
        return Guion(nombre, XML_FSR_ESCALERAS, dur, obj)

    if nombre == "escaleras":
        # El plan del banco: baja la cupula sobre cada peldano, o sea que trae
        # contacto A PROPOSITO (el planificador evita choques, no apoyos).
        path = resolve_repo_path(*PLAN_ESCALERAS)
        plan = np.loadtxt(path, delimiter=",", skiprows=1)
        t_plan = plan[:, 0]
        q_plan = np.deg2rad(plan[:, 1:4])      # JMAP es la identidad (demo_teleop_real)

        def obj(t: float) -> np.ndarray:
            return np.array([np.interp(t, t_plan, q_plan[:, j]) for j in range(3)])

        return Guion(nombre, XML_FSR_ESCALERAS, float(t_plan[-1]) + COLA_S, obj)

    raise SystemExit(f"escenario desconocido: {nombre}")


# -- una corrida --------------------------------------------------------------


@dataclass
class Corrida:
    t: np.ndarray            # (N,) s
    nombres: list[str]       # obstaculos
    d_est: np.ndarray        # (N, K) m
    d_true: np.ndarray       # (N, K) m
    normal: np.ndarray       # (N, K, 3) normal del obstaculo en la pose ESTIMADA
    sigma_n: np.ndarray      # (N, K) m, sigma del efector a lo largo de esa normal
    nis: np.ndarray          # (M,) NIS de las actualizaciones aplicadas
    t_nis: np.ndarray        # (M,) s, sello de muestra de cada una
    q_true: np.ndarray       # (N, 3) rad
    q_est: np.ndarray        # (N, 3) rad
    n_on: int = 0            # correcciones del FSR aplicadas en ON
    n_off: int = 0           # y en OFF (estimado adentro con el FSR sin apretar)


def correr(g: Guion, modela: str, degrade: str, umbral_n: float = FSR_UMBRAL_N) -> Corrida:
    """Una corrida. `modela` = nada | piso | todo | fsr.

    `fsr`: dinamica SIN contacto (el brazo con adaptador, que pesa lo mismo) +
    `EKFFSR`, con el mapa = el XML de la planta. Cada tick lee `fsr_force` de la
    planta y llama `update_contacto(fuerza > umbral)`. Simplificacion de tiempo,
    a sabiendas: la correccion se aplica al tiempo actual del filtro, que va un
    horizonte (6 ms) atras de la lectura. Sellarla bien es trabajo de
    `FilterRunner`, y queda para despues.
    """
    plant_xml = resolve_repo_path(*g.planta.parts)
    # El brazo del FILTRO lleva la misma punta que la planta. La dinamica es
    # identica (adaptador sin masa), pero lo que el piso MODELADO toca no: con
    # el brazo pelado su colisionador mas bajo es `ee_collider`, 33 mm arriba de
    # la cupula, y la columna `piso` nunca tocaria nada -- daba bit a bit lo
    # mismo que `nada`, que es como se encontro esto.
    con_adaptador = g.planta in (XML_FSR, XML_FSR_ESCALERAS)
    brazo_filtro = XML_FSR if con_adaptador else XML_ARM
    if modela == "todo":
        blind_xml = plant_xml
    else:
        blind_xml = resolve_repo_path(*brazo_filtro.parts)
    if modela == "fsr" and not con_adaptador:
        raise SystemExit(f"`fsr` necesita una planta con adaptador, no {g.planta.name}")
    teleop = base.Teleop(plant_xml, degrade, blind_xml=blind_xml,
                         floor_collides=modela in ("piso", "todo"),
                         geometria_xml=plant_xml if modela == "fsr" else None,
                         pen_on=PEN_ON_M)
    n_steps = max(1, round((1.0 / base.RENDER_HZ) / teleop.dt))
    model = teleop.model
    data_med = mj.MjData(model)      # PROPIO: nunca escribir el qpos de la planta
    herr, obs = geoms_por_clase(model)
    nv = int(teleop.model_b.nv)
    fila_fsr = rows_of(model, "fsr_force") if modela == "fsr" else None

    ts, de, dv, nn, qt, qe, XH, PP = [], [], [], [], [], [], [], []
    t_nis: list[float] = []
    n_on = n_off = 0
    while teleop.t < g.duracion_s:
        q = g.objetivo(teleop.t)
        for i in range(3):
            teleop.set_target(i, float(q[i]))
        teleop.jog()
        teleop.step(n_steps)
        t_nis.extend(ts_m for ts_m, _, _ in teleop.updates)
        if fila_fsr is not None:
            on = float(teleop.data.sensordata[fila_fsr][0]) > umbral_n
            if teleop.ekf.update_contacto(on) is not None:
                n_on, n_off = n_on + on, n_off + (not on)
        d_true, _ = distancias(model, data_med, teleop.data.qpos, herr, obs)
        d_est, normal = distancias(model, data_med, teleop.ekf.x[:nv], herr, obs)
        ts.append(teleop.t)
        dv.append(d_true)
        de.append(d_est)
        nn.append(normal)
        qt.append(teleop.q_true.copy())
        qe.append(teleop.q_est.copy())
        XH.append(teleop.ekf.x.copy())
        PP.append(teleop.ekf.P.copy())

    rows = rows_of(teleop.model_b, "efector_pos")
    _, C = propagate_to_site(np.array(XH), np.array(PP), teleop.model_b, teleop.data_b, rows)
    normal_a = np.array(nn)
    # sigma_n = sqrt(n^T C n) por obstaculo; normal cero (sin direccion) -> nan.
    var_n = np.einsum("nki,nij,nkj->nk", normal_a, C, normal_a)
    sigma_n = np.where(var_n > 0.0, np.sqrt(np.maximum(var_n, 0.0)), np.nan)
    return Corrida(
        t=np.array(ts), nombres=[model.geom(int(o)).name for o in obs],
        d_est=np.array(de), d_true=np.array(dv), normal=normal_a, sigma_n=sigma_n,
        nis=np.array(teleop.nis), t_nis=np.array(t_nis),
        q_true=np.array(qt), q_est=np.array(qe), n_on=n_on, n_off=n_off,
    )


def resumir(c: Corrida) -> tuple[ResumenPenetracion, dict[str, ResumenPenetracion]]:
    """(global, por obstaculo). Global = el obstaculo mas cercano en cada instante."""
    por_obs = {
        n: resumen_penetracion(c.t, c.d_est[:, k], c.d_true[:, k], c.sigma_n[:, k],
                               UMBRAL_MM / 1e3)
        for k, n in enumerate(c.nombres)
    }
    k_min = np.argmin(c.d_est, axis=1)
    idx = np.arange(len(c.t))
    glob = resumen_penetracion(c.t, c.d_est[idx, k_min], c.d_true.min(axis=1),
                               c.sigma_n[idx, k_min], UMBRAL_MM / 1e3)
    return glob, por_obs


def fila(nombre: str, r: ResumenPenetracion) -> str:
    rec = "inf" if np.isinf(r.recuperacion_s) else f"{r.recuperacion_s:.2f}"
    sig = "-" if np.isnan(r.max_exceso_sigma) else f"{r.max_exceso_sigma:.1f}"
    return (f"{nombre:<26}{r.n_contactos:>4}{r.max_pen_true_mm:>9.2f}{r.max_pen_est_mm:>9.2f}"
            f"{r.max_exceso_mm:>9.2f}{r.integral_mm_s:>10.2f}{r.frac_recorte:>8.3f}"
            f"{rec:>8}{sig:>8}")


ENCABEZADO = (f"{'corrida':<26}{'cont':>4}{'p_true':>9}{'p_est':>9}{'exceso':>9}"
              f"{'mm*s':>10}{'frac':>8}{'recup':>8}{'x sig':>8}\n"
              f"{'':<26}{'':>4}{'mm':>9}{'mm':>9}{'mm':>9}{'':>10}{'':>8}{'s':>8}{'':>8}")


def guardar_csv(c: Corrida, path: Path) -> None:
    cols = ["t"] + [f"d_est_{n}" for n in c.nombres] + [f"d_true_{n}" for n in c.nombres] \
        + ["q_true_rot", "q_true_link1", "q_true_link2", "q_est_rot", "q_est_link1", "q_est_link2"]
    datos = np.column_stack([c.t, c.d_est, c.d_true, c.q_true, c.q_est])
    np.savetxt(path, datos, delimiter=",", header=",".join(cols), comments="", fmt="%.6g")


# Paleta: los slots categoricos 1-5 de la paleta de referencia, validada (CVD y
# vision normal pasan; aqua/amarillo/magenta quedan < 3:1 contra el fondo, por
# eso las curvas de junta llevan rotulo directo ademas de la leyenda).
FONDO = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
GRILLA = "#e6e5e1"
CONTACTO = "#ecebe7"
C_VERDAD = "#2a78d6"     # slot 1
C_ESTIMADO = "#eb6834"   # slot 2
C_JUNTAS = ("#1baf7a", "#eda100", "#e87ba4")   # slots 3-5: rot, link1, link2
NOMBRES_JUNTAS = ("J1 rot", "J2 link1", "J3 link2")
TITULO_MODELA = {
    "nada": "filtro SIN el solido (piso no modelado)",
    "piso": "filtro CON piso solido",
    "todo": "filtro con TODOS los solidos",
    "fsr": "filtro SIN el solido + FSR ON/OFF",
}
D_VISTA_MM = (-12.0, 30.0)   # la distancia satura en DIST_MAX (50 mm): se recorta la vista


def _estilo(ax: Any) -> None:
    ax.set_facecolor(FONDO)
    ax.grid(True, color=GRILLA, lw=0.6)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRILLA)
    ax.tick_params(colors=TINTA_2, labelsize=8, length=0)


def _sombrear_contacto(ax: Any, c: Corrida) -> None:
    contacto = c.d_true.min(axis=1) <= 0.0
    ax.fill_between(c.t, 0, 1, where=contacto, transform=ax.get_xaxis_transform(),
                    color=CONTACTO, lw=0, zorder=0)


def figura_ab(corridas: list[tuple[str, Corrida]], escenario: str) -> Any:
    """Una columna por modelo del filtro; filas: distancia, error de junta, NIS.

    Distancia CON signo (no la penetracion recortada en 0): asi se ve el caso
    que importa, la verdad que sube mientras el estimado sigue abajo de cero.
    La banda es +-2 sigma del estimado a lo largo de la normal; un estimado
    afuera de su propia banda respecto de la verdad es el recorte sobreconfiado.
    Ejes Y compartidos por fila, para que las columnas se comparen a ojo.
    """
    import matplotlib.pyplot as plt

    n = len(corridas)
    fig, axs = plt.subplots(3, n, sharex="col", sharey="row", squeeze=False,
                            figsize=(5.2 * n + 0.6, 7.6),
                            gridspec_kw={"height_ratios": [1.4, 1.0, 0.8]})
    fig.patch.set_facecolor(FONDO)
    rotulos: list[tuple[Any, float, np.ndarray]] = []   # (eje, t final, error final por junta)

    for j, (modela, c) in enumerate(corridas):
        k = np.argmin(c.d_est, axis=1)
        idx = np.arange(len(c.t))
        d_est = c.d_est[idx, k] * 1e3
        d_true = c.d_true.min(axis=1) * 1e3
        sig = np.nan_to_num(c.sigma_n[idx, k]) * 1e3
        r, _ = resumir(c)
        ax_d, ax_q, ax_n = axs[:, j]
        for ax in (ax_d, ax_q, ax_n):
            _estilo(ax)
            _sombrear_contacto(ax, c)

        # -- fila 1: distancia con signo
        ax_d.axhspan(-1e3, 0.0, color="#f6f5f2", lw=0, zorder=0)
        ax_d.axhline(0.0, color=TINTA_2, lw=0.8)
        ax_d.fill_between(c.t, d_est - 2 * sig, d_est + 2 * sig, color=C_ESTIMADO,
                          alpha=0.18, lw=0, label="estimado ±2σ")
        ax_d.plot(c.t, d_true, color=C_VERDAD, lw=2, label="verdad")
        ax_d.plot(c.t, d_est, color=C_ESTIMADO, lw=2, label="estimado")
        # Abajo se abre hasta el punto mas hondo, para que el marcador no quede afuera.
        ax_d.set_ylim(min(D_VISTA_MM[0], float(np.min(d_est)) * 1.15), D_VISTA_MM[1])
        ax_d.text(0.01, 0.03, "adentro del sólido", transform=ax_d.transAxes,
                  color=TINTA_2, fontsize=8, va="bottom")
        # El peor exceso, marcado y rotulado: es la cifra de la corrida.
        exceso = np.maximum(0.0, -d_est) - np.maximum(0.0, -d_true)
        i_max = int(np.argmax(exceso))
        if exceso[i_max] > UMBRAL_MM:
            ax_d.plot([c.t[i_max]] * 2, [d_true[i_max] if d_true[i_max] < 0 else 0.0,
                                          d_est[i_max]], color=TINTA, lw=1)
            ax_d.plot(c.t[i_max], d_est[i_max], "o", ms=8, mfc=C_ESTIMADO,
                      mec=FONDO, mew=2, zorder=5)
            sig_txt = ("" if np.isnan(r.max_exceso_sigma)
                       else f" = {r.max_exceso_sigma:.1f}σ")
            ax_d.annotate(f"{exceso[i_max]:.1f} mm adentro{sig_txt}",
                          (c.t[i_max], d_est[i_max]), xytext=(8, -14),
                          textcoords="offset points", color=TINTA, fontsize=9)
        ax_d.set_title(TITULO_MODELA.get(modela, modela), color=TINTA, fontsize=10,
                       loc="left", pad=18)
        rec = "∞" if np.isinf(r.recuperacion_s) else f"{r.recuperacion_s:.2f} s"
        ax_d.text(0.0, 1.02, f"exceso máx {r.max_exceso_mm:.2f} mm · "
                  f"{r.integral_mm_s:.2f} mm·s · recuperación {rec}",
                  transform=ax_d.transAxes, color=TINTA_2, fontsize=8)

        # -- fila 2: error de junta, rotulado al final de cada curva
        err = np.rad2deg(c.q_est - c.q_true)
        for i in range(3):
            ax_q.plot(c.t, err[:, i], color=C_JUNTAS[i], lw=2, label=NOMBRES_JUNTAS[i])
        ax_q.axhline(0.0, color=TINTA_2, lw=0.8)
        rotulos.append((ax_q, c.t[-1], err[-1]))

        # -- fila 3: NIS por actualizacion, escala log
        ax_n.plot(c.t_nis, c.nis, "o", ms=3, color=TINTA_2, label="NIS")
        ax_n.axhline(12.0, color=TINTA_2, lw=0.8, ls="--")
        ax_n.text(c.t[-1] * 1.21, 12.0, "objetivo 12", color=TINTA_2, fontsize=8,
                  va="bottom", ha="right",
                  bbox={"facecolor": FONDO, "edgecolor": "none", "pad": 1})
        ax_n.set_yscale("log")
        ax_n.set_xlabel("t [s]", color=TINTA_2, fontsize=9)
        ax_n.set_xlim(c.t[0], c.t[-1] * 1.22)    # sitio para los rotulos de junta

    axs[0, 0].set_ylabel("distancia herramienta–sólido [mm]", color=TINTA_2, fontsize=9)
    axs[1, 0].set_ylabel("error de junta [°]", color=TINTA_2, fontsize=9)
    axs[2, 0].set_ylabel("NIS (log)", color=TINTA_2, fontsize=9)

    # Leyenda unica arriba: las mismas series en todas las columnas.
    from matplotlib.patches import Patch

    h, lab = axs[0, 0].get_legend_handles_labels()
    hq, labq = axs[1, 0].get_legend_handles_labels()
    h += hq + [Patch(color=CONTACTO)]
    lab += labq + ["contacto verdadero"]
    fig.legend(h, lab, loc="upper center", ncol=len(lab), frameon=False, fontsize=8,
               labelcolor=TINTA_2, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(f"Recorte del estimado en sólidos — escenario «{escenario}»",
                 color=TINTA, fontsize=12, x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.935))

    # Rotulos directos al final de cada curva de junta, separados en vertical
    # para que no se pisen cuando los errores finales son parecidos. Despues del
    # layout, porque la separacion se mide en pixeles del eje ya dibujado.
    fig.canvas.draw()
    for ax, t_fin, e_fin in rotulos:
        y_px = ax.transData.transform(np.column_stack([np.full(3, t_fin), e_fin]))[:, 1]
        orden = np.argsort(y_px)
        min_px = 11.0 * fig.dpi / 72.0
        pos = y_px.copy()
        for a, b in itertools.pairwise(orden):
            pos[b] = max(pos[b], pos[a] + min_px)
        pos -= (pos[orden].mean() - y_px[orden].mean())   # recentrar el grupo
        for i in range(3):
            ax.annotate(f"{NOMBRES_JUNTAS[i]} {e_fin[i]:+.1f}°", (t_fin, e_fin[i]),
                        xytext=(6, (pos[i] - y_px[i]) * 72.0 / fig.dpi),
                        textcoords="offset points", color=TINTA_2, fontsize=8,
                        va="center")
    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--escenario", default="piso",
                    choices=("libre", "piso", "lateral", "escaleras", "todos"))
    ap.add_argument("--filtro-modela", nargs="+", default=["nada"],
                    choices=("nada", "piso", "todo", "fsr"),
                    help="uno o varios; con --graficar, una columna por cada uno")
    ap.add_argument("--degrade", default="none", choices=base.DEGRADATIONS)
    ap.add_argument("--csv", type=Path, default=None, help="serie temporal de UNA corrida")
    ap.add_argument("--graficar", action="store_true",
                    help="figura por escenario, columnas = modelos del filtro ([viz])")
    ap.add_argument("--png", type=Path, default=None,
                    help="carpeta donde guardar las figuras en vez de abrir ventanas")
    ap.add_argument("--fsr", action="store_true",
                    help="agrega la columna `fsr` (EKFFSR) y pone el adaptador en la planta")
    ap.add_argument("--fsr-umbral", type=float, default=FSR_UMBRAL_N,
                    help="fuerza del <touch> que cuenta como ON, N")
    args = ap.parse_args(argv)

    q0 = np.asarray(base.Teleop(resolve_repo_path(*XML_ARM.parts), "none").q0)
    escenarios = (("libre", "piso", "lateral", "escaleras") if args.escenario == "todos"
                  else (args.escenario,))
    modelos = (("nada", "piso", "todo") if args.escenario == "todos"
               else tuple(dict.fromkeys(args.filtro_modela)))
    if args.fsr and "fsr" not in modelos:
        modelos = (*modelos, "fsr")
    con_fsr = "fsr" in modelos
    planes = [(e, m) for e in escenarios for m in modelos]
    figuras: dict[str, list[tuple[str, Corrida]]] = {}

    print(ENCABEZADO)
    for esc, modela in planes:
        g = guion(esc, q0, con_fsr)
        c = correr(g, modela, args.degrade, args.fsr_umbral)
        glob, por_obs = resumir(c)
        etiqueta = f"{esc}/{modela}" + ("" if args.degrade == "none" else f"/{args.degrade}")
        print(fila(etiqueta, glob))
        for n, r in por_obs.items():
            if r.n_contactos or r.max_exceso_mm > UMBRAL_MM:
                print(fila(f"  {n}", r))
        med = float(np.median(c.nis)) if len(c.nis) else float("nan")
        print(f"{'':<26}NIS mediana {med:.2f}, max {c.nis.max(initial=0):.1f}, "
              f"n = {len(c.nis)};  |q_est - q_true| final "
              f"{np.rad2deg(np.abs(c.q_est[-1] - c.q_true[-1])).round(2)} grados")
        if modela == "fsr":
            print(f"{'':<26}FSR: {c.n_on} correcciones en ON, {c.n_off} en OFF")
        if args.csv and len(planes) == 1:
            guardar_csv(c, args.csv)
            print(f"serie en {args.csv}")
        if args.graficar or args.png:
            figuras.setdefault(esc, []).append((modela, c))

    if figuras:
        import matplotlib

        if args.png:
            matplotlib.use("Agg")      # sin ventana: solo archivos
        import matplotlib.pyplot as plt

        for esc, corridas in figuras.items():
            fig = figura_ab(corridas, esc)
            if args.png:
                args.png.mkdir(parents=True, exist_ok=True)
                sufijo = "" if args.degrade == "none" else f"_{args.degrade}"
                path = args.png / f"penetracion_{esc}{sufijo}.png"
                fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
                print(f"figura en {path}")
        if not args.png:
            plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
