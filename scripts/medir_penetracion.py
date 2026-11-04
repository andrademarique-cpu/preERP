"""Cuanto se mete el estimado del EKF ciego adentro de los solidos. Sin visor.

    python scripts/medir_penetracion.py --escenario libre       # control: 0 mm
    python scripts/medir_penetracion.py --escenario piso        # piso NO modelado
    python scripts/medir_penetracion.py --escenario piso --filtro-modela piso   # A/B
    python scripts/medir_penetracion.py --escenario lateral     # J1 contra el costado de B3
    python scripts/medir_penetracion.py --escenario escaleras   # data/processed/plan_escaleras.csv
    python scripts/medir_penetracion.py --escenario piso --degrade swap        # DEBE recortar
    python scripts/medir_penetracion.py --escenario todos       # la tabla entera

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
XML_FSR_ESCALERAS = XML_ARM.with_name("MyPalletizer260_fsr_escaleras.xml")
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


def guion(nombre: str, q0: np.ndarray) -> Guion:
    if nombre == "libre":
        # Lejos de todo: ningun punto de este recorrido baja de ~5 cm.
        obj, dur = por_paradas([(np.array([1.0, 0.6, -0.5]), RETENCION_S),
                                (q0, 0.0)], q0)
        return Guion(nombre, XML_ARM, dur, obj)

    if nombre == "piso":
        # link1 al tope (1.57) con link2 en 0: la consigna queda ~4 cm debajo
        # del piso, asi que el servo EMPUJA contra el piso todo el tiempo.
        obj, dur = por_paradas([(np.array([0.0, 1.57, 0.0]), RETENCION_S),
                                (q0, 0.0)], q0)
        return Guion(nombre, XML_ARM, dur, obj)

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
    q_true: np.ndarray       # (N, 3) rad
    q_est: np.ndarray        # (N, 3) rad


def correr(g: Guion, modela: str, degrade: str) -> Corrida:
    plant_xml = resolve_repo_path(*g.planta.parts)
    blind_xml = plant_xml if modela == "todo" else resolve_repo_path(*XML_ARM.parts)
    teleop = base.Teleop(plant_xml, degrade, blind_xml=blind_xml,
                         floor_collides=modela in ("piso", "todo"))
    n_steps = max(1, round((1.0 / base.RENDER_HZ) / teleop.dt))
    model = teleop.model
    data_med = mj.MjData(model)      # PROPIO: nunca escribir el qpos de la planta
    herr, obs = geoms_por_clase(model)
    nv = int(teleop.model_b.nv)

    ts, de, dv, nn, qt, qe, XH, PP = [], [], [], [], [], [], [], []
    while teleop.t < g.duracion_s:
        q = g.objetivo(teleop.t)
        for i in range(3):
            teleop.set_target(i, float(q[i]))
        teleop.jog()
        teleop.step(n_steps)
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
        nis=np.array(teleop.nis), q_true=np.array(qt), q_est=np.array(qe),
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


def graficar(c: Corrida, titulo: str) -> None:
    import matplotlib.pyplot as plt

    p_est = np.maximum(0.0, -c.d_est.min(axis=1)) * 1e3
    p_true = np.maximum(0.0, -c.d_true.min(axis=1)) * 1e3
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 5))
    contacto = c.d_true.min(axis=1) <= 0.0
    for a in ax:
        a.fill_between(c.t, 0, 1, where=contacto, transform=a.get_xaxis_transform(),
                       alpha=0.15, label="contacto verdadero")
    ax[0].plot(c.t, p_true, label="verdad")
    ax[0].plot(c.t, p_est, label="estimado")
    ax[0].set_ylabel("penetracion [mm]")
    ax[0].legend()
    ax[1].plot(c.t, p_est - p_true)
    ax[1].axhline(UMBRAL_MM, ls="--", lw=0.8)
    ax[1].set_ylabel("exceso [mm]")
    ax[1].set_xlabel("t [s]")
    fig.suptitle(titulo)
    fig.tight_layout()
    plt.show()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--escenario", default="piso",
                    choices=("libre", "piso", "lateral", "escaleras", "todos"))
    ap.add_argument("--filtro-modela", default="nada", choices=("nada", "piso", "todo"))
    ap.add_argument("--degrade", default="none", choices=base.DEGRADATIONS)
    ap.add_argument("--csv", type=Path, default=None, help="serie temporal de UNA corrida")
    ap.add_argument("--graficar", action="store_true", help="matplotlib ([viz])")
    args = ap.parse_args(argv)

    q0 = np.asarray(base.Teleop(resolve_repo_path(*XML_ARM.parts), "none").q0)
    if args.escenario == "todos":
        planes = [(e, m) for e in ("libre", "piso", "lateral", "escaleras")
                  for m in ("nada", "piso", "todo")]
    else:
        planes = [(args.escenario, args.filtro_modela)]

    print(ENCABEZADO)
    for esc, modela in planes:
        g = guion(esc, q0)
        c = correr(g, modela, args.degrade)
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
        if args.csv and len(planes) == 1:
            guardar_csv(c, args.csv)
            print(f"serie en {args.csv}")
        if args.graficar:
            graficar(c, etiqueta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
