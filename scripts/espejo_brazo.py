"""Espejo del MyPalletizer260 real en MuJoCo: lo moves a mano y el modelo lo copia.

    python scripts/espejo_brazo.py                    # en vivo por COM7, suelta los servos
    python scripts/espejo_brazo.py --puerto COM6
    python scripts/espejo_brazo.py --no-soltar        # solo lee; el brazo queda firme
    python scripts/espejo_brazo.py --sin-brazo        # sin hardware: recorre las poses grabadas

QUE HACE. Lee `get_angles()` del brazo a ~15 Hz, pasa los grados API a radianes
del modelo con el `JMAP` de `demo_teleop_real.py` (no una copia mas: ya hay tres)
y pone esa pose en el modelo SIN simular fisica -- `mj_forward`, nunca `mj_step`.
Es cinematica pura: el modelo esta donde el brazo dice que esta, y nada mas.

Ademas dibuja una esfera verde en cada punto donde el MODELO pone la punta para
las poses de `data/raw/stair_poses_promedio*.csv` (las de las escaleras, medidas
a mano). Si al apoyar la herramienta en un peldano real la punta del modelo cae
sobre su esfera, el modelo y la lectura coinciden en ese punto.

LO UNICO QUE VIAJA POR EL PUERTO es `get_angles()`, `release_all_servos()` y
`focus_servo()`. Ningun comando de movimiento, igual que la celda de la
envolvente y la de las escaleras en `notebooks/Palletizer.ipynb`.

POR QUE SE ESCRIBE `qpos` Y NO ES LA TRAMPA DE `demo_teleop`. Alla escribir
`qpos` rompe la igualdad del tendon y la simulacion siguiente da 345 m/s^2. Aca
no se simula, y `act` se calcula para cumplir la igualdad: q_link1 + q_link2 +
q_act = L0, con L0 leido del keyframe `home` del propio modelo (1.5693). Medido:
residuo de la igualdad 3.7e-06 en cualquier pose, el mismo que en `home`.

LO QUE NO MUESTRA. J4 (el giro de la punta) no esta en el modelo y se ignora.
Las juntas del modelo tienen rangos (`link2` 0..60 grados) que el brazo real
excede: en las escaleras J3 llego a -21 grados. Como no se simula, el modelo
dibuja la pose igual; que el XML la declare imposible es un hallazgo sobre el
XML, no un error del espejo. Y el espejo no dice nada del EKF: no hay IMUs aca.

El lazo con ventana no tiene tests -- necesita pantalla y un brazo. Las funciones
de pose y de marcas si se pueden probar sin ventana.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import mujoco as mj
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_teleop_real import JMAP

from erp.io.paths import resolve_repo_path
from erp.sim.plant import load_model

XML_RELATIVE = Path("mechanical/mujoco_assets/MyPalletizer260/MyPalletizer260.xml")
RATE_HZ = 15.0          # lectura del brazo; la misma tasa que la envolvente
RADIO_MARCA_M = 0.006   # esfera de cada peldano, radio en metros
VERDE = np.array([0.1, 0.8, 0.2, 0.6], dtype=np.float32)


def leer(mc: Any) -> np.ndarray | None:
    """(4,) grados API, o None si la respuesta no vino o vino corta.

    `get_angles()` devuelve -1 ante un error, por eso el tipo se mira antes que
    el largo (un int no tiene len). Misma regla que `MyPalletizerArm.read`.
    """
    a = mc.get_angles()
    if not isinstance(a, (list, tuple)) or len(a) < 3:
        return None
    q = np.full(4, np.nan)
    q[: min(4, len(a))] = a[:4]
    return q


def suma_tendon(model: Any) -> float:
    """L0 = q_link1 + q_link2 + q_act que cumple la igualdad, en radianes.

    Leido del keyframe `home` en vez de copiado: si el XML cambia, esto sigue.
    """
    k = model.key("home").qpos
    return float(k[1] + k[2] + k[3])


def poner_pose(model: Any, data: Any, api_deg: np.ndarray, l0: float) -> None:
    """Pone en `data` la pose de (4,) grados API y recalcula la cinematica.

    `qpos` = [rot, link1, link2, act]; `act` sale de la igualdad del tendon.
    Velocidades en cero: es una foto, no un estado dinamico.
    """
    rot, link1, link2 = JMAP.to_model_rad(api_deg[None, :3])[0]
    data.qpos[:4] = [rot, link1, link2, l0 - link1 - link2]
    data.qvel[:] = 0.0
    mj.mj_forward(model, data)


def puntos_peldanos(model: Any, csv: Path | None) -> list[tuple[str, np.ndarray]]:
    """(nombre, posicion de la punta en m, marco mundo) por fila del CSV de promedios."""
    if csv is None or not csv.exists():
        return []
    filas = np.genfromtxt(csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    d = mj.MjData(model)
    l0 = suma_tendon(model)
    out = []
    for r in np.atleast_1d(filas):
        poner_pose(model, d, np.array([r["j1"], r["j2"], r["j3"], 0.0]), l0)
        out.append((f"{r['escalera']}{r['peldano']}", d.site("efector").xpos.copy()))
    return out


def dibujar_marcas(scn: Any, puntos: list[tuple[str, np.ndarray]]) -> None:
    """Una esfera translucida por peldano en `scn` (el `user_scn` del visor)."""
    n = min(len(puntos), scn.maxgeom)
    for i in range(n):
        mj.mjv_initGeom(scn.geoms[i], mj.mjtGeom.mjGEOM_SPHERE,
                        np.array([RADIO_MARCA_M, 0.0, 0.0]), puntos[i][1],
                        np.eye(3).flatten(), VERDE)
    scn.ngeom = n


def texto(data: Any, api_deg: np.ndarray | None, puntos: list[tuple[str, np.ndarray]],
          fallas: int) -> str:
    """Lo que se ve en la esquina: angulos, punta y el peldano mas cercano."""
    if api_deg is None:
        return "esperando lectura del brazo..."
    p = data.site("efector").xpos
    lineas = ["  ".join(f"J{k + 1} {v:7.1f}" for k, v in enumerate(api_deg)),
              f"punta  x {p[0] * 100:6.1f}  y {p[1] * 100:6.1f}  z {p[2] * 100:6.1f} cm"]
    if puntos:
        nombre, x = min(puntos, key=lambda nx: float(np.linalg.norm(nx[1] - p)))
        lineas.append(f"peldano mas cercano {nombre}: a {np.linalg.norm(x - p) * 100:.1f} cm")
    if fallas:
        lineas.append(f"lecturas fallidas seguidas: {fallas}")
    return "\n".join(lineas)


def recorrido_grabado(puntos_csv: Path | None) -> list[np.ndarray]:
    """Poses (4,) en grados API del CSV, para `--sin-brazo`."""
    if puntos_csv is None or not puntos_csv.exists():
        return [np.zeros(4), np.array([45.0, 30.0, 10.0, 0.0])]
    filas = np.genfromtxt(puntos_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return [np.array([r["j1"], r["j2"], r["j3"], 0.0]) for r in np.atleast_1d(filas)]


def ultimo_csv() -> Path | None:
    """El `stair_poses_promedio*.csv` mas reciente de `data/raw/`, o None."""
    raw = resolve_repo_path("data", "raw", must_exist=False)
    cands = sorted(raw.glob("stair_poses_promedio*.csv"), key=lambda f: f.stat().st_mtime)
    return cands[-1] if cands else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--puerto", default="COM7", help="puerto serie del brazo")
    ap.add_argument("--no-soltar", action="store_true",
                    help="no liberar los servos: solo se ve la pose, no se puede mover a mano")
    ap.add_argument("--sin-brazo", action="store_true",
                    help="no abre ningun puerto: recorre las poses del CSV")
    ap.add_argument("--csv", type=Path, default=None,
                    help="CSV de promedios para las marcas (default: el mas reciente)")
    args = ap.parse_args()

    model, data = load_model(resolve_repo_path(*XML_RELATIVE.parts))
    l0 = suma_tendon(model)
    csv = args.csv or ultimo_csv()
    puntos = puntos_peldanos(model, csv)
    print(f"marcas: {len(puntos)} peldanos de {csv}" if puntos else "sin marcas de peldanos")

    mc = None
    soltado = False
    if not args.sin_brazo:
        from pymycobot import MyPalletizer260

        mc = MyPalletizer260(args.puerto)
        time.sleep(0.5)   # que el puerto abra antes de la primera lectura
        q0 = leer(mc)
        if q0 is None:
            raise SystemExit(f"get_angles() no devolvio angulos por {args.puerto}: "
                             "brazo apagado, puerto equivocado u otro programa usandolo.")
        print("pose actual (grados API): "
              + ", ".join(f"J{k + 1}={v:.1f}" for k, v in enumerate(q0)))

    try:
        if mc is not None and not args.no_soltar:
            input("Los servos se van a SOLTAR y el brazo queda flojo. Sostenelo y apreta Enter... ")
            mc.release_all_servos()
            soltado = True

        recorrido = recorrido_grabado(csv) if mc is None else []
        periodo = 1.0 / RATE_HZ
        q: np.ndarray | None = None
        fallas = 0
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                          show_right_ui=False) as v:
            dibujar_marcas(v.user_scn, puntos)
            print("Ventana abierta. Cerrala (o Ctrl+C aca) para terminar.")
            t0 = time.perf_counter()
            i = 0
            while v.is_running():
                if mc is not None:
                    nuevo = leer(mc)
                    if nuevo is None:
                        fallas += 1
                    else:
                        q, fallas = nuevo, 0
                else:
                    # Sin brazo: interpola entre las poses grabadas, 2 s cada una.
                    s = (time.perf_counter() - t0) / 2.0
                    k = int(s)
                    a, b = recorrido[k % len(recorrido)], recorrido[(k + 1) % len(recorrido)]
                    f = min(1.0, 2.0 * (s - k))   # 1 s de viaje, 1 s quieto
                    q = a + f * (b - a)
                with v.lock():
                    if q is not None:
                        poner_pose(model, data, q, l0)
                    v.set_texts((mj.mjtFontScale.mjFONTSCALE_150, mj.mjtGridPos.mjGRID_TOPLEFT,
                                 texto(data, q, puntos, fallas), None))
                v.sync()
                # Plazos absolutos, no sleep(periodo) acumulado (ver core.clock.RateLoop).
                i += 1
                espera = t0 + i * periodo - time.perf_counter()
                if espera > 0:
                    time.sleep(espera)
                else:
                    i = int((time.perf_counter() - t0) / periodo)
    except KeyboardInterrupt:
        print("Interrumpido.")
    finally:
        if soltado:
            input("SOSTENE el brazo: Enter para reactivar los servos... ")
            for sid in (1, 2, 3, 4):
                mc.focus_servo(sid)
            print("servos activos.")


if __name__ == "__main__":
    main()
