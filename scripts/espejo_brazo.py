"""Espejo del MyPalletizer260 real en MuJoCo: lo moves a mano y el modelo lo copia.

    python scripts/espejo_brazo.py                    # en vivo por COM7, suelta los servos
    python scripts/espejo_brazo.py --puerto COM6
    python scripts/espejo_brazo.py --no-soltar        # solo lee; el brazo queda firme
    python scripts/espejo_brazo.py --sin-brazo        # sin hardware: recorre las poses grabadas
    python scripts/espejo_brazo.py --sin-escaleras    # el brazo solo, sin los bloques
    python scripts/espejo_brazo.py --punta sin        # sin el adaptador del FSR (la punta vieja)

LA PUNTA. Por defecto el modelo lleva el adaptador del FSR (cilindro + cupula,
`MyPalletizer260_fsr.xml`), y el sitio `efector` esta en la punta de la cupula,
36.5 mm mas abajo que sin el. Las poses de `stair_poses_promedio.csv` del
2026-10-21 se midieron SIN adaptador: con `--punta fsr` sus esferas verdes
quedan 3.65 cm adentro de los bloques, que es lo esperado y no un error. Al
re-medir con la punta puesta, el CSV nuevo reemplaza al viejo (se usa el mas
reciente) y las esferas vuelven a quedar sobre los peldanos.

QUE HACE. Lee `get_angles()` del brazo a ~15 Hz, pasa los grados API a radianes
del modelo con el `JMAP` de `demo_teleop_real.py` (no una copia mas: ya hay tres)
y pone esa pose en el modelo SIN simular fisica -- `mj_forward`, nunca `mj_step`.
Es cinematica pura: el modelo esta donde el brazo dice que esta, y nada mas.

Ademas dibuja una esfera verde en cada punto donde el MODELO pone la punta para
las poses de `data/raw/stair_poses_promedio*.csv` (las de las escaleras, medidas
a mano). Si al apoyar la herramienta en un peldano real la punta del modelo cae
sobre su esfera, el modelo y la lectura coinciden en ese punto.

LAS ESCALERAS SON SOLIDAS. Por defecto carga `MyPalletizer260_escaleras.xml`:
el brazo mas los 6 bloques medidos del banco (`prop_escaleras_reales.xml`, base
3 x 3 cm, alturas 3 / 6 / 9 cm). Como aca no hay fisica, el espejo no puede
IMPEDIR que la punta entre en un bloque -- el brazo real lo mueve una mano, no
el modelo --, pero lo DETECTA con los contactos que `mj_forward` calcula entre
la herramienta y cada bloque, y pinta el bloque segun `clasificar`:

  AMARILLO  la cupula PRESIONA el peldano desde arriba: esta sobre su huella
            (mas `HOLGURA_XY_M`) y aplasta a lo sumo `MAX_APRIETE_MM`. Es lo
            que se busca: la cupula es silicona sobre un FSR y hay que apretar.
  ROJO      CHOQUE: el cilindro (no la cupula) entra mas de `TOL_CHOQUE_MM`, o
            la cupula entra de costado, o aplasta mas de `MAX_APRIETE_MM`.
  (sin cambio) un roce de menos de `TOL_CHOQUE_MM`.

Las cifras salen de las 30 tomas del 2026-10-28, con la punta ya corregida a
35.1 mm: apretar aplasta la cupula del modelo 0 a 6.5 mm (mediana ~3).
Antes habia un solo umbral de 3 mm para todo, y con eso un peldano apretado
flojo (B2, promedio 1.6 mm) no se pintaba mientras A1 (4.8 mm) si: el color
decia "choque" para lo que era el objetivo.

Con fisica (`demo_teleop.py --prop escaleras`) los bloques si frenan: medido,
una consigna 15 grados por debajo del peldano A2 deja la punta en 58.6 mm contra
un techo de 60.0 (1.4 mm de contacto blando); sin bloques baja a 43.3 mm.

LO UNICO QUE VIAJA POR EL PUERTO es `get_angles()`, `release_all_servos()` y
`focus_servo()`. Ningun comando de movimiento, igual que la celda de la
envolvente y la de las escaleras en `notebooks/Palletizer.ipynb`.

POR QUE SE ESCRIBE `qpos` Y NO ES LA TRAMPA DE `demo_teleop`. Alla escribir
`qpos` rompe la igualdad del tendon y la simulacion siguiente da 345 m/s^2. Aca
no se simula, y `act` se calcula para cumplir la igualdad: q_link1 + q_link2 +
q_act = L0, con L0 leido del keyframe `home` del propio modelo (1.5693). Medido:
residuo de la igualdad 3.7e-06 en cualquier pose, el mismo que en `home`.

LO QUE NO MUESTRA. J4 (el giro de la punta) no esta en el modelo y se ignora.
Como no se simula, los rangos de las juntas del XML no limitan nada: el modelo
dibuja la pose que el brazo reporte. (Cuando se escribio esto, `link2` iba de 0
a 60 grados y las escaleras ya pedian J3 = -21; `a2ac051` lo bajo a -92.) Y el
espejo no dice nada del EKF: no hay IMUs aca.

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
XML_ESCALERAS = XML_RELATIVE.with_name("MyPalletizer260_escaleras.xml")
# Con el adaptador del FSR (punta de cupula, MyPalletizer260_fsr.xml) el sitio
# `efector` baja 36.5 mm a la punta de la cupula; los bloques no se mueven.
XML_FSR = XML_RELATIVE.with_name("MyPalletizer260_fsr.xml")
XML_FSR_ESCALERAS = XML_RELATIVE.with_name("MyPalletizer260_fsr_escaleras.xml")
RATE_HZ = 15.0          # lectura del brazo; la misma tasa que la envolvente
RADIO_MARCA_M = 0.006   # esfera de cada peldano, radio en metros
VERDE = np.array([0.1, 0.8, 0.2, 0.6], dtype=np.float32)
ROJO = np.array([0.95, 0.1, 0.1, 1.0], dtype=np.float32)
AMARILLO = np.array([1.0, 0.85, 0.1, 1.0], dtype=np.float32)
TOL_CHOQUE_MM = 3.0     # roce tolerado al resto de la herramienta (ver docstring)
MAX_APRIETE_MM = 8.0    # cuanto puede aplastarse la cupula al presionar: medido hasta 6.5
HOLGURA_XY_M = 0.012    # los peldanos medidos caen a 3-10 mm del centro del bloque


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
    poner_q(model, data, JMAP.to_model_rad(api_deg[None, :3])[0], l0)


def poner_q(model: Any, data: Any, q_rad: np.ndarray, l0: float) -> None:
    """Igual que `poner_pose`, pero desde (3,) radianes del modelo [rot, link1, link2]."""
    rot, link1, link2 = (float(v) for v in q_rad[:3])
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


def choques(model: Any, data: Any) -> dict[str, float]:
    """{nombre del geom: solape en mm} de cada geom que toca la HERRAMIENTA.

    La herramienta es todo geom de la clase de colision 2 (`contype=2`): el
    `ee_collider` y, con la punta del FSR, tambien `fsr_cyl` y `fsr_tip`. Se
    elige por clase y no por nombre para que siga valiendo si la punta cambia.

    Sale de los contactos que `mj_forward` ya calculo. `dist` de MuJoCo es
    negativa cuando hay solape, asi que el solape es `-dist`, en mm. Si un geom
    tiene varios puntos de contacto se queda el mas hundido. Incluye el piso.
    """
    herramienta = {g for g in range(model.ngeom) if int(model.geom_contype[g]) == 2}
    out: dict[str, float] = {}
    for c in data.contact[: data.ncon]:
        if c.geom1 in herramienta:
            otro = c.geom2
        elif c.geom2 in herramienta:
            otro = c.geom1
        else:
            continue
        nombre = model.geom(int(otro)).name or f"geom {int(otro)}"
        out[nombre] = max(out.get(nombre, 0.0), -float(c.dist) * 1000.0)
    return out


def clasificar(model: Any, data: Any) -> dict[str, tuple[str, float]]:
    """{obstaculo: (estado, mm)} con estado "presiona", "choque" o "roza".

    Regla del docstring del modulo. La cupula es `fsr_tip` si el modelo trae el
    adaptador, y si no el `ee_collider` (la punta vieja hace las veces). "mm" es
    el mayor solape de cualquier geom de la herramienta contra ese obstaculo.
    Un obstaculo con choque queda en choque aunque la cupula tambien lo presione.
    """
    herramienta = {g for g in range(model.ngeom) if int(model.geom_contype[g]) == 2}
    nombres = {model.geom(g).name: g for g in herramienta}
    cupula = nombres.get("fsr_tip", nombres.get("ee_collider"))
    p = data.site("efector").xpos
    rango = {"roza": 0, "presiona": 1, "choque": 2}
    out: dict[str, tuple[str, float]] = {}
    for c in data.contact[: data.ncon]:
        if c.geom1 in herramienta:
            g, otro = int(c.geom1), int(c.geom2)
        elif c.geom2 in herramienta:
            g, otro = int(c.geom2), int(c.geom1)
        else:
            continue
        mm = -float(c.dist) * 1000.0
        nombre = model.geom(otro).name or f"geom {otro}"
        estado = "choque" if mm > TOL_CHOQUE_MM else "roza"
        if g == cupula and nombre.startswith("esc_"):
            centro = model.geom_pos[otro][:2]
            techo = model.geom_pos[otro][2] + model.geom_size[otro][2]
            encima = bool(np.max(np.abs(p[:2] - centro)) <= model.geom_size[otro][0] + HOLGURA_XY_M)
            aprieta = p[2] >= techo - MAX_APRIETE_MM / 1000.0
            estado = "presiona" if encima and aprieta else "choque"
        viejo, mm_viejo = out.get(nombre, ("roza", 0.0))
        out[nombre] = (max(estado, viejo, key=rango.__getitem__), max(mm, mm_viejo))
    return out


def pintar(model: Any, colores: dict[int, np.ndarray],
           estados: dict[str, tuple[str, float]]) -> None:
    """Amarillo si la cupula presiona el bloque, rojo si choca; si no, su color."""
    for gid, rgba in colores.items():
        estado = estados.get(model.geom(gid).name, ("roza", 0.0))[0]
        model.geom_rgba[gid] = {"presiona": AMARILLO, "choque": ROJO}.get(estado, rgba)


def texto(data: Any, api_deg: np.ndarray | None, puntos: list[tuple[str, np.ndarray]],
          fallas: int, estados: dict[str, tuple[str, float]] | None = None) -> str:
    """Lo que se ve en la esquina: angulos, punta, peldano mas cercano y choques."""
    if api_deg is None:
        return "esperando lectura del brazo..."
    p = data.site("efector").xpos
    lineas = ["  ".join(f"J{k + 1} {v:7.1f}" for k, v in enumerate(api_deg)),
              f"punta  x {p[0] * 100:6.1f}  y {p[1] * 100:6.1f}  z {p[2] * 100:6.1f} cm"]
    if puntos:
        nombre, x = min(puntos, key=lambda nx: float(np.linalg.norm(nx[1] - p)))
        lineas.append(f"peldano mas cercano {nombre}: a {np.linalg.norm(x - p) * 100:.1f} cm")
    texto_estado = {"presiona": "presionando", "choque": "CHOQUE con", "roza": "roza"}
    for nombre, (estado, mm) in sorted((estados or {}).items()):
        lineas.append(f"{texto_estado[estado]} {nombre.removeprefix('esc_')}: {mm:.1f} mm")
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
    ap.add_argument("--sin-escaleras", action="store_true",
                    help="cargar el brazo solo, sin los bloques de las escaleras")
    ap.add_argument("--punta", choices=("fsr", "sin"), default="fsr",
                    help="fsr: con el adaptador de cupula (default). sin: el brazo sin punta,"
                         " como se midio el 2026-10-21")
    args = ap.parse_args()

    xml = {("fsr", False): XML_FSR_ESCALERAS, ("fsr", True): XML_FSR,
           ("sin", False): XML_ESCALERAS, ("sin", True): XML_RELATIVE}[
        (args.punta, args.sin_escaleras)]
    model, data = load_model(resolve_repo_path(*xml.parts))
    l0 = suma_tendon(model)
    colores = {g: model.geom_rgba[g].copy() for g in range(model.ngeom)
               if model.geom(g).name.startswith("esc_")}
    print(f"modelo: {xml.name} ({len(colores)} bloques de escalera)")
    csv = args.csv or ultimo_csv()
    puntos = puntos_peldanos(model, csv)
    print(f"marcas: {len(puntos)} peldanos de {csv}" if puntos else "sin marcas de peldanos")

    mc = None
    soltado = False
    if not args.sin_brazo:
        from pymycobot import MyPalletizer260

        try:
            mc = MyPalletizer260(args.puerto)
        except Exception as err:   # serial.SerialException; pyserial no se importa aca
            if "denied" in str(err).lower() or "denegado" in str(err).lower():
                raise SystemExit(
                    f"{args.puerto} esta OCUPADO por otro programa (acceso denegado). Casi"
                    " siempre es un notebook que abrio el brazo y sigue vivo: reiniciar su"
                    " kernel (Restart) o cerrar el otro programa, y volver a correr esto."
                ) from err
            raise
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
                    estados = clasificar(model, data)
                    pintar(model, colores, estados)
                    v.set_texts((mj.mjtFontScale.mjFONTSCALE_150, mj.mjtGridPos.mjGRID_TOPLEFT,
                                 texto(data, q, puntos, fallas, estados), None))
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
