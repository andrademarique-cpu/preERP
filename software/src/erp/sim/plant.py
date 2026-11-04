"""Carga del modelo, variante ciega, equilibrio de reposo y layout de sensores.

Las cuatro cosas que habia que hacerle al XML antes de poder filtrar, que
vivian en celdas de `mypalletizer260EKF.ipynb` y `viewer.ipynb` (ADR-0002, fase
P1). Nada de aca conoce sensores, puertos ni relojes: se le pasa una ruta y
devuelve objetos de MuJoCo.

Como `erp.sim.mujoco`, este modulo toca la API de MuJoCo directamente, asi que
vale la misma disciplina: todo lo que sale envuelto en `np.asarray(...,
dtype=np.float64)`, `int(...)` o un tipo concreto, porque `mujoco` no trae
`py.typed` y sin eso se escapa un `Any` al estimador y mypy strict lo acepta
sin chistar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np

from erp.core.types import Array, IntArray

# `Any` para los objetos de MuJoCo: la libreria no trae `py.typed`, asi que
# `mj.MjModel` y `mj.MjData` son `Any` para mypy de todas formas. Escribirlo
# explicito deja claro que el borde no esta tipado, en vez de aparentar que si.
MjModel = Any
MjData = Any

__all__ = [
    "blind_variant",
    "distancias",
    "geoms_por_clase",
    "load_model",
    "sensor_layout",
    "state_dim_of",
    "warmup_to_rest",
]

HOME_KEYFRAME = "home"

# Clases de colision de los XML: la herramienta lleva contype=2 (`ee_collider`,
# y `fsr_cyl`/`fsr_tip` en el modelo del FSR) y los solidos que la frenan
# conaffinity=2 (piso, props). El resto del brazo es 0/0 y atraviesa todo, en la
# planta tambien.
CLASE_HERRAMIENTA = 2
CLASE_OBSTACULO = 2
DIST_MAX_M = 0.05


def load_model(
    xml_path: str | Path, keyframe: str | None = HOME_KEYFRAME
) -> tuple[MjModel, MjData]:
    """Carga el XML y devuelve `(model, data)`, opcionalmente en un keyframe.

    `keyframe` por defecto es `"home"`, que es como arranca el brazo en todos
    los notebooks. Con `None` no se resetea y `data` queda en `qpos0`.

    OJO con el keyframe del palletizer: usa `act = 1.5693`, no 1.57, para que
    la igualdad de tendon se cumpla exacta. Con 1.57 el solver empuja el brazo
    en el primer paso y eso entra al log como un transitorio que no existe.
    """
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    if keyframe is not None:
        # mj_name2id devuelve -1 si no existe; model.key(...) levanta KeyError.
        # Se chequea explicito para dar un mensaje con los nombres disponibles.
        names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)]
        if keyframe not in names:
            raise KeyError(f"el modelo no tiene el keyframe {keyframe!r}; tiene {names}")
        mj.mj_resetDataKeyframe(model, data, model.key(keyframe).id)
    return model, data


FLOOR_GEOM = "floor"


def blind_variant(
    xml_path: str | Path,
    keyframe: str | None = HOME_KEYFRAME,
    floor_collides: bool = False,
) -> tuple[MjModel, MjData]:
    """El mismo XML con los actuadores pasados a `dyntype="integrator"`.

    Con los `<position>` del XML (`na = 0`) y `ctrl = 0`, el servo tira cada
    junta hacia cero: el filtro "sabria" que el brazo vuelve a home. Con
    `integrator` cada servo gana un estado de activacion -- el setpoint
    efectivo -- que con `ctrl = 0` se queda quieto, y Q lo vuelve random walk.
    Esa activacion es lo que el filtro estima en lugar del control que nunca
    recibe. `na` pasa de 0 a 3 y `nx` de 8 a 11.

    Se compila con MjSpec y NO editando el texto del XML: el archivo tiene un
    `<include>` y mallas con rutas relativas que `from_xml_string` no resuelve.
    Ganancias, forcerange y el bloque `<sensor>` quedan identicos, que es lo
    que permite comparar `h(x)` contra el log de la planta.

    El piso del modelo del FILTRO no choca (`floor_collides=False`, por
    defecto). La planta lo sigue teniendo solido -- `scene.xml` no se toca --
    asi que un toque del efector contra el piso pasa a ser dinamica NO
    modelada, igual que los props: es el banco para reproducir el sintoma de
    "el estimado atraviesa la superficie". Se apaga `contype`/`conaffinity` en
    vez de borrar el geom, porque borrarlo renumera los geoms y mueve lo que
    `erp.viz.ghost` y `test_plant.py` indexan. `floor_collides=True` devuelve
    el comportamiento anterior (piso modelado), para comparar.

    -> (model, data)
    """
    spec = mj.MjSpec.from_file(str(xml_path))
    for actuator in spec.actuators:
        actuator.dyntype = mj.mjtDyn.mjDYN_INTEGRATOR
    if not floor_collides:
        floor = spec.geom(FLOOR_GEOM)
        if floor is not None:
            floor.contype = 0
            floor.conaffinity = 0
    model = spec.compile()
    data = mj.MjData(model)
    if keyframe is not None:
        names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)]
        if keyframe in names:
            mj.mj_resetDataKeyframe(model, data, model.key(keyframe).id)
    return model, data


def geoms_por_clase(model: MjModel) -> tuple[IntArray, IntArray]:
    """(herramienta, obstaculos): ids de geoms segun su clase de colision.

    Por clase y no por nombre, la misma regla que `scripts/planear_escaleras.py`:
    un prop nuevo con conaffinity=2 entra solo. Solo cuenta la herramienta,
    porque es lo unico del brazo que choca EN LA PLANTA; los eslabones
    atraviesan los solidos tambien en la verdad, y medirles la penetracion no
    distinguiria el error del filtro del de la planta.
    """
    contype = np.asarray(model.geom_contype)
    conaff = np.asarray(model.geom_conaffinity)
    herramienta = np.flatnonzero(contype == CLASE_HERRAMIENTA).astype(np.intp)
    obstaculos = np.flatnonzero(conaff == CLASE_OBSTACULO).astype(np.intp)
    return herramienta, obstaculos


def distancias(
    model: MjModel,
    data: MjData,
    qpos: Array,
    herramienta: IntArray,
    obstaculos: IntArray,
    dist_max: float = DIST_MAX_M,
) -> tuple[Array, Array]:
    """Distancia CON SIGNO de la herramienta a cada obstaculo, en la pose `qpos`.

    Negativa = penetracion. Por obstaculo es el minimo sobre los geoms de la
    herramienta, y es lo mismo que el `contact.dist` mas hondo de
    `mj_collision` (test_plant.py lo usa de oraculo). Mas lejos que `dist_max`
    se reporta `dist_max`: `mj_geomDistance` no busca mas alla.

    Solo cinematica (`mj_kinematics`): no toca qvel, act ni fuerzas, asi que
    sirve igual para la pose verdadera y para la estimada. `data` tiene que ser
    un MjData PROPIO de esta medicion; escribe su `qpos`.

    `normal` es unitaria, sale del obstaculo hacia afuera (la direccion en que
    la herramienta tendria que moverse para separarse), marco mundo. Cero si
    los dos puntos coinciden y no hay direccion.

    -> dist (n_obs,) m;  normal (n_obs, 3) adimensional, marco mundo
    """
    nq = int(model.nq)
    data.qpos[:nq] = np.asarray(qpos, dtype=np.float64)[:nq]
    mj.mj_kinematics(model, data)
    dist = np.full(len(obstaculos), float(dist_max))
    normal = np.zeros((len(obstaculos), 3))
    fromto = np.zeros(6)
    for k, g_obs in enumerate(obstaculos):
        for g in herramienta:
            d = float(mj.mj_geomDistance(model, data, int(g), int(g_obs), float(dist_max), fromto))
            if d < dist[k]:
                dist[k] = d
                # fromto = (punto en la herramienta, punto en el obstaculo).
                # Separados, (desde - hasta) ya apunta hacia afuera; adentro se
                # invierte, y el signo de d lo corrige.
                v = fromto[:3] - fromto[3:]
                n = float(np.linalg.norm(v))
                normal[k] = np.sign(d) * v / n if n > 0.0 else 0.0
    return np.asarray(dist, dtype=np.float64), np.asarray(normal, dtype=np.float64)


def warmup_to_rest(model: MjModel, data: MjData, q_cmd: Array, warmup_steps: int = 100) -> Array:
    """`[qpos, qvel]` (rad, rad/s) del brazo QUIETO con la consigna `q_cmd` (3,) rad.

    Resetea al keyframe `home`, escribe la consigna y deja correr
    `warmup_steps` pasos para que el brazo se asiente bajo su propio peso.

    El estado inicial del filtro tiene que ser ESTE equilibrio y no ceros: con
    `q = 0` y activacion 0 el servo no hace fuerza, la gravedad acelera el brazo
    y `h(x0)` da `link2_acc_x` 4.1 m/s^2 contra 9.6 medido. El equilibrio hunde
    link1/link2 0.09/0.20 deg y da 9.81.

    -> (2*nv,) en el orden que espera `erp.sim.mujoco._load`.
    """
    mj.mj_resetDataKeyframe(model, data, model.key(HOME_KEYFRAME).id)
    data.ctrl[:3] = q_cmd
    for _ in range(warmup_steps):
        mj.mj_step(model, data)
    return np.asarray(np.r_[data.qpos, data.qvel], dtype=np.float64)


def sensor_layout(model: MjModel) -> dict[str, slice]:
    """`{nombre: slice}` dentro de `data.sensordata`, leido del modelo.

    Se lee del modelo en vez de escribir los indices a mano porque el orden y
    el ancho de `sensordata` los decide el XML: agregar un sensor arriba de la
    lista corre a todos los de abajo, y unos indices fijos seguirian dando
    numeros -- solo que del sensor equivocado. Esa es exactamente la falla que
    tenia el post-proceso del notebook con `sensor_log_sim[:, 0:3]`.

    Para indexar `z`, `R` y las filas de `H` a la vez conviene
    `erp.sensors.mujoco.rows_of`, que devuelve los indices concatenados.
    """
    layout: dict[str, slice] = {}
    for i in range(model.nsensor):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, i)
        adr = int(model.sensor_adr[i])
        layout[str(name)] = slice(adr, adr + int(model.sensor_dim[i]))
    return layout


def state_dim_of(model: MjModel) -> int:
    """`nx` del modelo: `2 * nv + na`. Alias de `erp.sim.mujoco.state_dim`.

    Repetido aca para que quien arma un modelo con `blind_variant` tenga el
    numero a mano sin importar dos modulos; posiciones en el espacio TANGENTE
    (`nv`, no `nq`), que es lo que hace valida la jacobiana por diferencias.
    """
    return int(2 * model.nv + model.na)
