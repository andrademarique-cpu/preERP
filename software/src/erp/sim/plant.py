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

from erp.core.types import Array

# `Any` para los objetos de MuJoCo: la libreria no trae `py.typed`, asi que
# `mj.MjModel` y `mj.MjData` son `Any` para mypy de todas formas. Escribirlo
# explicito deja claro que el borde no esta tipado, en vez de aparentar que si.
MjModel = Any
MjData = Any

__all__ = ["blind_variant", "load_model", "sensor_layout", "state_dim_of", "warmup_to_rest"]

HOME_KEYFRAME = "home"


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


def blind_variant(
    xml_path: str | Path, keyframe: str | None = HOME_KEYFRAME
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

    -> (model, data)
    """
    spec = mj.MjSpec.from_file(str(xml_path))
    for actuator in spec.actuators:
        actuator.dyntype = mj.mjtDyn.mjDYN_INTEGRATOR
    model = spec.compile()
    data = mj.MjData(model)
    if keyframe is not None:
        names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)]
        if keyframe in names:
            mj.mj_resetDataKeyframe(model, data, model.key(keyframe).id)
    return model, data


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
