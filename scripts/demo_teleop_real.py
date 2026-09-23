"""Teleop del MyPalletizer260 REAL: manejas el brazo y ves al EKF ciego seguirlo.

    python scripts/demo_teleop_real.py                              # ensayo: sin hardware
    python scripts/demo_teleop_real.py --imu real                   # solo las IMUs (COM7)
    python scripts/demo_teleop_real.py --arm real --arm-port COM6 --imu real
    python scripts/demo_teleop_real.py ... --plant-delay            # planta atrasada 0.395 s

ES EL HERMANO DE `demo_teleop.py`, con la misma estructura -- un objeto de
sesion, un tick de Qt a 60 Hz, el visor pasivo de MuJoCo con el fantasma en
`user_scn`, `LivePlots` y el panel de control -- y otras fuentes para cada
senal:

  planta     el modelo de planta SIN editar, integrado en tiempo real bajo la
             CONSIGNA que se le manda al brazo. Es un MODELO DE LA CONSIGNA, no
             la verdad: en hardware no hay verdad. Con `--plant-delay` la
             consigna le llega atrasada `arm_lag_s` (0.395 s, medido).
  medido     las IMUs reales (Teensy, `SerialIMUSensor`), decodificadas al marco
             del sitio y con el sesgo de reposo restado: exactamente lo que
             recibe el filtro. TODAS las muestras, no solo las aplicadas.
  estimado   el EKF ciego del notebook (celdas 18-19), en vivo. El fantasma.

SIN `--plant-delay` EL FANTASMA VA ~0.4 s ATRAS DE LA PLANTA durante el
movimiento, aunque el filtro ande perfecto: es el atraso de transporte del
brazo (cola de consignas del firmware, sin `set_fresh_mode`). No es un error del
filtro. Con `--plant-delay` los dos deberian superponerse cuando el filtro sigue.

EL FILTRO ES EL DEL NOTEBOOK, no el de `demo_teleop.py`, que difiere en dos
cosas: aca `P0` es el de la celda 18 (1/5/0.5 grados) y el reposo se calienta en
el modelo de PLANTA. `Q`, `R`, las sigmas y el cableado son los mismos que en
`make_golden_run.py`; el cableado sale de `config/estimation.yaml`. El sesgo se
estima como en la celda 19 (`rest_bias` con `expected_rest = h(x_rest)`), pero
en vivo, de ~1 s quieto despues del homing, y se resta a cada `z` -- la misma
aritmetica que `z - imu_bias` de la celda 19. `apply_calibration` NO se usa:
ademas del sesgo instalaria la `R` de reposo, 3-8x mas angosta que la nominal,
que P6 midio y rechazo.

ES LA PRIMERA VEZ QUE ESTE FILTRO CORRE EN VIVO SOBRE HARDWARE. El notebook lo
corrio fuera de linea (celda 19) o en vivo contra `DryRunArm` + `SimSensor`
(celda 21). Sobre datos reales dio NIS mediana 29 contra el objetivo 12; en
vivo, con sellos de LLEGADA (el firmware no manda tiempo), esperalo peor.

SEGURIDAD -- lo que hace el script y lo que NO:

  - Por defecto NO toca hardware: `--arm dry` (`DryRunArm` con el atraso medido)
    y `--imu sim` (`LiveSimSensor` sobre la planta). El brazo real pide
    `--arm real --arm-port COMx` Y escribir `si` en la consola.
  - Antes de moverse: homing a [0, 0, 0] a velocidad 30, como el notebook, y
    la LLEGADA se verifica leyendo los angulos (2 grados por junta, 3 lecturas
    seguidas; si no, aborta). `sync_send_angles` solo no alcanza: en pymycobot
    4.0.7 puede volver antes de que el brazo arranque y devuelve 1 siempre.
    El estado inicial del filtro supone esa pose. Segunda red: si el brazo
    todavia se mueve en la ventana de reposo, el giroscopo invalida el sesgo
    y tambien aborta.
  - Las teclas mueven un OBJETIVO; la consigna va hacia el a lo sumo `--slew`
    (0.5 rad/s por defecto, ~29 grados/s, contra 120 grados/s de spec) y las
    velocidades de jog tambien se recortan a ese tope. Cada consigna se recorta
    a `ctrlrange` del XML, que se verifica al arrancar que cae DENTRO de los
    limites de la API: pymycobot no revisa NADA, esta es la unica guarda.
  - A lo sumo 25 Hz de consignas (el techo medido; lo que sobra se encola en el
    firmware y ese es el atraso), solo si la consigna cambio, sin rafagas de
    recuperacion. Antes de cada envio una ultima guarda de rango y de paso.
  - Con el brazo real, en el VISOR las teclas solo dan pasos fijos: nada queda
    trabado. Movimiento continuo SOLO desde el panel, mientras se mantiene la
    tecla o el boton; al soltar o al perder el foco, frena.
  - Frenar (X), cerrar el visor, ESC, Ctrl-C o una IMU muerta dejan de mandar
    consignas. Los servos NO se liberan al salir: `release_all_servos` dejaria
    caer el brazo con la gravedad. El brazo se queda en la ultima consigna.
  - El freno tarda lo que la cola del firmware: ~0.4 s. **El freno de
    emergencia es el interruptor de alimentacion, no este script.** Que hace
    `mc.stop()` en el firmware del 260 no esta verificado, y nada depende de el.
  - El EKF NUNCA alimenta una consigna: el filtro solo se dibuja. Un estimado
    que diverge no puede mover el brazo. Y sigue ciego a `u` por construccion.

LA PRIMERA VEZ, en este orden y con la mano en el interruptor:
  1. `--imu real` solo (brazo desconectado o en ensayo): ~20 Hz, sesgo valido,
     `descartadas` en 0, NIS finita.
  2. `--arm real --imu real`: una junta, finura 5 en pasos, despues mantener
     en el panel a finura baja.

CADA SESION SE GUARDA en `data/sessions/<fecha>/` (nunca en `data/raw/` ni
`data/processed/`, donde vive el golden): el CSV CRUDO de la IMU -- con el
bloque de reposo en tiempos negativos --, `commands.npz` con cada consigna
mandada y `summary.txt`. Crudo, para poder re-decodificarlo si cambia la config.
`data/sessions/` esta en `.gitignore`: una sesion que valga se promueve a mano.

EL LAZO NO TIENE TESTS: necesita pantalla y hardware. Lo que si se verifico sin
hardware esta anotado en `docs/demo_teleop.md`. Los scripts no pasan por CI:
correr `ruff` a mano.

Castellano, como sus hermanos en `scripts/`. Reusa de `demo_teleop.py` el mapa
de teclas, los niveles de finura, el panel, la restauracion de banderas del
visor y el chequeo del fantasma -- importados, no copiados.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import demo_teleop as base
import mujoco as mj
import numpy as np

from erp.analysis import consistency_report
from erp.calibration import rest_bias
from erp.core.types import Measurement
from erp.estimators import EKF
from erp.fusion import FilterRunner
from erp.io.config import build_decoder, load_config
from erp.io.log import MeasurementLog
from erp.io.paths import repo_root, resolve_repo_path
from erp.robot import ArmInterface, DryRunArm, JointMap, MyPalletizerArm
from erp.sensors import LiveSimSensor, SensorError, SerialIMUSensor
from erp.sensors.clock import ArrivalClock
from erp.sensors.mujoco import make_R, rows_of
from erp.sim.dynamics import MujocoDynamics
from erp.sim.mujoco import h_dyn, make_Q
from erp.sim.plant import blind_variant, load_model, warmup_to_rest
from erp.trajectory import slew_limit

# -- el filtro: notebook celda 18, identico a make_golden_run.py -------------
SIG_ALPHA = 12.0        # rad/s^2 (1 sigma) por paso: DWNA a dt = 2 ms
SIG_ACT_BLIND = 5e-3    # rad por paso: random walk de la activacion
SIG_CM = 1e-4           # rad por paso a lo largo del modo comun
P0_DIAG = np.array(
    [np.deg2rad(1.0) ** 2] * 4 + [np.deg2rad(5.0) ** 2] * 4 + [np.deg2rad(0.5) ** 2] * 3
)
REST_GYRO_MAX = 0.05    # rad/s: la norma de giro por encima de la cual "no estaba quieto"

# -- el brazo: notebook celdas 7, 12, 14 y 15 ---------------------------------
# COPIA del `JMAP` del notebook (celda 7) y de `make_jmap()` en test_robot.py.
# Es la tercera; el cableado de las IMUs llego a tener cuatro y derivaron. Si
# se toca uno, tocar los tres -- o mejor, llevarlo a `config/estimation.yaml`.
JMAP = JointMap(
    names=("rot", "link1", "link2"),
    api_ids=(1, 2, 3),
    signs=np.array([1.0, 1.0, 1.0]),
    offsets_deg=np.array([0.0, 0.0, 0.0]),    # ceros verificados fisicamente
    api_limits_deg={1: (-162.0, 162.0), 2: (-2.0, 90.0), 3: (-92.0, 60.0),
                    4: (-180.0, 180.0)},
    vmax_deg_s=120.0,
    j4_hold_deg=0.0,
)
STREAM_RATE_HZ = 25.0   # techo medido de send (+ get_angles) en un puerto
STREAM_SPEED = 100      # escala de firmware 1..100, NO grados/s; la del notebook
HOMING_SPEED = 30
# Llegada a home, verificada leyendo angulos: 2 grados por junta, 3 lecturas
# seguidas cada 0.2 s. 20 s alcanzan para cruzar el rango entero a velocidad 30
# (el notebook no lo midio; la cota es generosa a proposito).
HOME_TOL_DEG = 2.0
HOME_STABLE_READS = 3
HOME_POLL_S = 0.2
HOME_TIMEOUT_S = 20.0
IMU_WAIT_S = 15.0
SETTLE_S = 0.5          # despues del homing, antes de la ventana de reposo
REST_WINDOW_S = 1.0     # ~20 muestras a 20 Hz; el offline usa 0.4 s porque no tiene mas
MIN_REST_SAMPLES = 10

# Tope de la consigna. 0.5 rad/s elegido para las primeras sesiones; el sim usa
# 1.0. Mas de 1.0 no se acepta con el brazo real.
SLEW_RAD_S = 0.5
SLEW_MAX_REAL = 1.0
MAX_PLANT_STEPS = 20    # por tick: una traba de la UI no se convierte en espiral
SIM_IMU_LATENCY_S = 0.005
# Rango fijo del giroscopo. Medido con el ensayo al tope de 0.5 rad/s: ver
# `docs/demo_teleop.md`. El acelerometro usa el +-20 del demo simulado.
GYRO_RANGE = (-1.5, 1.5)
LINKS = ("link1", "link2")
KINDS = ("acc", "gyro")


class _FnClock:
    """Un `Clock` sobre una funcion de tiempo. `FilterRunner` solo lee `now`."""

    def __init__(self, fn: Callable[[], float]) -> None:
        self._fn = fn

    def now(self) -> float:
        return float(self._fn())

    def sleep_until(self, t: float) -> None:
        """No se usa: el lazo lo marca el QTimer, no el runner."""


def check_ctrlrange_inside_api(jmap: JointMap, ctrl_range: Any) -> None:
    """Falla si algun extremo de `ctrlrange` (rad) cae fuera de la API (grados).

    Recortar a `ctrlrange` es la unica guarda de rango, asi que tiene que ser
    la mas estricta de las dos. Hoy lo es con margen: rot +-159.9 dentro de
    +-162, link1 0..90 dentro de -2..90, link2 0..59.6 dentro de -92..60.
    """
    for j, api_id in enumerate(jmap.api_ids):
        lo, hi = jmap.api_limits_deg[api_id]
        for v in ctrl_range[j]:
            deg = float(np.rad2deg(v) * jmap.signs[j] + jmap.offsets_deg[j])
            if not lo - 1e-9 <= deg <= hi + 1e-9:
                raise SystemExit(
                    f"ctrlrange de {jmap.names[j]} llega a {deg:.2f} grados, fuera de"
                    f" la API [{lo}, {hi}]. Nada se manda."
                )


class TeleopReal:
    """Estado de una sesion sobre el brazo real. La misma forma que `Teleop`.

    Expone lo que `demo_teleop.build_panel` lee -- `target`, `ctrl`,
    `ctrl_range`, `held`, `level`, `step_mode`, `q_true`, `q_est`, `nudge`,
    `set_target`, `stop`, `home`, `on_key` -- asi el panel anda sin cambios.

    UNA base de tiempo, `clock()` en segundos absolutos de `perf_counter`,
    porque eso es lo que sella `SerialIMUSensor`. El demo simulado usa el
    tiempo de la simulacion; aca eso pondria cada muestra real en el pasado o
    en el futuro por la diferencia entre las dos bases.

    `clock` y `sleep` se inyectan para poder ensayar sin pantalla ni hardware.
    """

    def __init__(
        self,
        *,
        arm: ArmInterface,
        real_arm: bool,
        imu: str = "sim",
        imu_port: str | None = None,
        imu_transport: Any = None,
        plant_delay_s: float = 0.0,
        slew_rad_s: float = SLEW_RAD_S,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if real_arm and slew_rad_s > SLEW_MAX_REAL:
            raise SystemExit(f"--slew {slew_rad_s} > {SLEW_MAX_REAL} rad/s con el brazo real")
        self.arm = arm
        self.real_arm = bool(real_arm)
        self.clock = clock
        self._sleep = sleep
        self.slew_rad_s = float(slew_rad_s)
        self.plant_delay_s = float(plant_delay_s)

        cfg = load_config()
        self.cfg = cfg
        self.decoder = build_decoder(cfg.imu)     # sesgo en CERO: se resta aparte
        xml = resolve_repo_path(*base.XML_RELATIVE.parts)
        self.model, self.data = load_model(xml)
        self.model_b, self.data_b = blind_variant(xml)
        self.dt = float(self.model.opt.timestep)

        self.imu_rows = rows_of(self.model_b, *cfg.imu.layout)
        self.imu_rows_plant = rows_of(self.model, *cfg.imu.layout)
        R_full = make_R(self.model_b, cfg.imu.sig_acc, cfg.imu.sig_gyro)
        self.R = R_full[np.ix_(self.imu_rows, self.imu_rows)]
        # Columnas a graficar dentro del vector de 12, en el orden que espera
        # `LivePlots` (link, despues tipo, despues eje), buscadas por NOMBRE.
        self.plot_cols = np.array([
            int(np.flatnonzero(self.imu_rows == r)[0])
            for link in LINKS for kind in KINDS
            for r in rows_of(self.model_b, f"{link}_{kind}")
        ])
        self.gyro_idx = self.decoder.indices_of(["link1_gyro", "link2_gyro"])
        self.acc_idx = self.decoder.indices_of(["link1_acc", "link2_acc"])

        # Reposo en [0, 0, 0], calentado en el modelo de PLANTA (celda 18). Deja
        # tambien a `self.data` en ese equilibrio: la planta arranca ahi.
        self.q0 = np.zeros(3)
        self.x_rest = np.r_[warmup_to_rest(self.model, self.data, self.q0), self.q0]
        u0 = np.zeros(self.model_b.nu)
        self.h_rest = h_dyn(self.x_rest, u0, self.model_b, self.data_b)[self.imu_rows]

        Q = make_Q(self.model_b, SIG_ALPHA, SIG_ACT_BLIND, SIG_CM)
        self.ekf = EKF(self.x_rest.copy(), np.diag(P0_DIAG), Q, R_full,
                       MujocoDynamics(self.model_b, self.data_b))
        H = float(cfg.time.buffer_horizon_s)
        # Entero de pasos de 2 ms: con un horizonte en un empate de redondeo la
        # mitad de las muestras se descarta (demo_teleop, 41 de 58). El epsilon
        # evita que 0.050/0.002 = 25.000...04 se vuelva 26 pasos.
        self.horizon = float(np.ceil(H / self.dt - 1e-9) * self.dt)
        self.runner: FilterRunner | None = None

        self.ctrl_range = np.asarray(self.model.actuator_ctrlrange[:3], dtype=np.float64)
        check_ctrlrange_inside_api(JMAP, self.ctrl_range)
        self.ctrl = self.q0.copy()
        self.target = self.q0.copy()
        # Con el brazo real se arranca en finura 3 (0.1 rad/s, pasos de 0.5 grados).
        self.level = 2 if self.real_arm else 0
        self.step_mode = False
        self.held: dict[int, float] = {}
        self.show_ghost = True

        self.imu_kind = imu
        if imu == "sim":
            self.sensor: Any = LiveSimSensor(
                rows=self.imu_rows, R=self.R, rate_hz=cfg.imu.rate_hz,
                latency_s=SIM_IMU_LATENCY_S, seed=0, clock=self.clock, keep_truth=False,
            )
        elif imu == "real":
            # Como la celda 9: sellos de LLEGADA, el firmware no manda tiempo.
            self.sensor = SerialIMUSensor(
                imu_port, self.decoder, self.imu_rows, self.R, fmt=cfg.imu.fmt,
                clock=ArrivalClock(0.0), transport=imu_transport,
                on_text=lambda s: print(f"[IMU] {s}"),
            )
        else:
            raise ValueError(f"imu debe ser 'sim' o 'real', no {imu!r}")

        self.bias = np.zeros(self.imu_rows.size)
        self.cal: Any = None
        self.log = MeasurementLog()
        self.t0 = float("nan")
        self.t_plant = float(self.clock())
        self._ctrl_hist: deque[tuple[float, Any]] = deque([(-np.inf, self.q0.copy())])
        self._t_jog = float("nan")
        self._next_send = float("inf")
        self._last_sent = self.q0.copy()
        self._t_last_sent = float("nan")
        self.sent: list[tuple[float, Any]] = []
        self.faulted: str | None = None
        self.nis: list[float] = []
        self.applied = 0
        self.updates: list[tuple[float, Any, float]] = []
        self.meas_new: list[tuple[float, Any]] = []

    # -- arranque -----------------------------------------------------------

    def prepare(self) -> None:
        """IMU -> homing -> reposo -> sesgo -> t0. Bloquea; corre antes del lazo.

        Aborta (SystemExit) si la IMU no manda nada o si la ventana de reposo no
        es valida: una calibracion invalida viene con sesgo CERO y parece exito.
        """
        if self.imu_kind == "real":
            self.sensor.start()
            if not self.sensor.wait_first(IMU_WAIT_S):
                raise SystemExit(f"la IMU no mando nada en {IMU_WAIT_S:.0f} s")
        self._home_arm()
        self._collect(SETTLE_S)           # lo del homing se descarta

        t, Z = self._collect(REST_WINDOW_S)
        if len(t) < MIN_REST_SAMPLES:
            raise SystemExit(f"reposo: {len(t)} muestras en {REST_WINDOW_S} s, hacen falta"
                             f" {MIN_REST_SAMPLES}. La IMU no esta mandando a ~20 Hz.")
        cal = rest_bias(
            t - t[0], Z, expected_rest=self.h_rest, gyro_idx=self.gyro_idx,
            acc_idx=self.acc_idx, rest_s=np.inf, max_gyro_norm=REST_GYRO_MAX,
            min_samples=MIN_REST_SAMPLES,
        )
        if not cal.valid:
            raise SystemExit(f"calibracion de reposo INVALIDA ({cal.note}). No se sigue:"
                             " el sesgo seria cero. El brazo se movio o la IMU esta mal.")
        self.cal = cal
        self.bias = np.asarray(cal.bias, dtype=np.float64)

        self.sensor.drain()
        self.t0 = float(self.clock())
        self.runner = FilterRunner(self.ekf, t0=self.t0, buffer_horizon=self.horizon,
                                   clock=_FnClock(self.clock), record=False)
        self._t_jog = self.t0
        self._next_send = self.t0
        self._t_last_sent = self.t0

    def _home_arm(self) -> None:
        """Homing a [0, 0, 0] a velocidad 30, y NO se sigue hasta VERLO llegar.

        El primer movimiento puede ser el mas grande de la sesion: el brazo
        arranca donde este. Se manda como el notebook -- `sync_send_angles` si
        el transporte lo tiene, un `send` comun si no (`DryRunArm`, pruebas) --
        pero **esa llamada no alcanza como confirmacion**. En pymycobot 4.0.7,
        `MyPalletizer260.sync_send_angles` manda y sale del bucle en cuanto
        `is_moving()` da 0, y devuelve 1 SIEMPRE, incluso por timeout. Con
        ~0.4 s de cola en el firmware, el primer `is_moving()` puede llegar
        antes de que el brazo arranque y volver al instante. Por eso la llegada
        se verifica aparte, leyendo los angulos (`_wait_home`).
        """
        api = JMAP.to_api_deg(self.q0)[0]
        print(f"homing a {api[:3].tolist()} grados, velocidad {HOMING_SPEED} ...")
        mc = getattr(self.arm, "mc", None)
        if mc is not None and hasattr(mc, "sync_send_angles"):
            mc.sync_send_angles([float(d) for d in api], HOMING_SPEED)
        else:
            self.arm.send(self.q0, speed=HOMING_SPEED)
        self._wait_home(api)

    def _wait_home(self, api: Any) -> None:
        """Lee los angulos hasta que las tres juntas esten a HOME_TOL_DEG de
        `api`, HOME_STABLE_READS lecturas seguidas. Si no pasa en HOME_TIMEOUT_S,
        aborta: el filtro arranca suponiendo esa pose, y un lazo que empieza con
        el brazo en otro lado manda consignas desde un lugar equivocado.

        Varias lecturas seguidas, no una: cubre los ~0.4 s en que el brazo
        todavia no arranco (si arranca lejos, el error es grande y se espera) y
        una lectura suelta buena en medio del movimiento. Una respuesta mala de
        `get_angles` (`-1`, corta) cuenta como "sin lectura", nunca como llegada.
        Abortar no deja nada raro en el brazo: ya tiene mandado ir a home.
        """
        t_end = float(self.clock()) + HOME_TIMEOUT_S
        good = 0
        last: Any = None
        while good < HOME_STABLE_READS:
            if float(self.clock()) > t_end:
                seen = ("sin lectura valida de get_angles" if last is None else
                        f"ultimo error {np.max(np.abs(last[:3] - api[:3])):.1f} grados,"
                        f" angulos {np.round(last[:3], 1).tolist()}")
                raise SystemExit(
                    f"el brazo NO llego a home en {HOME_TIMEOUT_S:.0f} s ({seen}). No se"
                    f" sigue. Tolerancia {HOME_TOL_DEG} grados por junta."
                )
            self._sleep(HOME_POLL_S)
            a = self.arm.read()
            if a is None:
                good = 0
                continue
            last = np.asarray(a, dtype=np.float64)
            err = float(np.max(np.abs(last[:3] - api[:3])))
            good = good + 1 if err <= HOME_TOL_DEG else 0
        print(f"en home: {np.round(last[:3], 2).tolist()} grados"
              f" (error {np.max(np.abs(last[:3] - api[:3])):.2f})")

    def _collect(self, seconds: float) -> tuple[Any, Any]:
        """Junta `seconds` de muestras crudas. -> (t absolutos, Z sin sesgo restado)."""
        t_end = float(self.clock()) + seconds
        ts: list[float] = []
        zs: list[Any] = []
        while float(self.clock()) < t_end:
            self._advance_plant(float(self.clock()))
            batch = self.sensor.drain()
            self.log.extend(batch)
            for m in batch:
                ts.append(float(m.timestamp))
                zs.append(np.asarray(m.z, dtype=np.float64))
            self._sleep(0.02)
        Z = np.array(zs) if zs else np.empty((0, self.imu_rows.size))
        return np.array(ts), Z

    # -- teclado (la misma interfaz que Teleop) -----------------------------

    def on_key(self, code: int) -> None:
        """`key_callback` del visor. Con el brazo real las teclas de junta dan
        UN PASO y nunca quedan trabadas: el visor no avisa la suelta."""
        if code in base.KEYMAP:
            idx, sign = base.KEYMAP[code]
            if self.step_mode or self.real_arm:
                self.nudge(idx, sign)
            else:
                self.held[idx] = sign
        elif code in base.KEYS_LEVEL:
            self.level = base.KEYS_LEVEL[code]
        elif code == base.KEY_STEP_MODE:
            self.step_mode = not self.step_mode
            self.held.clear()
        elif code == base.KEY_STOP:
            self.stop()
        elif code == base.KEY_GHOST:
            self.show_ghost = not self.show_ghost
        elif code == base.KEY_HOME:
            self.home()

    @property
    def jog_rad_s(self) -> float:
        """La velocidad del nivel, recortada al tope: el objetivo nunca se le
        adelanta a la consigna, asi que al soltar el brazo frena donde iba."""
        return min(base.NIVELES[self.level][0], self.slew_rad_s)

    @property
    def step_deg(self) -> float:
        return base.NIVELES[self.level][1]

    def set_target(self, idx: int, value_rad: float) -> None:
        lo, hi = self.ctrl_range[idx]
        self.target[idx] = float(np.clip(value_rad, lo, hi))

    def nudge(self, idx: int, sign: float) -> None:
        self.set_target(idx, self.target[idx] + sign * np.deg2rad(self.step_deg))

    def stop(self) -> None:
        """Frena donde esta la CONSIGNA. El brazo llega ahi ~0.4 s despues."""
        self.held.clear()
        self.target[:] = self.ctrl

    def home(self) -> None:
        self.held.clear()
        self.target[:] = self.q0

    def fault(self, why: str) -> None:
        """Deja de mandar consignas para siempre en esta sesion."""
        self.faulted = why
        self.stop()

    # -- un tick ------------------------------------------------------------

    def jog(self, now: float) -> None:
        """Objetivo con las teclas activas, consigna hacia el objetivo en rampa.

        Con el tiempo REAL transcurrido, no con 1/60: el QTimer no es exacto y
        las velocidades tienen que ser rad/s de verdad. Recortado a 0.1 s por si
        la UI se traba, para que un tick largo no sea un salto."""
        el = min(max(0.0, now - self._t_jog), 0.1)
        self._t_jog = now
        for idx, sign in self.held.items():
            self.set_target(idx, self.target[idx] + sign * self.jog_rad_s * el)
        self.ctrl[:] = slew_limit(self.ctrl, self.target, self.slew_rad_s * el)
        self._ctrl_hist.append((now, self.ctrl.copy()))

    def send_if_due(self, now: float) -> None:
        """A lo sumo STREAM_RATE_HZ, solo si la consigna cambio, sin rafagas."""
        if self.faulted is not None or now < self._next_send:
            return
        period = 1.0 / STREAM_RATE_HZ
        self._next_send += period
        if self._next_send <= now:
            self._next_send = now + period       # un atraso se saltea, no se paga
        if np.array_equal(self.ctrl, self._last_sent):
            return
        q = self.ctrl.copy()
        self._guard(q, now)
        self.arm.send(q, speed=STREAM_SPEED)
        self._last_sent = q
        self._t_last_sent = now
        self.sent.append((now, q))

    def _guard(self, q: Any, now: float) -> None:
        """La ultima guarda antes del puerto. Si salta, el codigo de arriba esta
        roto: se levanta y el lazo deja de mandar."""
        lo, hi = self.ctrl_range[:, 0], self.ctrl_range[:, 1]
        if np.any(q < lo - 1e-9) or np.any(q > hi + 1e-9):
            raise RuntimeError(f"consigna fuera de ctrlrange: {q}")
        allowed = self.slew_rad_s * max(now - self._t_last_sent, 1.0 / STREAM_RATE_HZ)
        if np.max(np.abs(q - self._last_sent)) > allowed * 1.05 + 1e-9:
            raise RuntimeError(
                f"paso de consigna {np.max(np.abs(q - self._last_sent)):.4f} rad >"
                f" {allowed:.4f} permitido por el tope de {self.slew_rad_s} rad/s"
            )

    def _plant_ctrl(self, t: float) -> Any:
        """La consigna que ve la planta en `t - plant_delay_s`, INTERPOLADA.

        La historia tiene un punto por tick (60 Hz) y la planta pide uno por
        paso (500 Hz). Sostener el ultimo punto sumaba hasta un tick de atraso:
        medido en el ensayo, `--plant-delay` agregaba 411 ms en vez de 395. Sin
        atraso pasaba lo contrario -- la planta veia la consigna del tick entero
        desde el primer paso, medio tick ADELANTADA. Interpolando, los dos
        modos ven la misma rampa y el atraso agregado es `plant_delay_s`.
        """
        t_seen = t - self.plant_delay_s
        hist = self._ctrl_hist
        # Tras podar, hist[0] es el ultimo punto <= t_seen (el tiempo de la
        # planta solo avanza) y hist[1], si existe, el primero posterior.
        while len(hist) >= 2 and hist[1][0] <= t_seen:
            hist.popleft()
        ta, qa = hist[0]
        if len(hist) == 1 or not np.isfinite(ta):
            return qa          # despues del ultimo punto, o antes del primero real
        tb, qb = hist[1]
        return qa + (t_seen - ta) / (tb - ta) * (qb - qa)

    def _advance_plant(self, now: float) -> None:
        """Integra la planta hasta `now`. En modo `sim` la IMU la muestrea."""
        n = int((now - self.t_plant) / self.dt)
        if n > MAX_PLANT_STEPS:
            self.t_plant += (n - MAX_PLANT_STEPS) * self.dt   # tiempo salteado, no fisica
            n = MAX_PLANT_STEPS
        for _ in range(max(n, 0)):
            self.data.ctrl[:3] = self._plant_ctrl(self.t_plant + self.dt)
            mj.mj_step(self.model, self.data)
            self.t_plant += self.dt
            if self.imu_kind == "sim":
                self.sensor.poll(self.t_plant, self.data.sensordata)

    def step(self, now: float) -> tuple[Any, Any]:
        """Planta hasta `now`, drenar, restar el sesgo, filtrar. -> (planta, h(x_hat)).

        Mismo orden que el demo simulado: drenar e ingerir UNA vez por tick,
        despues `advance_to_safe`. `SensorError` (lector muerto) sube tal cual:
        el lazo la convierte en `fault`."""
        assert self.runner is not None, "prepare() antes del lazo"
        self.updates = []
        self.meas_new = []
        self._advance_plant(now)
        batch = self.sensor.drain()
        self.log.extend(batch)
        for m in batch:
            z = np.asarray(m.z, dtype=np.float64) - self.bias
            self.meas_new.append((float(m.timestamp), z))
            mc = Measurement(z=z, timestamp=m.timestamp, rows=m.rows, R=m.R, source=m.source)
            info = self.runner.ingest(mc)
            if info is not None:
                self.nis.append(float(info.nis))
                self.applied += 1
                self.updates.append((float(m.timestamp), z, float(info.nis)))
        self.runner.advance_to_safe(now)

        y_plant = np.asarray(self.data.sensordata, dtype=np.float64)[self.imu_rows_plant]
        y_est = h_dyn(self.ekf.x, np.zeros(self.model_b.nu), self.model_b, self.data_b)
        return y_plant, y_est[self.imu_rows]

    def tick(self, now: float) -> tuple[Any, Any]:
        self.jog(now)
        self.send_if_due(now)
        return self.step(now)

    @property
    def q_true(self) -> Any:
        """Juntas del MODELO DE LA CONSIGNA (rad). El panel lo rotula 'planta'."""
        return np.asarray(self.data.qpos[:3], dtype=np.float64)

    @property
    def q_est(self) -> Any:
        return np.asarray(self.ekf.x[:3], dtype=np.float64)

    def ghost_data(self) -> Any:
        """x_hat en un MjData, con mj_forward, para dibujar el fantasma."""
        nv = self.model_b.nv
        self.data_b.qpos[:nv] = self.ekf.x[:nv]
        self.data_b.qvel[:nv] = self.ekf.x[nv : 2 * nv]
        if self.model_b.na:
            self.data_b.act[:] = self.ekf.x[2 * nv :]
        mj.mj_forward(self.model_b, self.data_b)
        return self.data_b

    # -- salida ---------------------------------------------------------------

    def save(self, root: Path) -> Path:
        """CSV crudo de la IMU, consignas y resumen en `root/<fecha>/`."""
        out = root / datetime.now().strftime("%Y%m%d-%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        src = self.sensor.name
        if self.imu_kind == "real":
            # `decoder.b` quedo en cero toda la sesion (el sesgo se resta
            # aparte), asi que `to_raw` reconstruye exacto TODAS las muestras,
            # incluidas las del reposo.
            self.log.save_csv(out / "imu_raw.csv", src, t_ref=self.t0,
                              column_names=list(self.cfg.imu.keys),
                              transform=self.decoder.to_raw)
        else:
            self.log.save_csv(out / "imu_sim.csv", src, t_ref=self.t0,
                              column_names=self.decoder.column_names())
        t_s = np.array([t for t, _ in self.sent]) - self.t0
        q_s = np.array([q for _, q in self.sent]).reshape(-1, 3)
        np.savez(out / "commands.npz", t=t_s, q_rad=q_s,
                 api_deg=JMAP.to_api_deg(q_s) if len(q_s) else np.empty((0, 4)))
        lines = [
            f"arm: {self.arm.name}",
            f"imu: {self.imu_kind}  plant_delay_s: {self.plant_delay_s}"
            f"  slew_rad_s: {self.slew_rad_s}  horizon_s: {self.horizon}",
            f"bias ({'valido' if self.cal is not None and self.cal.valid else 'SIN'}): "
            f"{np.array2string(self.bias, precision=5)}",
            f"steps {self.runner.steps if self.runner else 0}  applied {self.applied}"
            f"  discarded {self.runner.discarded if self.runner else 0}"
            f"  sends {len(self.sent)}",
            f"fault: {self.faulted}",
        ]
        if self.nis:
            lines.append(f"NIS median {np.median(self.nis):.2f}  mean {np.mean(self.nis):.2f}"
                         f"  (n = {len(self.nis)}, objetivo {self.imu_rows.size})")
        (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out


def run(teleop: TeleopReal, with_plots: bool, with_panel: bool) -> None:
    """El lazo: el de `demo_teleop.run`, sobre el reloj de pared."""
    import mujoco.viewer
    from pyqtgraph.Qt import QtCore, QtWidgets

    from erp.viz.ghost import draw_ghost

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    # Ctrl-C: Qt se traga el KeyboardInterrupt. El tick vuelve a Python 60
    # veces por segundo, asi que el handler corre.
    signal.signal(signal.SIGINT, lambda *_: app.quit())

    plots = None
    if with_plots:
        from erp.viz.live import LivePlots

        plots = LivePlots(
            links=list(LINKS),
            joint_labels=list(base.JOINT_NAMES),
            nis_target=float(teleop.imu_rows.size),
            kinds=[("acc", "m/s^2", (-20.0, 20.0)), ("gyro", "rad/s", GYRO_RANGE)],
            truth_label="command model",
            title=f"EKF ciego vs consigna ({teleop.arm.name}, imu {teleop.imu_kind})",
        )
    panel = base.build_panel(teleop) if with_panel else None

    cols = teleop.plot_cols
    state: dict[str, Any] = {"frame": 0, "texto": None}
    err0 = base._check_ghost_starts_on_the_arm(teleop)
    print(f"fantasma sobre la planta al arrancar: {err0:.1e} m de separacion")
    print(base.AYUDA)
    if teleop.real_arm:
        print("  BRAZO REAL: en el visor las teclas de junta dan UN paso. Para mover\n"
              "  continuo, mantener la tecla o el boton en el PANEL. X frena.\n")

    with mujoco.viewer.launch_passive(
        teleop.model, teleop.data, key_callback=teleop.on_key,
        show_left_ui=False, show_right_ui=False,
    ) as viewer:
        view_opts = base._snapshot_view_options(viewer)

        def tick() -> None:
            if not viewer.is_running():
                app.quit()
                return
            base._restore_view_options(viewer, view_opts)
            now = float(teleop.clock())
            try:
                y_plant, y_est = teleop.tick(now)
            except (SensorError, RuntimeError) as e:
                teleop.fault(f"{type(e).__name__}: {e}")
                print(f"\nFALLA, no se mandan mas consignas: {teleop.faulted}")
                app.quit()
                return

            if teleop.show_ghost:
                draw_ghost(viewer.user_scn, teleop.model_b, teleop.ghost_data())
            else:
                viewer.user_scn.ngeom = 0
            runner = teleop.runner
            texto = (
                f"{base._mode_text(teleop)}\n"
                f"IMU aplicadas {teleop.applied}  descartadas "
                f"{runner.discarded if runner else 0}  consignas {len(teleop.sent)}"
            )
            if texto != state["texto"]:
                viewer.set_texts((mj.mjtFontScale.mjFONTSCALE_150,
                                  mj.mjtGridPos.mjGRID_BOTTOMLEFT, texto, None))
                state["texto"] = texto
            viewer.sync()
            if panel is not None:
                panel.refresh()

            if plots is not None:
                t_rel = now - teleop.t0
                plots.push_plant(t_rel, y_plant[cols], teleop.q_true)
                plots.push_estimate(t_rel, y_est[cols], teleop.q_est)
                for t_m, z in teleop.meas_new:
                    plots.push_measurement(t_m - teleop.t0, z[cols])
                for t_m, _, nis in teleop.updates:
                    plots.push_nis(t_m - teleop.t0, nis)
                state["frame"] += 1
                if state["frame"] % max(1, round(base.RENDER_HZ / base.PLOT_HZ)) == 0:
                    plots.redraw()

        timer = QtCore.QTimer()
        timer.timeout.connect(tick)
        timer.start(int(1000.0 / base.RENDER_HZ))
        try:
            app.exec() if hasattr(app, "exec") else app.exec_()
        finally:
            timer.stop()
            # Cerrar la ventana tambien es frenar: nada mas sale al puerto.
            if teleop.faulted is None:
                teleop.fault("fin de la sesion")

    if plots is not None:
        plots.close()
    if panel is not None:
        panel.close()


CHECKLIST = """
  BRAZO REAL. Antes de seguir:
    - la calibracion de ceros esta hecha (los ceros del robot y del modelo coinciden);
    - no hay nada ni nadie en el alcance del brazo;
    - una mano en el interruptor de alimentacion: es el freno de emergencia;
    - el brazo va a hacer HOMING a [0, 0, 0] a velocidad 30, desde donde este.
"""


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    ap.add_argument("--arm", choices=("dry", "real"), default="dry",
                    help="dry = DryRunArm con el atraso medido (defecto); real = pymycobot")
    ap.add_argument("--arm-port", default=None, help="p.ej. COM6; obligatorio con --arm real")
    ap.add_argument("--imu", choices=("sim", "real"), default="sim",
                    help="sim = LiveSimSensor sobre la planta (defecto); real = Teensy")
    ap.add_argument("--imu-port", default=cfg.imu.port, help=f"defecto {cfg.imu.port}")
    ap.add_argument("--plant-delay", action="store_true",
                    help=f"la planta ve la consigna atrasada {cfg.imu.arm_lag_s} s (arm_lag_s)")
    ap.add_argument("--slew", type=float, default=SLEW_RAD_S,
                    help=f"tope de la consigna en rad/s (defecto {SLEW_RAD_S};"
                         f" maximo {SLEW_MAX_REAL} con el brazo real)")
    ap.add_argument("--no-plots", action="store_true", help="sin los graficos en vivo")
    ap.add_argument("--no-panel", action="store_true", help="sin el panel de control")
    ap.add_argument("--no-save", action="store_true", help="no guardar la sesion")
    args = ap.parse_args(argv)

    real_arm = args.arm == "real"
    if real_arm:
        if not args.arm_port:
            ap.error("--arm real necesita --arm-port (p.ej. COM6)")
        print(CHECKLIST)
        if input("  Escribi 'si' para seguir: ").strip().lower() != "si":
            print("cancelado, nada se movio")
            return 1
        arm: ArmInterface = MyPalletizerArm(args.arm_port, JMAP, speed=STREAM_SPEED,
                                            blocking_send=False)
    else:
        arm = DryRunArm(JMAP, latency_s=cfg.imu.arm_lag_s)

    with arm:
        teleop = TeleopReal(
            arm=arm, real_arm=real_arm, imu=args.imu, imu_port=args.imu_port,
            plant_delay_s=cfg.imu.arm_lag_s if args.plant_delay else 0.0,
            slew_rad_s=args.slew,
        )
        try:
            teleop.prepare()
            print(f"sesgo de reposo valido ({teleop.cal.note}); t0 fijado, arranca el lazo")
            run(teleop, with_plots=not args.no_plots, with_panel=not args.no_panel)
        finally:
            if teleop.faulted is None:
                teleop.fault("salida")
            if teleop.imu_kind == "real":
                teleop.sensor.stop()
            hardware = real_arm or args.imu == "real"
            if hardware and not args.no_save and teleop.runner is not None:
                where = teleop.save(repo_root() / "data" / "sessions")
                print(f"sesion guardada en {where}")

    runner = teleop.runner
    print(f"\n=== teleop real (arm = {args.arm}, imu = {args.imu},"
          f" plant_delay = {teleop.plant_delay_s} s) ===")
    print(f"{runner.steps if runner else 0} predicts, {teleop.applied} actualizaciones,"
          f" {runner.discarded if runner else 0} descartadas, {len(teleop.sent)} consignas."
          f" Motivo de fin: {teleop.faulted}")
    if teleop.nis:
        rep = consistency_report(np.asarray(teleop.nis), nz=int(teleop.imu_rows.size),
                                 warmup_fraction=0.5)
        print(rep.summary())
    print(
        "\nQue significa: la PLANTA es un modelo de la CONSIGNA, no la verdad -- en\n"
        "hardware no hay verdad. Sin --plant-delay el fantasma va ~0.4 s atras de\n"
        "ella en movimiento por el atraso de transporte del brazo, no por el filtro.\n"
        "Sobre datos reales fuera de linea este filtro dio NIS mediana 29 contra el\n"
        "objetivo 12; en vivo, con sellos de llegada, esperalo peor. `descartadas`\n"
        "distinto de 0 es lo primero que hay que mirar."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
