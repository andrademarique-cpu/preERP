"""Demo interactivo: manejas la planta con el teclado y ves al EKF ciego seguirla.

    python scripts/demo_teleop.py
    python scripts/demo_teleop.py --degrade swap          # el fantasma se despega
    python scripts/demo_teleop.py --degrade overconfident
    python scripts/demo_teleop.py --no-plots              # solo el visor

LA TESIS. El EKF es ciego a la consigna **por construccion**: `EKF` no tiene
donde meter una `u`, y un test lo afirma por firma. Nunca ve la tecla que
apretas. Estima las activaciones de los tres servos a partir de dos IMUs
ruidosas y nada mas. Por eso esta demo muestra algo que un barrido sinusoidal
graficado despues no puede: cuando movés una junta, el fantasma se atrasa y
despues alcanza a la verdad, y eso es el filtro *infiriendo* un comando que no
recibio.

LAS TRES SENALES, que son de la MISMA magnitud y no son intercambiables:

  verdad     el modelo de planta SIN editar, integrado bajo tu consigna. Solo
             existe en simulacion. No es el modelo sobre el que corre el filtro.
  medido     `LiveSimSensor` muestreando esa planta a ~20 Hz, con ruido de
             N(0, R) y latencia de transporte.
  estimado   `h(x_hat)` del EKF ciego, que corre sobre `blind_variant`.

Y LO INCOMODO, que el script tambien imprime al salir: el sensor virtual saca
su ruido de LA MISMA `R` que se le da al filtro, y la planta ES el modelo. Al
filtro se le entrega exactamente el mundo que supone. Sobre el brazo real el
mismo filtro da NIS mediana 29 contra ~9.6 aca. Esto es un chequeo de PLOMERIA
-- sellos de tiempo, indices de fila, mapas de ejes y lazo -- y no evidencia de
que el estimador sea bueno.

`--degrade` existe para que la demo PUEDA fallar. Una demo que solo se ve bien
no distingue un stack que anda de uno que anda de casualidad.

EL LAZO NO TIENE TESTS, y es deliberado: necesita pantalla y bloquea. Lo que si
esta cubierto en la suite rapida es la aritmetica de abajo -- `test_live_sensor.py`
para el muestreo y la latencia, `test_viz_live.py` para el buffer circular y
para la construccion de los geoms del fantasma (incluida la trampa del
`dataid`), `test_sensor_contract.py` para el contrato del sensor nuevo.

Castellano, como su unico hermano en `scripts/`. Las etiquetas salen de
`erp.viz`, que es ingles: el mismo corte que el resto del repo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np

from erp.analysis import consistency_report
from erp.estimators import EKF
from erp.fusion import FilterRunner
from erp.io.config import build_decoder, load_config
from erp.io.paths import resolve_repo_path
from erp.sensors import LiveSimSensor
from erp.sensors.mujoco import make_R, rows_of
from erp.sim.dynamics import MujocoDynamics
from erp.sim.mujoco import h_dyn, make_Q, state_dim
from erp.sim.plant import blind_variant, load_model, warmup_to_rest

XML_RELATIVE = Path("mechanical/mujoco_assets/MyPalletizer260/MyPalletizer260.xml")

# Identicos a los de `make_golden_run.py` y `demo_three_way.py`: las tres corridas
# tienen que ser comparables, y tres juegos de constantes serian tres experimentos.
WARMUP_STEPS = 100
SIG_ALPHA = 12.0        # rad/s^2 (1 sigma) por paso: DWNA a dt = 2 ms
SIG_ACT_BLIND = 5e-3    # rad por paso: random walk de la activacion
SIG_CM = 1e-4           # rad por paso a lo largo del modo comun
IMU_RATE_HZ = 20.0
IMU_LATENCY_S = 0.005

RENDER_HZ = 60.0        # el tick de Qt; ~8 pasos de fisica por tick a dt = 2 ms
PLOT_HZ = 30.0          # redibujar la mitad de los ticks: el resto es presupuesto
JOG_RAD_S = 0.6         # velocidad de comando mientras una tecla esta apretada
DEGRADATIONS = ("none", "swap", "overconfident", "zeros")

# Teclas -> (indice de actuador, signo). `key_callback` recibe un codigo GLFW,
# que para letras es ord(MAYUSCULA). Las teclas solo llegan a la ventana que
# tiene el foco, o sea la del visor de MuJoCo, no la de los graficos.
KEYMAP = {
    ord("Q"): (0, +1.0), ord("A"): (0, -1.0),
    ord("W"): (1, +1.0), ord("S"): (1, -1.0),
    ord("E"): (2, +1.0), ord("D"): (2, -1.0),
}
AYUDA = """
  Q / A   rot_servo    -/+        G     fantasma on/off
  W / S   link1_servo  -/+        R     volver al reposo
  E / D   link2_servo  -/+        ESC   salir  (o cerra el visor)

  La 4a junta (`act`) esta acoplada por tendon, rango [0, 0]: sigue, no se
  comanda. Las teclas van a la ventana del VISOR, no a la de los graficos.
"""


class SimClock:
    """El reloj de la simulacion, como `Clock`. No es un reloj de pared.

    El lazo avanza la fisica en pasos de `dt`, asi que el tiempo de la corrida
    es `t` y no `perf_counter()`. Darle ESTE reloj al runner es lo que hace que
    una muestra retenida se libere cuando el tiempo de la SIMULACION la alcanza;
    con un reloj de pared, una demo que corre mas lento o mas rapido que el
    tiempo real liberaria las muestras en el momento equivocado.
    """

    def __init__(self, owner: Teleop) -> None:
        self._owner = owner

    def now(self) -> float:
        return self._owner.t

    def sleep_until(self, t: float) -> None:
        """No-op: el que avanza el tiempo es el lazo de fisica, no este reloj."""


class Teleop:
    """Estado de una sesion interactiva. Un tick = varios pasos de fisica.

    El tiempo de esta demo es `self.t`, que avanza `dt` por paso de fisica y es
    la base de tiempo **unica**: el sensor sella con ella, el filtro avanza con
    ella y la latencia se libera con ella. Usar `perf_counter()` para una parte
    y el reloj de simulacion para otra pone cada muestra en el pasado o en el
    futuro por la diferencia entre las dos bases, que es constante y silenciosa.
    """

    def __init__(self, xml_path: Path, degrade: str) -> None:
        self.degrade = degrade
        cfg = load_config()
        build_decoder(cfg.imu)      # valida el cableado del config antes de arrancar

        self.model, self.data = load_model(xml_path)
        self.dt = float(self.model.opt.timestep)
        self.model_b, self.data_b = blind_variant(xml_path)
        self.nx = state_dim(self.model_b)

        self.imu_rows = rows_of(self.model_b, *cfg.imu.layout)
        self.imu_rows_plant = rows_of(self.model, *cfg.imu.layout)
        R_full = make_R(self.model_b, cfg.imu.sig_acc, cfg.imu.sig_gyro)
        self.R = R_full[np.ix_(self.imu_rows, self.imu_rows)]

        # Consigna inicial: el keyframe `home`, que usa act = 1.5693 y no 1.57
        # para que la igualdad del tendon se cumpla exacto y el solver no
        # empuje el brazo en el primer paso.
        self.ctrl = np.asarray(self.data.ctrl[:3], dtype=np.float64).copy()
        self.ctrl_range = np.asarray(self.model.actuator_ctrlrange, dtype=np.float64)
        self.q0 = self.ctrl.copy()
        for _ in range(WARMUP_STEPS):
            self.data.ctrl[:3] = self.ctrl
            mj.mj_step(self.model, self.data)

        # `swap`: la lectura de link2 entregada en la ranura de link1, con `rows`
        # diciendo todavia link1. Nada cambia de forma ni de unidades -- es el
        # modo de falla silencioso de ADR-0002 3.3. Un solo lado: permutar los
        # dos lados es consistente y no degrada nada.
        self.swap_pairs: list[tuple[Any, Any]] = []
        if degrade == "swap":
            self.swap_pairs = [
                (rows_of(self.model, a), rows_of(self.model, b))
                for a, b in (("link1_acc", "link2_acc"), ("link1_gyro", "link2_gyro"))
            ]

        R_declared = self.R / 100.0 if degrade == "overconfident" else self.R
        self.t = 0.0
        self.sensor = LiveSimSensor(
            rows=self.imu_rows, R=R_declared, rate_hz=IMU_RATE_HZ,
            latency_s=IMU_LATENCY_S, seed=0, clock=lambda: self.t, keep_truth=False,
        )

        # `zeros`: arrancar del keyframe en vez del equilibrio servoado. Es un
        # sesgo de 113 sigma en h(x0) que NO sobrevive la corrida -- el tendon
        # devuelve el estado en ~2 pasos, antes de la primera medicion a 5 ms.
        x_raw = np.zeros(self.nx)
        x_raw[: self.model_b.nv] = self.data_b.qpos[: self.model_b.nv]
        x_warm = np.r_[warmup_to_rest(self.model_b, self.data_b, self.q0), self.q0]
        x0 = x_raw if degrade == "zeros" else x_warm

        Q = make_Q(self.model_b, SIG_ALPHA, SIG_ACT_BLIND, SIG_CM)
        P0 = np.diag([np.deg2rad(1.0) ** 2] * self.nx)
        self.ekf = EKF(x0.copy(), P0, Q, R_declared, MujocoDynamics(self.model_b, self.data_b))
        # `buffer_horizon` = la latencia del sensor, y NO 0.0. Es el unico valor
        # que funciona con una fuente viva, y descubrirlo costo una corrida:
        # `advance_to_safe(t)` con horizonte 0 ES `advance_to(t)`, asi que el
        # filtro se adelanta hasta ahora y cada muestra llega sellada 5 ms en su
        # pasado -> descartada. Medido: 0 de 60 actualizaciones aplicadas, sin
        # un solo error, con el visor andando y los graficos dibujando.
        # El `clock` acompana por obligacion: con horizonte > 0 y sin reloj, la
        # marca de agua es el sello mas nuevo INGERIDO, o sea que cada muestra
        # espera a su sucesora -- 50 ms de atraso en vez de 5.
        #
        # Y el horizonte se redondea ARRIBA a un numero entero de pasos de
        # fisica, que es la segunda mitad del mismo descubrimiento.
        # `_step_of` usa `round`, y 5 ms son 2.5 pasos de 2 ms: la marca de
        # agua `t - horizonte` cae SIEMPRE exactamente en un empate de
        # redondeo, y `round` en Python desempata al par. O sea que la mitad
        # de las veces el filtro se pasa un paso de la muestra que estaba por
        # llegar y la descarta. Medido con horizonte = 5 ms: 41 descartadas de
        # 58, NIS mediana 6242 contra 12, y el estimado sin seguir nada --
        # con el visor andando y los graficos dibujando, que es lo que hace
        # que esto sobreviva a una demo.
        horizon = np.ceil(IMU_LATENCY_S / self.dt) * self.dt
        self.runner = FilterRunner(
            self.ekf, t0=0.0, buffer_horizon=float(horizon),
            clock=SimClock(self), record=False,
        )

        self.held: dict[int, float] = {}
        self.show_ghost = True
        self.nis: list[float] = []
        self.applied = 0

    # -- teclado ------------------------------------------------------------

    def on_key(self, code: int) -> None:
        """`key_callback` del visor pasivo. Recibe un codigo GLFW."""
        if code in KEYMAP:
            idx, sign = KEYMAP[code]
            self.held[idx] = sign
        elif code == ord("G"):
            self.show_ghost = not self.show_ghost
        elif code == ord("R"):
            self.ctrl[:] = self.q0

    def jog(self) -> None:
        """Mueve la consigna mientras hay teclas activas, recortada al rango.

        Recorta contra `actuator_ctrlrange` y no contra +-algo simetrico: los
        rangos de este brazo son asimetricos (rot +-2.79, link1 0..1.57, link2
        0..1.04), asi que un paso simetrico desde cero se sale del rango en dos
        de los tres en el primer tick.
        """
        if not self.held:
            return
        step = JOG_RAD_S / RENDER_HZ
        for idx, sign in self.held.items():
            lo, hi = self.ctrl_range[idx]
            self.ctrl[idx] = float(np.clip(self.ctrl[idx] + sign * step, lo, hi))
        self.held.clear()       # sin repeticion de tecla: un evento, un paso

    # -- un tick ------------------------------------------------------------

    def step(self, n_steps: int) -> tuple[Any, Any]:
        """Avanza `n_steps` de fisica, muestrea, filtra. -> (lectura, h(x_hat))."""
        for _ in range(n_steps):
            self.data.ctrl[:3] = self.ctrl
            mj.mj_step(self.model, self.data)
            self.t += self.dt

            reading = np.asarray(self.data.sensordata, dtype=np.float64).copy()
            if self.swap_pairs:
                for ra, rb in self.swap_pairs:
                    reading[ra], reading[rb] = (
                        np.array(reading[rb]), np.array(reading[ra]),
                    )
            self.sensor.poll(self.t, reading)

        # Drenar y filtrar UNA vez por tick, no una vez por paso de fisica.
        # Por paso, `advance_to_safe` corre 8 veces por tick y cada vez empuja
        # el filtro hasta `t - horizonte`, que es justo el instante de la
        # muestra que el sensor esta por liberar: el filtro le gana la carrera
        # a su propia medicion. El orden tambien importa -- primero ingerir
        # (cada `_apply` avanza hasta SU sello) y despues alcanzar el presente.
        for m in self.sensor.drain():
            info = self.runner.ingest(m)
            if info is not None:
                self.nis.append(float(info.nis))
                self.applied += 1
        # `advance_to_safe`, no `advance_to(now)`: el sensor sella el instante
        # de MUESTRA y la entrega despues, asi que un filtro avanzado hasta
        # ahora esta adelante de lo que llega y lo descarta.
        self.runner.advance_to_safe(self.t)

        y_plant = np.asarray(self.data.sensordata, dtype=np.float64)[self.imu_rows_plant]
        u0 = np.zeros(self.model_b.nu)
        y_est = h_dyn(self.ekf.x, u0, self.model_b, self.data_b)[self.imu_rows]
        return y_plant, y_est

    @property
    def q_true(self) -> Any:
        return np.asarray(self.data.qpos[:3], dtype=np.float64)

    @property
    def q_est(self) -> Any:
        return np.asarray(self.ekf.x[:3], dtype=np.float64)

    def ghost_data(self) -> Any:
        """Pone x_hat en un MjData y corre mj_forward, para dibujar el fantasma."""
        self.data_b.qpos[: self.model_b.nv] = self.ekf.x[: self.model_b.nv]
        self.data_b.qvel[: self.model_b.nv] = self.ekf.x[self.model_b.nv : 2 * self.model_b.nv]
        if self.model_b.na:
            self.data_b.act[:] = self.ekf.x[2 * self.model_b.nv :]
        mj.mj_forward(self.model_b, self.data_b)
        return self.data_b


def run(teleop: Teleop, with_plots: bool) -> None:
    """El lazo. Qt es el dueno del loop principal; el visor se sincroniza a mano.

    `launch_passive` NO bloquea: devuelve un handle y uno llama `.sync()`. Por
    eso se puede tener un `QTimer` manejando todo y no dos event loops peleando
    por el hilo principal.
    """
    import mujoco.viewer
    from pyqtgraph.Qt import QtCore, QtWidgets

    from erp.viz.ghost import draw_ghost

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    plots = None
    if with_plots:
        from erp.viz.live import LivePlots

        # Dos canales, uno de cada tipo. Con 12 el grafico no se lee y la
        # pregunta es la misma.
        plots = LivePlots(
            channel_labels=["link1_acc_x", "link1_gyro_x"],
            joint_labels=["rot", "link1", "link2"],
            nis_target=float(teleop.imu_rows.size),
            title=f"EKF ciego vs planta (degrade = {teleop.degrade})",
        )

    n_steps = max(1, round((1.0 / RENDER_HZ) / teleop.dt))
    channels = [0, 6]
    state = {"frame": 0}
    print(AYUDA)

    with mujoco.viewer.launch_passive(
        teleop.model, teleop.data, key_callback=teleop.on_key
    ) as viewer:

        def tick() -> None:
            if not viewer.is_running():
                app.quit()
                return
            teleop.jog()
            y_plant, y_est = teleop.step(n_steps)

            if teleop.show_ghost:
                draw_ghost(viewer.user_scn, teleop.model_b, teleop.ghost_data())
            else:
                viewer.user_scn.ngeom = 0
            viewer.sync()

            if plots is not None:
                plots.push_plant(teleop.t, y_plant[channels], teleop.q_true)
                plots.push_estimate(teleop.t, y_est[channels], teleop.q_est)
                state["frame"] += 1
                if state["frame"] % max(1, round(RENDER_HZ / PLOT_HZ)) == 0:
                    plots.redraw()

        timer = QtCore.QTimer()
        timer.timeout.connect(tick)
        timer.start(int(1000.0 / RENDER_HZ))
        app.exec() if hasattr(app, "exec") else app.exec_()
        timer.stop()

    if plots is not None:
        plots.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    ap.add_argument("--degrade", choices=DEGRADATIONS, default="none",
                    help="rompe la corrida a proposito, para que la demo pueda fallar")
    ap.add_argument("--no-plots", action="store_true", help="solo el visor 3D")
    args = ap.parse_args(argv)

    xml_path = resolve_repo_path(*XML_RELATIVE.parts)
    teleop = Teleop(xml_path, args.degrade)
    run(teleop, with_plots=not args.no_plots)

    print(f"\n=== teleop (degrade = {args.degrade}) ===")
    print(f"{teleop.runner.steps} predicts, {teleop.applied} actualizaciones aplicadas, "
          f"{teleop.runner.discarded} descartadas")
    if teleop.nis:
        rep = consistency_report(np.asarray(teleop.nis), nz=int(teleop.imu_rows.size),
                                 warmup_fraction=0.5)
        print(rep.summary())
    else:
        print("sin actualizaciones: no se movio nada o se cerro muy rapido")
    print(
        "\nQue significa: el sensor virtual saca su ruido de la MISMA R que se le\n"
        "da al filtro y la planta ES el modelo, asi que esto verifica la plomeria\n"
        "(sellos de tiempo, filas, ejes, lazo), no la calidad del estimador. Sobre\n"
        "el brazo real el mismo filtro da NIS mediana 29 contra el objetivo 12.\n"
        "Para ver la demo fallar: --degrade swap | overconfident."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
