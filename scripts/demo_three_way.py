"""Demo de tres vias: verdad de la planta, sensor virtual con ruido, estimacion del EKF.

Fase P8 de ADR-0002. Es el artefacto de aceptacion de la fase y lo que hay que
correr para *ver* al filtro trabajar.

    python scripts/demo_three_way.py                 # sin pantalla, escribe las figuras
    python scripts/demo_three_way.py --live          # ademas abre el visor de MuJoCo
    python scripts/demo_three_way.py --degrade swap  # la misma corrida, rota a proposito

QUE MUESTRA, Y QUE NO. Las tres senales son de la MISMA magnitud y no son
intercambiables:

  verdad     el modelo de planta SIN editar, integrado bajo la consigna. Solo
             existe en simulacion. No es el modelo sobre el que corre el filtro.
  medido     `SimSensor` sobre esa salida grabada, con ruido sacado de N(0, R) y
             liberado a ~20 Hz con latencia.
  estimado   `h(x_hat)` del EKF ciego, que corre sobre `blind_variant` y nunca ve
             la consigna.

Y aca esta la parte incomoda, que el script tambien imprime al terminar: el
sensor virtual saca su ruido de LA MISMA `R` que se le da al filtro, y la planta
ES el modelo. O sea que al filtro se le entrega exactamente el mundo que supone.
Medido sobre esta corrida: NIS mediana ~9.6 contra un objetivo de 12. Sobre el
brazo real, el mismo filtro da NIS mediana 29. Esa diferencia es el contenido
honesto de la demo.

Por eso esto es un chequeo de PLOMERIA -- que los sellos de tiempo, los indices
de fila, los mapas de ejes y el lazo estan alineados -- y no evidencia de que el
estimador sea bueno. La evidencia son NEES y NIS, que
`erp.analysis.consistency_report` calcula y este script reporta, nunca lo bien
que una linea sigue a otra.

`--degrade` existe para que la demo PUEDA fallar. Una demo que solo se ve bien
no distingue un stack que anda de uno que anda de casualidad.

El lazo en vivo no tiene tests y es deliberado: necesita pantalla y bloquea. Lo
que si esta cubierto en la suite rapida es la aritmetica de abajo --
`test_viz.py` para la geometria de la elipse y `test_analysis.py` para el
veredicto.

Castellano, como su unico hermano en `scripts/`. Las etiquetas de las figuras
salen de `erp.viz`, que es ingles: es el mismo corte que el resto del repo,
plomeria en ingles y fisica en castellano.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np

from erp.analysis import consistency_report, propagate_to_site
from erp.core.clock import RateLoop, WallClock
from erp.core.types import Measurement
from erp.estimators import EKF
from erp.fusion import FilterRunner
from erp.io.config import build_decoder, load_config
from erp.io.paths import resolve_repo_path
from erp.sensors import SimSensor
from erp.sensors.base import shared_rows_R
from erp.sensors.mujoco import make_R, rows_of
from erp.sim.dynamics import MujocoDynamics
from erp.sim.mujoco import h_dyn, make_Q, state_dim
from erp.sim.plant import blind_variant, load_model, warmup_to_rest

XML_RELATIVE = Path("mechanical/mujoco_assets/MyPalletizer260/MyPalletizer260.xml")
FIGURE_DIR = Path("data/processed/figures")

# Identicos a los de `make_golden_run.py`: la demo tiene que ser comparable con
# la corrida congelada, y dos juegos de constantes serian dos experimentos.
PERIODO_S = 3
AMPLITUDES_DEG = (30.0, 10.0, 20.0)
WARMUP_STEPS = 100
SIG_ALPHA = 12.0        # rad/s^2 (1 sigma) por paso: DWNA a dt = 2 ms
SIG_ACT_BLIND = 5e-3    # rad por paso: random walk de la activacion
SIG_CM = 1e-4           # rad por paso a lo largo del modo comun
IMU_RATE_HZ = 20.0
IMU_LATENCY_S = 0.005

DEGRADATIONS = ("none", "swap", "zeros", "overconfident")


def build_truth(xml_path: Path) -> dict[str, Any]:
    """Integra la planta SIN editar bajo la consigna y graba lo que leeria.

    La planta, no `blind_variant`: el modelo sin editar es el que GENERA la
    verdad y nunca es sobre el que corre el filtro -- `test_plant.py` afirma que
    la planta falla el contrato `na == 3` del filtro.

    -> dict con t (s), sensordata (N, nsensordata), q_target (N, 3) rad y el
       modelo/data de la planta para el visor.
    """
    from erp.trajectory import sine_sweep

    model, data = load_model(xml_path)
    dt = float(model.opt.timestep)
    t, q_target, _ = sine_sweep(AMPLITUDES_DEG, PERIODO_S, dt)

    data.ctrl[:3] = q_target[0]
    for _ in range(WARMUP_STEPS):
        mj.mj_step(model, data)

    n = len(t)
    sensordata = np.empty((n, model.nsensordata), dtype=np.float64)
    qpos = np.empty((n, model.nq), dtype=np.float64)
    for i in range(n):
        data.ctrl[:3] = q_target[i]
        mj.mj_step(model, data)
        # sensordata se lee DESPUES del paso, como en la corrida congelada.
        sensordata[i] = data.sensordata
        qpos[i] = data.qpos
    return {"t": t, "sensordata": sensordata, "qpos": qpos, "q_target": q_target,
            "model": model, "dt": dt}


def run_estimator(truth: dict[str, Any], xml_path: Path, degrade: str) -> dict[str, Any]:
    """Corre el EKF ciego sobre un `SimSensor` armado desde la verdad."""
    cfg = load_config()
    decoder = build_decoder(cfg.imu)
    model_b, data_b = blind_variant(xml_path)
    nx = state_dim(model_b)

    # Las filas son SIEMPRE las correctas. Una permutacion aplicada a los dos
    # lados -- verdad y medicion -- es consistente y no rompe nada, que es el
    # primer intento de este toggle y no degradaba absolutamente nada.
    imu_rows = rows_of(model_b, *cfg.imu.layout)
    ef_rows = rows_of(model_b, "efector_pos")

    sensordata = truth["sensordata"]
    if degrade == "swap":
        # El cableado que el ajuste descarta: la lectura de link2 entregada en la
        # ranura de link1. `rows` sigue diciendo link1, asi que el filtro compara
        # su prediccion de un eslabon contra el acelerometro del otro. Nada
        # cambia de forma ni de unidades -- es el modo de falla silencioso que
        # ADR-0002 3.3 describe.
        sensordata = np.array(truth["sensordata"], copy=True)
        for a, b in (("link1_acc", "link2_acc"), ("link1_gyro", "link2_gyro")):
            ra, rb = rows_of(model_b, a), rows_of(model_b, b)
            sensordata[:, ra] = truth["sensordata"][:, rb]
            sensordata[:, rb] = truth["sensordata"][:, ra]

    R = make_R(model_b, cfg.imu.sig_acc, cfg.imu.sig_gyro)[np.ix_(imu_rows, imu_rows)]

    sensor = SimSensor(
        truth["t"], sensordata, rows=imu_rows, R=R,
        rate_hz=IMU_RATE_HZ, latency_s=IMU_LATENCY_S, seed=0,
    )
    ms = sensor.drain()
    if degrade == "overconfident":
        # La R del FILTRO es la que viaja en la medicion, no la del constructor
        # del EKF: `FilterRunner.ingest` pasa `m.R` a `update`. Degradar
        # `EKF.R` no hacia nada, que es el segundo toggle que no degradaba nada.
        # El ruido ya se sorteo con la R honesta; aca solo se declara 100 veces
        # menor, o sea un filtro que se cree 10 veces mas preciso de lo que es.
        rows_ro, R_small = shared_rows_R(imu_rows, R / 100.0)
        ms = [Measurement(m.z, m.timestamp, rows_ro, R_small, m.source) for m in ms]

    q0 = truth["q_target"][0]
    x_raw = np.zeros(nx)
    x_raw[: model_b.nv] = data_b.qpos[: model_b.nv]     # keyframe `home`, sin calentar
    x_warm = np.r_[warmup_to_rest(model_b, data_b, q0), q0]
    x0 = x_raw if degrade == "zeros" else x_warm

    # Sesgo ESTATICO de arrancar mal, en sigmas del canal peor. Medido aca y no
    # citado: h(x_warm)[link2_acc_x] = 9.810 m/s^2 contra 4.144 desde el
    # keyframe, o sea 5.7 m/s^2 sobre un sigma de 0.05 -> ~113 sigma.
    u0 = np.zeros(model_b.nu)
    dh = np.abs(h_dyn(x_raw, u0, model_b, data_b)[imu_rows]
                - h_dyn(x_warm, u0, model_b, data_b)[imu_rows])
    rest_bias_sigmas = float((dh / np.sqrt(np.diag(R))).max())

    Q = make_Q(model_b, SIG_ALPHA, SIG_ACT_BLIND, SIG_CM)
    P0 = np.diag([np.deg2rad(1.0) ** 2] * nx)
    ekf = EKF(x0.copy(), P0, Q, R, MujocoDynamics(model_b, data_b))

    runner = FilterRunner(ekf, t0=0.0)
    hist = runner.run(ms)
    p_est, C_est = propagate_to_site(hist.x, hist.P, model_b, data_b, ef_rows)

    # Verdad del efector: la planta la publica como sensor `framepos`, asi que
    # sale del mismo log, sin volver a calcular cinematica.
    ef_rows_plant = rows_of(truth["model"], "efector_pos")
    p_true_full = truth["sensordata"][:, ef_rows_plant]
    p_true = np.asarray([
        p_true_full[min(round(tk / truth["dt"]), len(p_true_full) - 1)] for tk in hist.t
    ])

    return {
        "hist": hist, "imu_rows": imu_rows, "decoder": decoder,
        "t_meas": np.asarray([m.timestamp for m in ms]),
        "y_meas": np.asarray([m.z for m in ms]),
        "p_est": p_est, "C_est": C_est, "p_true": p_true,
        "model_b": model_b, "data_b": data_b, "nz": len(imu_rows), "discarded": hist.discarded,
        "rest_bias_sigmas": rest_bias_sigmas,
    }


def _figures(truth: dict[str, Any], est: dict[str, Any], out_dir: Path, degrade: str) -> None:
    from erp.viz.figures import effector_band, sensor_compare, three_way

    out_dir.mkdir(parents=True, exist_ok=True)
    hist = est["hist"]
    rows = est["imu_rows"]
    t_meas, y_meas = est["t_meas"], est["y_meas"]

    # Dos canales, uno de cada tipo: un acelerometro y un giroscopo. Con 12 el
    # grafico no se lee y la pregunta es la misma.
    channels = [0, 6]
    labels = [f"{n}_{a}" for n, a in (("link1_acc", "x"), ("link1_gyro", "x"))]
    idx = np.clip((hist.t / truth["dt"]).round().astype(int), 0, len(truth["t"]) - 1)
    y_truth = truth["sensordata"][np.ix_(idx, rows)][:, channels]

    from erp.sim.mujoco import h_dyn
    u = np.zeros(est["model_b"].nu)
    y_est = np.asarray([
        h_dyn(x, u, est["model_b"], est["data_b"])[rows][channels] for x in hist.x
    ])

    suffix = "" if degrade == "none" else f"_{degrade}"
    three_way(
        hist.t, y_truth, t_meas, y_meas[:, channels], hist.t, y_est,
        labels=labels, ylabel="m/s^2 and rad/s, site frame",
        title=f"plant vs simulated IMU vs blind EKF ({degrade})",
    ).savefig(out_dir / f"three_way{suffix}.png")

    sensor_compare(
        hist.t_update, hist.nis, nis_target=float(est["nz"]),
        title=f"NIS per applied measurement ({degrade})",
    ).savefig(out_dir / f"sensor_compare{suffix}.png")

    effector_band(
        hist.t, est["p_est"], est["C_est"], p_true=est["p_true"],
        title=f"end-effector, +-2 sigma ({degrade})",
    ).savefig(out_dir / f"effector_band{suffix}.png")
    print(f"figuras en {out_dir}")


def _live(truth: dict[str, Any], est: dict[str, Any]) -> None:
    """Visor de MuJoCo con el brazo moviendose, al ritmo de un `RateLoop`.

    Se pauta con `RateLoop` y no con `sleep(1/rate)` acumulado: medido sobre 76
    ticks a 25 Hz, el lazo acumulado termina 65.4 ms tarde y su error crece
    monotonamente, contra 0.0 ms con plazos absolutos.
    """
    import mujoco.viewer

    model, dt = truth["model"], truth["dt"]
    data = mj.MjData(model)
    n = len(truth["t"])
    stride = max(1, round((1.0 / 60.0) / dt))   # ~60 fps, no un frame por paso
    print("visor abierto: cerralo para terminar")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for tick in RateLoop(WallClock(), 60.0, count=n // stride):
            if not viewer.is_running():
                break
            i = min(tick.i * stride, n - 1)
            data.qpos[:] = truth["qpos"][i]
            mj.mj_forward(model, data)
            viewer.sync()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true", help="abre el visor de MuJoCo")
    ap.add_argument("--degrade", choices=DEGRADATIONS, default="none",
                    help="rompe la corrida a proposito, para que la demo pueda fallar")
    ap.add_argument("--out", type=Path, default=None, help="directorio de figuras")
    ap.add_argument("--no-figures", action="store_true", help="solo el reporte")
    args = ap.parse_args(argv)

    xml_path = resolve_repo_path(*XML_RELATIVE.parts)
    truth = build_truth(xml_path)
    est = run_estimator(truth, xml_path, args.degrade)
    hist = est["hist"]

    kw: dict[str, Any] = {
        "nz": est["nz"], "p_true": est["p_true"],
        "p_est": est["p_est"], "C_site": est["C_est"],
    }
    rep = consistency_report(np.asarray(hist.nis), warmup_fraction=0.5, **kw)
    rep_all = consistency_report(np.asarray(hist.nis), warmup_fraction=0.0, **kw)
    print(f"\n=== demo de tres vias (degrade = {args.degrade}) ===")
    print(f"{len(hist.t)} pasos, {len(hist.nis)} actualizaciones, "
          f"{est['discarded']} descartadas")
    print("\n-- segunda mitad (el veredicto estandar) --")
    print(rep.summary())
    # La ventana completa va tambien, y no por completitud: descartar la primera
    # mitad descarta el transitorio de P0 A PROPOSITO, y un arranque mal
    # inicializado VIVE en ese transitorio. Con solo la segunda mitad,
    # `--degrade zeros` daba numeros identicos a la corrida sana -- un
    # diagnostico que no diagnostica.
    print("\n-- ventana completa --")
    print(rep_all.summary())
    print(
        f"\nsesgo estatico de arrancar del keyframe en vez del equilibrio servoado:"
        f" {est['rest_bias_sigmas']:.0f} sigma en el peor canal.\n"
        "Y esto hay que decirlo con precision, porque exagerarlo seria peor que\n"
        "callarlo: ese sesgo NO sobrevive a la corrida. La igualdad del tendon\n"
        "devuelve el estado al equilibrio en ~2 pasos de fisica, antes de que\n"
        "llegue la primera medicion (5 ms de latencia = 2.5 pasos), asi que\n"
        "`--degrade zeros` deja el NIS intacto. Es un sesgo de h(x0), no una\n"
        "corrida degradada. Los toggles que SI degradan son swap y overconfident."
    )
    print(
        "\nQue significa: el sensor virtual saca su ruido de la MISMA R que se le\n"
        "da al filtro y la planta ES el modelo, asi que esto verifica la plomeria\n"
        "(sellos de tiempo, filas, ejes, lazo), no la calidad del estimador. Sobre\n"
        "el brazo real el mismo filtro da NIS mediana 29 contra el objetivo 12.\n"
        "Para ver la demo fallar: --degrade swap | overconfident."
    )

    if not args.no_figures:
        _figures(truth, est, args.out or resolve_repo_path(*FIGURE_DIR.parts,
                                                           must_exist=False), args.degrade)
    if args.live:
        _live(truth, est)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
