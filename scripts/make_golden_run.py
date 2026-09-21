"""Congela la corrida de referencia del EKF ciego (ADR-0002, fase P0.5).

Reproduce la MITAD DE ESTIMACION de `notebooks/mypalletizer260EKF.ipynb`
-- la de la seccion 2.2 del ADR -- contra el log ya grabado en
`data/raw/imu_trajectory_raw.csv`, y guarda lo que produce en
`data/processed/golden_ekf_run.npz`.

No toca hardware: el brazo y el Teensy ya corrieron, su salida esta commiteada.
Las unicas dependencias son mujoco, numpy y el propio paquete `erp`.

POR QUE EXISTE ESTE ARCHIVO. M1 (estimacion simulada) ya esta logrado, en
celdas. El riesgo de M1 no es construirlo sino PERDERLO al moverlo al paquete,
asi que se congela primero la salida numerica y despues se refactoriza contra
ella. `software/tests/test_golden_run.py` importa `run_pipeline` de aca y
compara: una sola implementacion, usada para generar y para verificar, para que
el fixture no pueda quedar desincronizado del codigo que dice defender.

ESTE ARCHIVO SE VA VACIANDO SOLO. Arranco con cinco funciones copiadas del
notebook (ADR-0002 5.3: mover sin refactorizar). La fase P1 ya se llevo tres
--- `find_repo_root` -> `erp.io.paths.repo_root`, `compile_blind_model` ->
`erp.sim.plant.blind_variant`, `rest_state` -> `erp.sim.plant.warmup_to_rest`
--- la P3 se llevo la trayectoria, que estaba escrita a mano aca adentro
(`erp.trajectory.sine_sweep`), y la P5 se llevo `run_imu_ekf`
(`erp.fusion.FilterRunner`). El .npz no se movio ni un digito en ninguna de las
tres, que es exactamente el chequeo que P0.5 existe para permitir. Quedan
`estimate_lag` y `site_position_cov`, las dos de la P8 (`ConsistencyReport` y
`propagate_to_site`). Cuando se vayan, este script es solo configuracion y una
llamada --- y ahi recien se cierra M1, porque M1 pide que reproduzca la corrida
EL PAQUETE, no un script.

    python scripts/make_golden_run.py            # escribe el .npz
    python scripts/make_golden_run.py --check    # corre y compara, no escribe
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np

from erp.calibration import rest_bias
from erp.estimators import EKF
from erp.fusion import FilterRunner
from erp.io.paths import repo_root, resolve_repo_path
from erp.sensors import IMUDecoder, ReplaySensor
from erp.sensors.mujoco import make_R, rows_of
from erp.sim.mujoco import H_dyn, h_dyn, make_Q
from erp.sim.dynamics import MujocoDynamics
from erp.sim.plant import blind_variant, load_model, warmup_to_rest
from erp.trajectory import sine_sweep

# --- configuracion, identica a la del notebook -----------------------------

XML_RELATIVE_PATH = Path("mechanical/mujoco_assets/MyPalletizer260/MyPalletizer260.xml")
IMU_CSV_RELATIVE = Path("data/raw/imu_trajectory_raw.csv")
GOLDEN_RELATIVE = Path("data/processed/golden_ekf_run.npz")

PERIODO_S = 3                       # s por ciclo de la trayectoria
AMPLITUDES_DEG = (30.0, 10.0, 20.0)  # rot, link1, link2
WARMUP_STEPS = 100

IMU_KEYS = [
    "IMU_0.ax", "IMU_0.ay", "IMU_0.az",
    "IMU_1.ax", "IMU_1.ay", "IMU_1.az",
    "IMU_0.wx", "IMU_0.wy", "IMU_0.wz",
    "IMU_1.wx", "IMU_1.wy", "IMU_1.wz",
]

# Cableado fisico: IMU_0 en link1, IMU_1 en link2. Ajustado 2026-09-16 contra un
# replay de MuJoCo (48 permutaciones con signo, las dos asignaciones chip->link),
# reproducido en dos grabaciones: gyro rms 0.06-0.10 rad/s contra 0.34 del
# cableado invertido. Coincide con config/estimation.yaml; `conftest.py` y el
# docstring de `IMUDecoder` todavia dicen lo contrario.
IMU_LAYOUT = {
    "link1_acc":  ("IMU_0.ax", "IMU_0.ay", "IMU_0.az"),
    "link2_acc":  ("IMU_1.ax", "IMU_1.ay", "IMU_1.az"),
    "link1_gyro": ("IMU_0.wx", "IMU_0.wy", "IMU_0.wz"),
    "link2_gyro": ("IMU_1.wx", "IMU_1.wy", "IMU_1.wz"),
}
M_IMU0_LINK1 = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]   # Rz(-90)
M_IMU1_LINK2 = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]   # Rz(+90)
IMU_AXIS_MAPS = {
    "link1_acc": M_IMU0_LINK1, "link1_gyro": M_IMU0_LINK1,
    "link2_acc": M_IMU1_LINK2, "link2_gyro": M_IMU1_LINK2,
}

SIG_ACC = 0.05          # m/s^2 por canal -- placeholder hasta calibrar R
SIG_GYRO = 0.005        # rad/s por canal -- placeholder hasta calibrar R
SIG_ALPHA = 12.0        # rad/s^2 (1 sigma) por paso: DWNA a dt = 2 ms
SIG_ACT_BLIND = 5e-3    # rad por paso: random walk de la activacion
SIG_CM = 1e-4           # rad por paso a lo largo del modo comun (q_j, a_j)
REST_S = 0.4            # s: ventana quieta al principio del log
REST_GYRO_MAX = 0.05    # rad/s: si la norma del gyro pasa esto, NO es reposo


# --- funciones levantadas del notebook, sin cambios ------------------------


def estimate_lag(t_ref, q_ref_deg, t_meas, q_meas_deg, max_lag_s=0.6, n=241):
    """Retardo, en s, que mejor alinea la medida con la consigna (RMS minimo).

    Positivo = la medida va ATRASADA respecto de la consigna. Se barre el
    retardo y se interpola la consigna corrida sobre los sellos de tiempo
    REALES de la medida, no sobre los nominales. Solo se usan las muestras que
    caen dentro de la ventana valida, para que el arranque no invente
    correlacion donde no la hay.
    """
    if len(t_meas) < 4:
        return float("nan")
    inside = (t_meas >= t_ref[0] + max_lag_s) & (t_meas <= t_ref[-1])
    if inside.sum() < 4:
        return float("nan")
    tm, qm = t_meas[inside], q_meas_deg[inside]
    best, best_lag = np.inf, float("nan")
    for lag in np.linspace(0.0, max_lag_s, n):
        ref = np.column_stack([np.interp(tm - lag, t_ref, q_ref_deg[:, k])
                               for k in range(q_ref_deg.shape[1])])
        rms = float(np.sqrt(np.mean((qm - ref) ** 2)))
        if rms < best:
            best, best_lag = rms, lag
    return best_lag


def site_position_cov(XH, PP, model, data, rows):
    """Posicion de un sensor `framepos` y su covarianza, propagada desde el estado.

    p = h(x)[rows],   Sigma_p = H P H^T   con   H = dh/dx [rows]   (3 x nx)

    Covarianza COMPLETA, no sqrt(diag(P)) junta por junta: el efector depende de
    las tres juntas a la vez y las correlaciones que el filtro tiene entre ellas
    cambian la banda. H es la misma diferencia finita que usa el filtro; su
    bloque de q coincide con mj_jacSite a 3e-8.

    -> p (N, 3) m, marco mundo;  C (N, 3, 3) m^2
    """
    u = np.zeros(model.nu)
    p = np.empty((len(XH), 3))
    C = np.empty((len(XH), 3, 3))
    for i, (x, P) in enumerate(zip(XH, PP)):
        p[i] = h_dyn(x, u, model, data)[rows]
        H = H_dyn(x, u, model, data)[rows]
        C[i] = H @ P @ H.T
    return p, C


# --- la corrida ------------------------------------------------------------


def run_pipeline(
    root_dir: Path | None = None,
    *,
    layout: dict[str, tuple[str, ...]] | None = None,
    axis_maps: dict[str, Any] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """La mitad de estimacion del notebook, de punta a punta, sin hardware.

    Determinista: no hay RNG en este camino (el ruido solo aparece en
    `SimSensor`, que aca no se usa -- las mediciones son las grabadas).

    `layout` / `axis_maps` existen para poder FALSIFICAR el fixture: el test
    vuelve a correr esto con el cableado invertido y exige que NO reproduzca la
    corrida congelada. Un fixture insensible al cableado no defiende nada.

    -> dict con los arrays que congela el fixture y los escalares de resumen.
    """
    layout = IMU_LAYOUT if layout is None else layout
    axis_maps = IMU_AXIS_MAPS if axis_maps is None else axis_maps
    root = repo_root() if root_dir is None else Path(root_dir)
    model_path = (root / XML_RELATIVE_PATH).resolve()
    csv_path = (root / IMU_CSV_RELATIVE).resolve()
    for p in (model_path, csv_path):
        if not p.is_file():
            raise FileNotFoundError(f"falta {p}")

    # -- 1. planta y trayectoria (celdas 1, 3, 4) ---------------------------
    model, data = load_model(model_path)
    dt = model.opt.timestep

    # La formula del seno estaba copiada aca a mano. Ese era el problema que la
    # fase P3 vino a cerrar: este script REGENERA la trayectoria y con ella el
    # `sensor_log_sim` del que sale `lag_imu`, asi que cambiar el periodo en el
    # notebook y no aca dejaba el fixture describiendo una corrida que el
    # notebook ya no hacia --- y `--check` seguia pasando, porque vuelve a
    # correr este mismo script en vez de comparar contra el notebook.
    t, q_target, _ = sine_sweep(AMPLITUDES_DEG, PERIODO_S, dt)
    n_samples = len(t)

    mj.mj_resetDataKeyframe(model, data, model.key("home").id)
    data.ctrl[:3] = q_target[0]
    for _ in range(WARMUP_STEPS):
        mj.mj_step(model, data)
    data.time = 0.0
    sensor_log_sim = np.empty((n_samples, model.nsensordata), dtype=np.float64)
    for i in range(n_samples):
        data.ctrl[:3] = q_target[i]
        mj.mj_step(model, data)
        # sensordata se lee DESPUES del paso: mj_step calcula los sensores en el
        # estado de ANTES de integrar, asi que esta fila es la del instante i*dt.
        sensor_log_sim[i] = data.sensordata

    # -- 2. configuracion de IMUs (celda 9) ---------------------------------
    imu_decoder = IMUDecoder(IMU_KEYS, layout, axis_maps)
    imu_rows = rows_of(model, *layout)
    imu_R = make_R(model, SIG_ACC, SIG_GYRO)[np.ix_(imu_rows, imu_rows)]

    # -- 3. el log grabado, decodificado con la configuracion de HOY --------
    replay = ReplaySensor.from_legacy_imu_csv(csv_path, imu_decoder, rows=imu_rows,
                                              R=imu_R, name="imu")
    ms = replay.drain()
    if not ms:
        raise RuntimeError(f"{csv_path.name} no tiene muestras")
    imu_t = np.array([m.timestamp for m in ms])
    imu_z = np.stack([m.z for m in ms])

    # -- 4. retardo del brazo, sobre los 6 canales de gyro (celda 16) -------
    # Las columnas del gyro salen de `rows_of`, NO de un slice escrito a mano.
    # El notebook hacia `sensor_log_sim[:, 6:9]` y `[:, 9:12]`: agregar un
    # sensor arriba en el XML corre todo hacia abajo y esos indices siguen
    # devolviendo numeros, solo que del sensor equivocado (ADR-0002 3.6).
    col = {name: imu_decoder.indices_of([name]) for name in layout}
    gyro_sim = sensor_log_sim[:, rows_of(model, "link1_gyro", "link2_gyro")]
    gyro_real = imu_z[:, np.r_[col["link1_gyro"], col["link2_gyro"]]]
    lag_imu = estimate_lag(t, gyro_sim, imu_t, gyro_real)
    if not np.isfinite(lag_imu):
        lag_imu = 0.0

    # -- 5. modelo ciego, reposo y bias (celdas 18, 19) ---------------------
    model_blind, data_blind = blind_variant(model_path)
    assert model_blind.nsensordata == model.nsensordata, "el bloque <sensor> tiene que ser el mismo"
    ef_rows = rows_of(model_blind, "efector_pos")

    x_rest = np.r_[warmup_to_rest(model, data, q_target[0]), q_target[0]]
    h_rest = h_dyn(x_rest, np.zeros(model_blind.nu), model_blind, data_blind)[imu_rows]

    # El bias de reposo se fue a `erp.calibration.rest_bias` en la fase P6. Dos
    # cosas cambian de forma y ninguna de numero:
    #
    # 1. Devuelve un `CalibrationResult` en vez de levantar. Quien llama decide
    #    si una calibracion invalida es fatal; ACA LO ES, porque seguir de
    #    largo restaria un bias de cero y eso se parece demasiado a que salio
    #    bien. El mensaje de la guarda es el mismo de siempre.
    # 2. `min_samples=3`: la ventana de 0.4 s a ~20 Hz tiene 8 muestras, y el
    #    default de `calibration_from_samples` (20, pensado para una
    #    calibracion en vivo de 2 s) rechazaria toda corrida offline.
    #
    # `expected_rest=h_rest` se aplica a TODOS los canales, gyro incluido, que
    # es lo que hacia la celda. Los canales de gyro de `h_rest` no son cero
    # sino hasta 2.4e-4 rad/s --- velocidad residual que deja `warmup_to_rest`,
    # o sea un artefacto del MODELO metido dentro del bias del SENSOR. Esperar
    # cero seria mejor fisica y mueve la corrida congelada un 4.7% de
    # `sig_gyro` en dos canales, asi que no se hace aca: ver el registro de P6.
    cal = rest_bias(
        imu_t, imu_z,
        expected_rest=h_rest,
        gyro_idx=imu_decoder.indices_of(["link1_gyro", "link2_gyro"]),
        acc_idx=imu_decoder.indices_of(["link1_acc", "link2_acc"]),
        rest_s=REST_S,
        max_gyro_norm=REST_GYRO_MAX,
    )
    if not cal.valid:
        raise ValueError(cal.note)
    imu_bias = cal.bias

    # El bias entra POR EL DECODER y se vuelve a decodificar, en vez de
    # restarse despues: es el mismo mecanismo que `apply_calibration` usa en el
    # sensor vivo (`IMUDecoder.apply` hace `raw @ axis_map - b`), asi que
    # offline y en vivo hacen una sola cosa. Es exactamente para esto que el
    # log se guarda CRUDO. Bit a bit identico a restar despues: la expresion y
    # el orden de las operaciones son los mismos.
    imu_decoder.b = imu_bias
    ms = ReplaySensor.from_legacy_imu_csv(csv_path, imu_decoder, rows=imu_rows,
                                          R=imu_R, name="imu").drain()

    # La `R` calibrada (`cal.R`) queda SIN USAR a proposito. Medida sobre estas
    # 8 muestras quietas es 3-8x mas angosta que la nominal, porque es el piso
    # de ruido EN REPOSO: no dice nada del ruido en movimiento ni del error de
    # modelo. Adoptarla haria mas confiado a un filtro que ya lo es de mas
    # (NIS 29 contra un objetivo de 12). `test_calibration.py` lo mide.

    # -- 6. el filtro (celda 19) --------------------------------------------
    R_full = make_R(model_blind, SIG_ACC, SIG_GYRO)
    Q_blind = make_Q(model_blind, SIG_ALPHA, SIG_ACT_BLIND, SIG_CM)
    P0 = np.diag([np.deg2rad(1.0) ** 2] * 4
                 + [np.deg2rad(5.0) ** 2] * 4
                 + [np.deg2rad(0.5) ** 2] * 3)

    # El EKF ya no tiene un MjModel adentro (fase P4): habla con el protocolo
    # `DiscreteDynamics`, y `MujocoDynamics` es quien lo cumple con fisica.
    dyn = MujocoDynamics(model_blind, data_blind)
    ekf = EKF(x_rest, P0, Q_blind, R_full, dyn)

    # `run_imu_ekf` vivia aca; se fue a `erp.fusion.FilterRunner` en la P5. Dos
    # cosas cambian de forma y ninguna de numero:
    #
    # 1. El filtro consume `Measurement`s, no `(t, Z, rows)` sueltos, asi que la
    #    `R` de la medicion ENTRA al update. Es la otra mitad de ADR-0002 3.2.
    #    Hoy da igual --- el sensor arma su R como el mismo bloque que el filtro
    #    ya recortaba de `R_full` --- y por eso la corrida congelada no se mueve.
    #    Empieza a importar en P6, cuando `calibrate()` corra sobre el brazo.
    # 2. El bias ya viene restado por el decoder (paso 5, fase P6).
    runner = FilterRunner(ekf, t0=0.0)
    t_start = time.perf_counter()
    hist = runner.run(ms)
    T_ekf, XH_ekf, PP_ekf, NIS_ekf = hist.t, hist.x, hist.P, hist.nis
    # El log grabado viene ordenado y a 50 ms, o sea ~25 pasos entre muestras:
    # nada puede llegar tarde. Si esto salta, el log dejo de ser lo que el
    # fixture congelo, y las cuentas de abajo no son comparables.
    if hist.discarded:
        raise RuntimeError(f"{hist.discarded} mediciones descartadas por llegar tarde")
    p_ef, C_ef = site_position_cov(XH_ekf, PP_ekf, model_blind, data_blind, ef_rows)
    sig_ef = np.sqrt(np.einsum("nii->ni", C_ef))      # (N, 3) m, 1 sigma por eje del mundo
    wall_s = time.perf_counter() - t_start

    moving = imu_t >= REST_S
    out: dict[str, Any] = {
        "T_ekf": T_ekf,
        "XH_ekf": XH_ekf,
        "NIS_ekf": NIS_ekf,
        "p_ef": p_ef,
        "sig_ef": sig_ef,
        "imu_bias": imu_bias,
        "x_rest": x_rest,
        "lag_imu": np.float64(lag_imu),
        "nis_median_moving": np.float64(np.median(NIS_ekf[moving])),
        "nis_mean_moving": np.float64(NIS_ekf[moving].mean()),
        "sig_ef_median_mm": np.median(sig_ef, axis=0) * 1e3,
        "n_measurements": np.int64(len(imu_t)),
        "n_steps": np.int64(len(T_ekf)),
        # Forma del modelo ciego. Es la obligacion de la fase P1 ("blind_variant
        # da na = 3, nx = 11 y el mismo nsensordata que la planta"), y se congela
        # aca porque si cambia, todo lo de arriba deja de ser comparable.
        "na_blind": np.int64(model_blind.na),
        "nx_blind": np.int64(2 * model_blind.nv + model_blind.na),
        "nsensordata": np.int64(model_blind.nsensordata),
    }
    if verbose:
        print(f"EKF ciego: {len(imu_t)} mediciones, {len(T_ekf)} pasos en "
              f"{T_ekf[-1]:.2f} s, corrio en {wall_s:.1f} s")
        print("bias de reposo restado:", np.round(imu_bias, 3))
        print(f"retardo brazo vs MuJoCo (gyro): {lag_imu * 1e3:.0f} ms")
        print(f"NIS en movimiento ({moving.sum()} muestras, objetivo {len(imu_rows)}): "
              f"media {out['nis_mean_moving']:.1f}, mediana {out['nis_median_moving']:.1f}")
        print("sigma del efector [mm] x/y/z, mediana:",
              np.round(out["sig_ef_median_mm"], 1))
        print("juntas al final [deg] (rot, link1, link2):",
              np.round(np.rad2deg(XH_ekf[-1, :3]), 2))
    return out


def _metadata() -> dict[str, Any]:
    return {
        "meta_mujoco": mj.__version__,
        "meta_numpy": np.__version__,
        "meta_python": platform.python_version(),
        "meta_platform": platform.platform(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="corre y compara contra el .npz existente, sin escribir")
    ap.add_argument("--rtol", type=float, default=1e-12)
    args = ap.parse_args(argv)

    root = repo_root()
    out_path = resolve_repo_path(GOLDEN_RELATIVE, must_exist=False)
    result = run_pipeline(root, verbose=True)

    if args.check:
        if not out_path.is_file():
            print(f"\nno existe {out_path}; corre sin --check para crearlo", file=sys.stderr)
            return 1
        ref = np.load(out_path)
        bad = []
        for key in sorted(k for k in result if not k.startswith("meta_")):
            if key not in ref.files:
                bad.append(f"{key}: no esta en el fixture")
                continue
            a, b = np.asarray(result[key]), np.asarray(ref[key])
            if a.shape != b.shape:
                bad.append(f"{key}: shape {a.shape} vs {b.shape}")
            elif not np.allclose(a, b, rtol=args.rtol, atol=0.0):
                d = np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))
                bad.append(f"{key}: rtol maximo {d:.3e} > {args.rtol:.0e}")
        if bad:
            print("\nDIFERENCIAS contra el fixture:", file=sys.stderr)
            for line in bad:
                print("  " + line, file=sys.stderr)
            return 1
        print(f"\nOK: reproduce {out_path.name} a rtol={args.rtol:.0e}")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **result, **_metadata())
    size_kb = out_path.stat().st_size / 1024
    print(f"\nescrito {out_path.relative_to(root)} ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
