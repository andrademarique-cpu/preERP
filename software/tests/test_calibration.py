"""ADR-0002 P6: the rest-window calibration, and what a calibrated R does.

Two obligations from section 6, and they pull in opposite directions:

* **the obligation** -- a calibrated ``R`` must *change* the filter's NIS,
  otherwise the whole `Measurement.R` path P5 wired is decorative;
* **the falsification** -- an invalid calibration must be *refused*, rather
  than quietly applied as a zero bias, which looks exactly like success to
  anything that does not check ``valid``.

P6 is also the first phase whose obligation is that numbers move, in a tree
whose standard is that they do not. The resolution is that nothing adopts the
calibrated ``R``: it is measured here and used nowhere, and
``test_a_calibrated_R_makes_the_filter_worse_on_real_data`` records why.

Most of this file needs no mujoco. The two that read the recorded log carry
the marker.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from erp.calibration import calibration_from_samples, rest_bias
from erp.core.types import Measurement
from erp.estimators import EKF
from erp.fusion import FilterRunner
from erp.models.linear import LinearDynamics

DIM = 12
ACC = np.arange(0, 6, dtype=np.intp)
GYRO = np.arange(6, 12, dtype=np.intp)
IMU_PERIOD = 0.05


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("no pyproject.toml above this file")


REPO_ROOT = _repo_root()
SCRIPT_PATH = REPO_ROOT / "scripts" / "make_golden_run.py"


def still_run(n: int = 40, *, gyro_bias: float = 0.0, acc_read: float = 9.81,
              seed: int = 0, moving_from: int | None = None):
    """``(t, Z)`` for a logged run: still at first, optionally moving later."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * IMU_PERIOD
    Z = rng.normal(0.0, 0.003, size=(n, DIM))
    Z[:, GYRO] += gyro_bias
    Z[:, ACC] += acc_read
    if moving_from is not None:
        Z[moving_from:, GYRO] += np.sin(np.linspace(0.0, 6.0, n - moving_from))[:, None]
    return t, Z


def expected_at_rest(acc: float = 9.70, gyro: float = 0.0) -> np.ndarray:
    exp = np.zeros(DIM)
    exp[ACC] = acc
    exp[GYRO] = gyro
    return exp


# ------------------------------------------- the arithmetic the cell used to do


def legacy_rest_bias(t, Z, h_rest, *, rest_s, max_gyro_norm, gyro_idx):
    """Notebook cell 19's block, written out. The oracle, not the code under test.

    Raises where `rest_bias` returns ``valid=False``: that difference in shape
    is the phase, and the numbers either side of it must be the same.
    """
    rest = t < rest_s
    rest_gyro = (
        np.linalg.norm(Z[rest][:, gyro_idx], axis=1).max() if rest.any() else np.inf
    )
    if rest.sum() < 3 or rest_gyro > max_gyro_norm:
        raise ValueError("rest window is not still")
    return Z[rest].mean(axis=0) - h_rest


def test_rest_bias_reproduces_the_notebook_cell_bit_for_bit() -> None:
    """P6 moved code; it must not have moved numbers. `array_equal`, not `allclose`."""
    t, Z = still_run(40, gyro_bias=0.01)
    # Gyro entries deliberately NON-zero: this is the case that distinguishes
    # the two formulas, and the case the real pipeline is in.
    h_rest = expected_at_rest(acc=9.70, gyro=2.4e-4)

    cal = rest_bias(t, Z, expected_rest=h_rest, gyro_idx=GYRO, acc_idx=ACC,
                    rest_s=0.4, max_gyro_norm=0.05)
    want = legacy_rest_bias(t, Z, h_rest, rest_s=0.4, max_gyro_norm=0.05, gyro_idx=GYRO)

    assert cal.valid, cal.note
    assert np.array_equal(cal.bias, want)


def test_a_nonzero_gyro_expectation_is_honoured() -> None:
    """The semantic change P6 made, and the reason the test above can pass.

    Before P6 `expected_rest` steered the accelerometer channels only and the
    gyro was always measured against zero. The pipeline passes `h(x_rest)`,
    whose gyro entries are ~2.4e-4 rad/s, so the difference is real.
    """
    t, Z = still_run(40)
    with_expectation = rest_bias(t, Z, expected_rest=expected_at_rest(gyro=1e-3),
                                 gyro_idx=GYRO, acc_idx=ACC)
    against_zero = rest_bias(t, Z, expected_rest=expected_at_rest(gyro=0.0),
                             gyro_idx=GYRO, acc_idx=ACC)

    assert np.allclose(against_zero.bias[GYRO] - with_expectation.bias[GYRO], 1e-3)
    assert np.array_equal(against_zero.bias[ACC], with_expectation.bias[ACC])


def test_zero_gyro_expectation_preserves_the_pre_p6_behaviour() -> None:
    """The pairing: every existing caller passes zeros there, so nothing moved for them."""
    t, Z = still_run(40, gyro_bias=0.01)
    exp = expected_at_rest(gyro=0.0)
    packaged = rest_bias(t, Z, expected_rest=exp, gyro_idx=GYRO, acc_idx=ACC)
    direct = calibration_from_samples(Z[t < 0.4], np.zeros(DIM), gyro_idx=GYRO,
                                      acc_idx=ACC, expected_rest=exp, min_samples=3)
    assert np.array_equal(packaged.bias, direct.bias)
    # Exact, because that IS the definition -- no tolerance to pick.
    assert np.array_equal(packaged.bias, Z[t < 0.4].mean(axis=0) - exp)
    # And it recovers the planted bias to within the sampling error of an
    # 8-sample mean: sigma 0.003 / sqrt(8) = 1.1e-3 per channel, so 4e-3 is
    # ~3.8 standard errors. A tighter bound here would be a flaky test, not a
    # stronger one.
    np.testing.assert_allclose(packaged.bias[GYRO], 0.01, atol=4e-3)


# ---------------------------------------------------------- the falsification


def test_an_invalid_calibration_is_refused_not_returned_as_zero() -> None:
    """The falsification section 6 pairs with the obligation.

    A zero bias is not a neutral default: it is indistinguishable from a
    successful calibration to any caller that does not check `valid`, and it
    silently changes what the run means.
    """
    t, Z = still_run(40, moving_from=0)  # moving from the first sample
    cal = rest_bias(t, Z, expected_rest=expected_at_rest(), gyro_idx=GYRO, acc_idx=ACC)

    assert not cal.valid
    assert "not still" in cal.note
    assert np.array_equal(cal.bias, np.zeros(DIM))  # unchanged, not estimated

    # And the oracle agrees it is unusable -- it raises on the same data.
    with pytest.raises(ValueError):
        legacy_rest_bias(t, Z, expected_at_rest(), rest_s=0.4,
                         max_gyro_norm=0.05, gyro_idx=GYRO)


def test_the_same_data_over_a_still_window_is_accepted() -> None:
    """Paired with the above: a guard that cannot pass is as useless as one that cannot fail."""
    t, Z = still_run(40, moving_from=20)  # still for the first second
    cal = rest_bias(t, Z, expected_rest=expected_at_rest(), gyro_idx=GYRO, acc_idx=ACC)
    assert cal.valid, cal.note


def test_the_live_min_samples_default_would_reject_every_offline_run() -> None:
    """Why `rest_bias` lowers it to 3, pinned so nobody "tidies" it back up.

    0.4 s at ~20 Hz is 8 samples. The live default of 20 suits a 2 s standstill
    on the bench; used offline it returns `valid=False` AND a zero bias, which
    is the failure the test above is about.
    """
    t, Z = still_run(40)
    window = Z[t < 0.4]
    assert len(window) == 8

    strict = calibration_from_samples(window, np.zeros(DIM), gyro_idx=GYRO, acc_idx=ACC,
                                      expected_rest=expected_at_rest(), min_samples=20)
    assert not strict.valid
    assert "need 20" in strict.note
    assert np.array_equal(strict.bias, np.zeros(DIM))

    assert rest_bias(t, Z, expected_rest=expected_at_rest(),
                     gyro_idx=GYRO, acc_idx=ACC).valid


def test_overlapping_gyro_and_acc_indices_are_refused() -> None:
    """A layout mistake that would otherwise produce a plausible-looking bias."""
    t, Z = still_run(40)
    with pytest.raises(ValueError, match="overlap"):
        rest_bias(t, Z, expected_rest=expected_at_rest(), gyro_idx=GYRO, acc_idx=GYRO)


def test_mismatched_shapes_are_refused() -> None:
    t, Z = still_run(40)
    with pytest.raises(ValueError, match="samples"):
        rest_bias(t[:10], Z, expected_rest=expected_at_rest(), gyro_idx=GYRO, acc_idx=ACC)
    with pytest.raises(ValueError, match="channels"):
        rest_bias(t, Z, expected_rest=np.zeros(6), gyro_idx=GYRO, acc_idx=ACC)


# ------------------------------------------------------------- the obligation


def run_with_R(R_block: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """One scalar-channel filter run under a given measurement noise. Returns NIS."""
    dt = 0.002
    A = np.array([[1.0, dt], [0.0, 1.0]])
    C = np.array([[1.0, 0.0]])
    ekf = EKF(np.array([0.0, 1.0]), np.diag([1.0, 1.0]), np.diag([1e-8, 1e-6]),
              np.array([[0.25]]), LinearDynamics(A, C=C, dt=dt))
    rows = np.asarray([0], dtype=np.intp)
    ms = [
        Measurement(z=np.asarray(z, dtype=np.float64), timestamp=0.05 + i * IMU_PERIOD,
                    rows=rows, R=R_block, source="imu")
        for i, z in enumerate(zs)
    ]
    return FilterRunner(ekf, t0=0.0).run(ms).nis


def test_a_calibrated_R_changes_the_filter_nis() -> None:
    """Section 6's P6 obligation, in the fast loop.

    Until P5 the measurement's `R` died at `MeasurementLog`, so calibrating a
    sensor changed nothing downstream. This is the end-to-end statement of the
    opposite: a different `R` on the measurement reaches the filter and moves
    its diagnostic.
    """
    zs = np.random.default_rng(6).normal(0.0, 0.5, size=(12, 1))
    nominal = run_with_R(np.array([[0.25]]), zs)
    calibrated = run_with_R(np.array([[0.01]]), zs)

    assert not np.allclose(nominal, calibrated)
    # A tighter R trusts the measurement more, so the SAME innovation counts
    # for more sigmas: NIS rises. The direction is the point, not the size.
    assert np.median(calibrated) > np.median(nominal)


# ------------------------------------------------- what the real log says (mujoco)

pytest_mujoco = pytest.mark.mujoco


@pytest.fixture(scope="module")
def pipeline() -> ModuleType:
    """`scripts/make_golden_run.py`, imported by path -- `scripts/` is not a package."""
    pytest.importorskip("mujoco", reason="the recorded pipeline is a MuJoCo pipeline")
    if not SCRIPT_PATH.is_file():
        pytest.skip(f"{SCRIPT_PATH.name} is missing")
    spec = importlib.util.spec_from_file_location("erp_p6_pipeline", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest_mujoco
def test_the_recorded_rest_window_calibrates_and_says_how(pipeline: ModuleType) -> None:
    """The real log's rest window is usable, and the result explains itself."""
    cal, _, _ = _recorded_calibration(pipeline)
    assert cal.valid, cal.note
    assert "samples" in cal.note
    assert np.isfinite(cal.bias).all()
    assert np.abs(cal.bias).max() < 5.0  # a bias, not a decoding error


@pytest_mujoco
def test_a_calibrated_R_would_make_the_filter_worse_on_real_data(
    pipeline: ModuleType,
) -> None:
    """Why the pipeline computes `cal.R` and then does not use it.

    The calibrated `R` is the sample covariance of a **static** window: the
    noise floor at rest. It carries nothing about the noise in motion and
    nothing about model error, and the filter's problem on this arm is model
    error -- a ~395 ms transport lag the blind model does not have. Adopting it
    would narrow `R` and make an already over-confident filter worse.

    Asserted as a direction and a floor, not a number, because the recorded log
    in the working tree can legitimately be re-recorded.
    """
    cal, R_nominal, _ = _recorded_calibration(pipeline)
    sigma_ratio = np.sqrt(np.diag(cal.R)) / np.sqrt(np.diag(R_nominal))

    assert (sigma_ratio < 1.0).all(), "a rest-window R that is not tighter is a surprise"
    assert sigma_ratio.max() < 0.75  # measured 0.13-0.39 on the committed log


def _recorded_calibration(pipeline: ModuleType):
    """`(CalibrationResult, R_nominal_block, h_rest)` from the recorded log."""
    import mujoco as mj

    from erp.sensors import IMUDecoder, ReplaySensor
    from erp.sensors.mujoco import make_R, rows_of
    from erp.sim.mujoco import h_dyn
    from erp.sim.plant import blind_variant, load_model, warmup_to_rest
    from erp.trajectory import sine_sweep

    model_path = (REPO_ROOT / pipeline.XML_RELATIVE_PATH).resolve()
    csv_path = (REPO_ROOT / pipeline.IMU_CSV_RELATIVE).resolve()
    if not (model_path.is_file() and csv_path.is_file()):
        pytest.skip("model or recorded log missing (Git-LFS checkout?)")

    model, data = load_model(model_path)
    _, q_target, _ = sine_sweep(pipeline.AMPLITUDES_DEG, pipeline.PERIODO_S,
                                model.opt.timestep)
    decoder = IMUDecoder(pipeline.IMU_KEYS, pipeline.IMU_LAYOUT, pipeline.IMU_AXIS_MAPS)
    rows = rows_of(model, *pipeline.IMU_LAYOUT)
    R_nominal = make_R(model, pipeline.SIG_ACC, pipeline.SIG_GYRO)[np.ix_(rows, rows)]

    ms = ReplaySensor.from_legacy_imu_csv(csv_path, decoder, rows=rows, R=R_nominal,
                                          name="imu").drain()
    t = np.array([m.timestamp for m in ms])
    Z = np.stack([m.z for m in ms])

    model_blind, data_blind = blind_variant(model_path)
    x_rest = np.r_[warmup_to_rest(model, data, q_target[0]), q_target[0]]
    h_rest = h_dyn(x_rest, np.zeros(model_blind.nu), model_blind, data_blind)[rows]
    del mj  # imported only to fail early if mujoco is unusable

    cal = rest_bias(
        t, Z, expected_rest=h_rest,
        gyro_idx=decoder.indices_of(["link1_gyro", "link2_gyro"]),
        acc_idx=decoder.indices_of(["link1_acc", "link2_acc"]),
        rest_s=pipeline.REST_S, max_gyro_norm=pipeline.REST_GYRO_MAX,
    )
    return cal, R_nominal, h_rest
