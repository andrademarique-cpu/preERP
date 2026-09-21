"""The frozen EKF run: the pipeline must still produce what it produced (ADR-0002 P0.5).

M1 -- consistent estimation against MuJoCo-generated data -- is already
achieved, but it lives in notebook cells. Its risk is therefore regression, not
construction: moving that code into the package can only lose it. This module
is what makes the loss detectable, by replaying the whole estimation half
against the recorded log in ``data/raw/imu_trajectory_raw.csv`` and diffing
every array against ``data/processed/golden_ekf_run.npz``.

The pipeline itself lives in ``scripts/make_golden_run.py`` and is imported
here rather than copied, so the fixture and the test can never drift apart:
whatever wrote the ``.npz`` is exactly what this file re-runs.

Two ways this differs from the rest of the suite, both deliberate:

- **It needs mujoco and the Git-LFS assets, and it is not fast** (~0.4 s per
  pipeline run, against ~0.4 s for the whole rest of the suite). mujoco has
  been a hard dependency since ADR-0002 P0, but the model also pulls STL meshes
  out of LFS, so these are marked ``mujoco``; ``pytest -m "not mujoco"`` keeps
  the quick loop quick.
- **It asserts against a binary fixture.** That is only evidence if the fixture
  is sensitive to the things that matter, which is what
  :func:`test_swapped_wiring_does_not_reproduce_the_golden_run` establishes.
  A golden test that passes under a known-wrong configuration is theatre.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

# rtol from ADR-0002 section 6: this is a "did anything move" bound, not an
# accuracy claim. The pipeline has no RNG, so on one machine it is exact; the
# tolerance is here for BLAS differences across platforms.
RTOL = 1e-12

# Arrays the fixture freezes, compared one by one so a failure names the stage
# that moved rather than just "the run changed".
GOLDEN_ARRAYS = (
    "T_ekf",        # (n,) s, filter time at every step
    "XH_ekf",       # (n, 11) state at every step
    "NIS_ekf",      # (72,) NIS of each measurement
    "p_ef",         # (n, 3) m, end-effector position, world frame
    "sig_ef",       # (n, 3) m, 1 sigma per world axis from H P H^T
    "imu_bias",     # (12,) rest-window bias subtracted, calibrated units
    "x_rest",       # (11,) servoed equilibrium the filter starts from
)
GOLDEN_SCALARS = (
    "lag_imu",              # s, arm transport lag against the MuJoCo replay
    "nis_median_moving",    # NIS median with the arm moving (target 12)
    "nis_mean_moving",
    "n_measurements",
    "n_steps",
    "na_blind",
    "nx_blind",
    "nsensordata",
)

# The wiring the config's fit rejects: IMU_1 on link1, IMU_0 on link2. This is
# what software/tests/conftest.py and IMUDecoder's own docstring still claim,
# and what config/estimation.yaml contradicts with measured residuals.
SWAPPED_LAYOUT = {
    "link1_acc":  ("IMU_1.ax", "IMU_1.ay", "IMU_1.az"),
    "link2_acc":  ("IMU_0.ax", "IMU_0.ay", "IMU_0.az"),
    "link1_gyro": ("IMU_1.wx", "IMU_1.wy", "IMU_1.wz"),
    "link2_gyro": ("IMU_0.wx", "IMU_0.wy", "IMU_0.wz"),
}

pytestmark = pytest.mark.mujoco


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("no pyproject.toml above this file")


REPO_ROOT = _repo_root()
GOLDEN_PATH = REPO_ROOT / "data" / "processed" / "golden_ekf_run.npz"
SCRIPT_PATH = REPO_ROOT / "scripts" / "make_golden_run.py"


def _load_pipeline() -> ModuleType:
    """Import ``scripts/make_golden_run.py`` by path; ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("erp_golden_pipeline", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pipeline() -> ModuleType:
    pytest.importorskip("mujoco", reason="the golden run is a MuJoCo pipeline")
    if not SCRIPT_PATH.is_file():
        pytest.skip(f"{SCRIPT_PATH.name} is missing")
    return _load_pipeline()


@pytest.fixture(scope="module")
def golden() -> Any:
    if not GOLDEN_PATH.is_file():
        pytest.skip(f"no fixture at {GOLDEN_PATH}; run `python scripts/make_golden_run.py`")
    return np.load(GOLDEN_PATH)


@pytest.fixture(scope="module")
def result(pipeline: ModuleType) -> dict[str, Any]:
    """One pipeline run, shared by every comparison in this module."""
    return dict(pipeline.run_pipeline(REPO_ROOT))


@pytest.mark.parametrize("key", GOLDEN_ARRAYS)
def test_arrays_reproduce_the_golden_run(result: dict[str, Any], golden: Any, key: str) -> None:
    got, want = np.asarray(result[key]), np.asarray(golden[key])
    assert got.shape == want.shape, f"{key}: shape {got.shape}, fixture has {want.shape}"
    assert np.allclose(got, want, rtol=RTOL, atol=0.0), (
        f"{key}: worst relative difference "
        f"{np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-300)):.3e} > {RTOL:.0e}"
    )


@pytest.mark.parametrize("key", GOLDEN_SCALARS)
def test_scalars_reproduce_the_golden_run(result: dict[str, Any], golden: Any, key: str) -> None:
    got, want = float(result[key]), float(golden[key])
    assert got == pytest.approx(want, rel=RTOL, abs=0.0), f"{key}: {got} vs fixture {want}"


def test_blind_model_has_the_documented_shape(result: dict[str, Any]) -> None:
    """``na`` 0 -> 3 and ``nx`` 8 -> 11, with the sensor block untouched.

    Written out rather than left inside the binary fixture: these three numbers
    are the reason the activation is estimable at all, and P1's ``blind_variant``
    has to keep producing them.
    """
    assert int(result["na_blind"]) == 3, "the three position servos must become activations"
    assert int(result["nx_blind"]) == 11, "nx = 2 * nv + na = 2 * 4 + 3"
    assert int(result["nsensordata"]) == 15, "12 IMU channels + 3 efector_pos"


def test_filter_is_not_yet_consistent_on_real_data(result: dict[str, Any]) -> None:
    """The frozen run records NIS median ~29 against a target of 12.

    Asserted so the fixture cannot be mistaken for a passing consistency check.
    ADR-0002 section 4.8 is explicit that real data is ~2.4x overconfident, and
    M1's actual acceptance criterion (NIS 11-13) is measured on *simulated*
    data, which this run is not. If this ever drops near 12, that is a real
    result and this test should be the thing that makes you notice.
    """
    nis = float(result["nis_median_moving"])
    assert 25.0 < nis < 35.0, f"NIS median moved to {nis:.1f}; the fixture assumes ~29"


def test_swapped_wiring_does_not_reproduce_the_golden_run(
    pipeline: ModuleType, result: dict[str, Any]
) -> None:
    """Falsification: the fixture must be sensitive to the chip->link assignment.

    Re-runs the same pipeline with IMU_0 and IMU_1 exchanged -- the assignment
    ``conftest.py`` and ``IMUDecoder``'s docstring still assert, and the one the
    axis-map fit rejected at gyro rms 0.34 rad/s against 0.06-0.10.

    How it fails, precisely: the filter does not merely drift, it comes apart.
    NIS median goes from ~29 to ~5e4 (three orders of magnitude, against a
    target of 12), and the lag search saturates at its 0.6 s ceiling because no
    shift aligns the swapped gyros with the MuJoCo replay at all.
    """
    swapped = pipeline.run_pipeline(REPO_ROOT, layout=SWAPPED_LAYOUT)

    good_nis = float(result["nis_median_moving"])
    bad_nis = float(swapped["nis_median_moving"])
    assert bad_nis > 100 * good_nis, (
        f"swapped wiring gives NIS median {bad_nis:.0f} against {good_nis:.1f} correct; "
        "expected it to blow up by orders of magnitude"
    )
    assert float(swapped["lag_imu"]) == pytest.approx(0.6), (
        "the lag search should saturate at max_lag_s when the gyros do not align"
    )
    assert not np.allclose(swapped["p_ef"], result["p_ef"], rtol=RTOL, atol=0.0)
