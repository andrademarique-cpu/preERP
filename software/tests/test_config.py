"""``config/estimation.yaml`` is now definition, not documentation (ADR-0002 P7).

The phase's obligation from ADR-0002 section 6 is *"config round-trips to
``IMUDecoder``; rows contiguous"*, and its falsification is *"the swapped wiring
must produce a worse fit"* -- which ``test_golden_run`` already holds, over the
hardcoded layout in ``scripts/make_golden_run.py``. What that leaves for this
module is the link P7 actually adds: proving the YAML and the pipeline's dict
are the **same wiring**, so the fixture's evidence transfers to the config.
:func:`test_config_layout_matches_the_golden_pipeline` is that link, and it is
what fails if someone edits the YAML.

Every validation test below is paired with the malformed variant it rejects. A
loader that only ever sees a good file proves nothing about a bad one, and a bad
config is the failure this module exists to convert from "the filter answers
worse" into "the load raises".
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import yaml
from conftest import LAYOUT as CONFTEST_LAYOUT

from erp.io.config import (
    ConfigError,
    build_decoder,
    default_config_path,
    load_config,
)


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("no pyproject.toml above this file")


REPO_ROOT = _repo_root()
SCRIPT_PATH = REPO_ROOT / "scripts" / "make_golden_run.py"
XML_PATH = REPO_ROOT / "mechanical" / "mujoco_assets" / "MyPalletizer260" / "MyPalletizer260.xml"

# The order the XML's <sensor> block declares, which is what makes the rows
# contiguous. Stated here so a reordering of the YAML fails loudly.
EXPECTED_BLOCKS = ("link1_acc", "link2_acc", "link1_gyro", "link2_gyro")


@pytest.fixture(scope="module")
def cfg() -> Any:
    return load_config()


@pytest.fixture
def raw_yaml() -> dict[str, Any]:
    """The repo config as plain dicts, for tests that break one field."""
    with default_config_path().open("r", encoding="utf-8") as fh:
        return dict(yaml.safe_load(fh))


def _write(tmp_path: Path, doc: dict[str, Any]) -> Path:
    path = tmp_path / "estimation.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


# ---------------------------------------------------------------- the real file


def test_repo_config_loads_with_the_values_the_arm_runs_on(cfg: Any) -> None:
    assert cfg.source == default_config_path()
    assert cfg.time.base == "monotonic_host"
    assert cfg.time.buffer_horizon_s == pytest.approx(0.050)
    # 25 Hz, not the finger stack's 200: this is what the notebook streams at
    # and what the golden run was generated with.
    assert cfg.inputs.rate_hz == pytest.approx(25.0)
    assert cfg.inputs.hold == "zoh"
    assert cfg.imu.fmt == "kv"
    assert cfg.imu.clock == "arrival"
    assert cfg.imu.rate_hz == pytest.approx(20.0)
    assert cfg.imu.arm_lag_s == pytest.approx(0.395)
    assert cfg.imu.dim == 12


def test_layout_order_is_the_xml_sensor_order(cfg: Any) -> None:
    """Iteration order is load-bearing: it is what ``rows_of`` is indexed by."""
    assert tuple(cfg.imu.layout) == EXPECTED_BLOCKS


def test_the_wiring_is_imu0_on_link1(cfg: Any) -> None:
    """The assignment the fit chose, against which two copies disagreed until P7."""
    assert cfg.imu.layout["link1_acc"] == ("IMU_0.ax", "IMU_0.ay", "IMU_0.az")
    assert cfg.imu.layout["link2_acc"] == ("IMU_1.ax", "IMU_1.ay", "IMU_1.az")


def test_conftest_fixture_agrees_with_the_config(cfg: Any) -> None:
    """Two of ADR-0002 3.3's four copies, now checked against each other.

    This is the drift check that could not exist before a loader did. The other
    two copies are ``scripts/make_golden_run.py`` (see the mujoco-marked test
    below) and ``IMUDecoder``'s docstring, which is prose and cannot be asserted.
    """
    assert {k: tuple(v) for k, v in CONFTEST_LAYOUT.items()} == dict(cfg.imu.layout)


def test_config_is_read_only(cfg: Any) -> None:
    """A caller mutating the layout would rewire every decoder built afterwards."""
    with pytest.raises(TypeError):
        cfg.imu.layout["link1_acc"] = ("IMU_9.ax",)  # type: ignore[index]
    with pytest.raises(ValueError, match="read-only"):
        cfg.imu.axis_maps["link1_acc"][0, 0] = 99.0
    with pytest.raises(AttributeError):
        cfg.imu.port = "COM9"


# ------------------------------------------------------- the round-trip itself


def test_build_decoder_round_trips_the_config(cfg: Any) -> None:
    """P7's obligation: the config reproduces the decoder, channel for channel."""
    decoder = build_decoder(cfg.imu)
    assert decoder.keys == cfg.imu.keys
    assert tuple(name for name, _ in decoder.blocks) == EXPECTED_BLOCKS
    assert decoder.dim == cfg.imu.dim
    assert decoder.column_names()[:2] == ["link1_acc_x", "link1_acc_y"]

    rng = np.random.default_rng(0)
    raw = rng.normal(size=len(cfg.imu.keys))
    position = {k: i for i, k in enumerate(cfg.imu.keys)}

    expected = np.concatenate([
        np.asarray(cfg.imu.axis_maps[sensor]) @ [raw[position[c]] for c in channels]
        for sensor, channels in cfg.imu.layout.items()
    ])
    np.testing.assert_allclose(decoder.apply(raw), expected, rtol=0, atol=1e-15)
    # to_raw is the inverse, which is what lets a raw log be re-decoded after
    # the layout or the axis maps are corrected.
    np.testing.assert_allclose(decoder.to_raw(decoder.apply(raw)), raw, atol=1e-12)


def test_build_decoder_leaves_the_bias_at_zero(cfg: Any) -> None:
    """Bias is calibration, not configuration -- it is not in the file to load."""
    np.testing.assert_array_equal(build_decoder(cfg.imu).b, np.zeros(cfg.imu.dim))


# ------------------------------------------------ malformed configs are refused


def test_layout_channel_not_in_keys_is_refused(tmp_path: Path, raw_yaml: Any) -> None:
    raw_yaml["sensors"]["palletizer_imu"]["layout"]["link1_acc"] = ["IMU_9.ax", "a", "b"]
    with pytest.raises(ConfigError, match=r"layout\.link1_acc: channels not in keys"):
        load_config(_write(tmp_path, raw_yaml))


def test_a_channel_in_two_blocks_is_refused(tmp_path: Path, raw_yaml: Any) -> None:
    """One device channel feeds exactly one MuJoCo sensor.

    ``IMUDecoder`` accepts the duplicate happily and emits a ``z`` in which the
    same reading appears twice, which the filter then treats as two independent
    measurements of different quantities.
    """
    imu = raw_yaml["sensors"]["palletizer_imu"]
    imu["layout"]["link2_acc"] = list(imu["layout"]["link1_acc"])
    with pytest.raises(ConfigError, match="appear in more than one block"):
        load_config(_write(tmp_path, raw_yaml))


def test_duplicate_keys_are_refused(tmp_path: Path, raw_yaml: Any) -> None:
    imu = raw_yaml["sensors"]["palletizer_imu"]
    imu["keys"] = [imu["keys"][0], *imu["keys"][1:-1], imu["keys"][0]]
    with pytest.raises(ConfigError, match="keys: duplicated"):
        load_config(_write(tmp_path, raw_yaml))


def test_a_non_orthogonal_axis_map_is_refused(tmp_path: Path, raw_yaml: Any) -> None:
    """The falsification of the orthogonality check.

    1% off a rotation is not a shape error and not a unit error: the reading is
    still a 3-vector in m/s^2, it is simply 1% too large on every axis, for the
    whole run, in a way no ``R`` describes. Without the check it loads.
    """
    imu = raw_yaml["sensors"]["palletizer_imu"]
    imu["axis_maps"]["link1_acc"] = [[0, 1.01, 0], [-1.01, 0, 0], [0, 0, 1.01]]
    with pytest.raises(ConfigError, match="not orthogonal"):
        load_config(_write(tmp_path, raw_yaml))


def test_a_rotation_that_is_not_a_permutation_is_still_accepted(
    tmp_path: Path, raw_yaml: Any
) -> None:
    """Paired with the test above: the check is orthogonality, not tidiness.

    A 30 deg rotation about z is a legitimate mounting and must load.
    """
    c, s = float(np.cos(np.pi / 6)), float(np.sin(np.pi / 6))
    imu = raw_yaml["sensors"]["palletizer_imu"]
    imu["axis_maps"]["link1_acc"] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    cfg = load_config(_write(tmp_path, raw_yaml))
    assert cfg.imu.axis_maps["link1_acc"][0, 0] == pytest.approx(c)


def test_buffer_horizon_below_the_sensor_latency_is_refused(
    tmp_path: Path, raw_yaml: Any
) -> None:
    """The one cross-section invariant the file states in prose.

    A horizon shorter than the latency releases each measurement before it
    could have arrived, so the runner discards exactly what the horizon exists
    to keep -- and the loss is intermittent, which is what makes it survive a
    demo.
    """
    raw_yaml["time"]["buffer_horizon_s"] = 0.001
    raw_yaml["sensors"]["palletizer_imu"]["latency_s"] = 0.020
    with pytest.raises(ConfigError, match="below"):
        load_config(_write(tmp_path, raw_yaml))


def test_an_unknown_line_format_is_refused(tmp_path: Path, raw_yaml: Any) -> None:
    raw_yaml["sensors"]["palletizer_imu"]["format"] = "json"
    with pytest.raises(ConfigError, match=r"format: expected one of \['kv', 'csv'\]"):
        load_config(_write(tmp_path, raw_yaml))


def test_a_bool_where_a_number_belongs_is_refused(tmp_path: Path, raw_yaml: Any) -> None:
    """``bool`` is an ``int`` subclass, so ``rate_hz: true`` would load as 1 Hz."""
    raw_yaml["sensors"]["palletizer_imu"]["rate_hz"] = True
    with pytest.raises(ConfigError, match="expected a number"):
        load_config(_write(tmp_path, raw_yaml))


def test_a_missing_key_names_its_full_path(tmp_path: Path, raw_yaml: Any) -> None:
    del raw_yaml["sensors"]["palletizer_imu"]["sig_gyro"]
    with pytest.raises(ConfigError, match=r"sensors\.palletizer_imu\.sig_gyro: missing"):
        load_config(_write(tmp_path, raw_yaml))


def test_an_unknown_sensor_key_lists_what_is_there(tmp_path: Path, raw_yaml: Any) -> None:
    with pytest.raises(ConfigError, match=r"sensors\.finger_imu: missing \(have"):
        load_config(_write(tmp_path, raw_yaml), imu_key="finger_imu")


def test_a_missing_file_is_not_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_the_stripped_finger_sections_are_gone(raw_yaml: Any) -> None:
    """P7 removed them; ADR-0001 appendix A is where their values went.

    Asserted rather than assumed because "strip the dead sections" is half the
    phase, and a half-applied strip leaves the file looking authoritative about
    a stack that no longer exists.
    """
    assert "joints" not in raw_yaml
    assert "estimators" not in raw_yaml
    assert set(raw_yaml["sensors"]) == {"palletizer_imu"}


# ------------------------------------------------------- the import boundary


def test_importing_erp_io_does_not_import_erp_sensors() -> None:
    """``erp.io`` must stay a ``core``-only import.

    ``erp.fusion`` may depend on ``erp.io`` and may never reach ``erp.sensors``.
    Since ``erp.io.config`` imports ``IMUDecoder``, re-exporting it from
    ``erp/io/__init__.py`` would put a hardware-facing package one hop from the
    filter -- the indirect reach-through the CI grep cannot see. A subprocess,
    because by the time this suite runs everything is already imported.
    """
    code = (
        "import sys; import erp.io; "
        "leaked = [m for m in sys.modules if m.startswith('erp.sensors')]; "
        "print(leaked)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]", f"erp.io pulled in {out.stdout.strip()}"


def test_erp_io_config_is_importable_by_name() -> None:
    """The other half: not re-exporting must not mean not reachable."""
    code = "from erp.io.config import build_decoder, load_config; print('ok')"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "ok"


# --------------------------------------------------------- against the model


def _load_pipeline() -> ModuleType:
    """Import ``scripts/make_golden_run.py`` by path; ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("erp_golden_pipeline_cfg", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.mujoco
def test_rows_from_the_config_layout_are_contiguous(cfg: Any) -> None:
    """P7's second obligation, and the reason the YAML's order matters."""
    mj = pytest.importorskip("mujoco")
    if not XML_PATH.is_file():
        pytest.skip(f"{XML_PATH.name} is missing (Git LFS?)")
    from erp.sensors.mujoco import rows_of

    model = mj.MjModel.from_xml_path(str(XML_PATH))
    rows = rows_of(model, *cfg.imu.layout)
    np.testing.assert_array_equal(rows, np.arange(cfg.imu.dim))


@pytest.mark.mujoco
def test_a_reordered_layout_is_not_contiguous(cfg: Any) -> None:
    """The falsification of the test above.

    Contiguity is a property of the *order* the YAML lists the blocks in, not
    of the block names. Reordered, ``rows_of`` still returns twelve valid
    indices -- so ``z`` and ``rows`` still line up in length, and every
    downstream shape check still passes while the channels are permuted.
    """
    mj = pytest.importorskip("mujoco")
    if not XML_PATH.is_file():
        pytest.skip(f"{XML_PATH.name} is missing (Git LFS?)")
    from erp.sensors.mujoco import rows_of

    model = mj.MjModel.from_xml_path(str(XML_PATH))
    reordered = ("link1_gyro", "link1_acc", "link2_acc", "link2_gyro")
    rows = rows_of(model, *reordered)
    assert sorted(rows) == list(range(cfg.imu.dim))
    assert not np.array_equal(rows, np.arange(cfg.imu.dim))


@pytest.mark.mujoco
def test_config_layout_matches_the_golden_pipeline(cfg: Any) -> None:
    """The last two of the four copies, and the point of the whole phase.

    ``scripts/make_golden_run.py`` carries its own ``IMU_LAYOUT`` and
    ``IMU_AXIS_MAPS``, and it is what generated ``golden_ekf_run.npz``. If they
    and the YAML ever disagree, the fixture's evidence stops applying to the
    config -- and the config is what everything else is now built from. This
    test is what makes editing the YAML alone fail.
    """
    pytest.importorskip("mujoco")
    if not SCRIPT_PATH.is_file():
        pytest.skip(f"{SCRIPT_PATH.name} is missing")
    pipeline = _load_pipeline()

    assert {k: tuple(v) for k, v in pipeline.IMU_LAYOUT.items()} == dict(cfg.imu.layout)
    assert tuple(pipeline.IMU_KEYS) == cfg.imu.keys
    for sensor, M in pipeline.IMU_AXIS_MAPS.items():
        np.testing.assert_array_equal(np.asarray(M, dtype=float), cfg.imu.axis_maps[sensor])
    assert pipeline.SIG_ACC == pytest.approx(cfg.imu.sig_acc)
    assert pipeline.SIG_GYRO == pytest.approx(cfg.imu.sig_gyro)
