"""``erp.sim.plant``: the blind model, the servoed equilibrium, the sensor layout.

This is ADR-0002 phase P1's obligation -- "``blind_variant`` yields ``na = 3``,
``nx = 11``, and the same ``nsensordata`` as the plant" -- with the
falsification the ADR asks for next to it: the un-edited plant model must fail
that same contract.

Marked ``mujoco``: unlike the rest of the suite this needs mujoco *and* the
Git-LFS assets (``MyPalletizer260.xml`` pulls STL meshes), so it cannot run off
a bare checkout. ``pytest -m "not mujoco"`` keeps the fast loop fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from erp.io.paths import resolve_repo_path
from erp.sensors.mujoco import rows_of
from erp.sim.mujoco import h_dyn, state_dim
from erp.sim.plant import blind_variant, load_model, sensor_layout, state_dim_of, warmup_to_rest

pytestmark = pytest.mark.mujoco

XML = ("mechanical", "mujoco_assets", "MyPalletizer260", "MyPalletizer260.xml")
IMU_SENSORS = ("link1_acc", "link2_acc", "link1_gyro", "link2_gyro")

# The trajectory's first setpoint, which is what the filter is initialised at.
Q_CMD_REST = np.zeros(3)


@pytest.fixture(scope="module")
def xml_path():
    return resolve_repo_path(*XML)


@pytest.fixture(scope="module")
def plant(xml_path):
    return load_model(xml_path)


@pytest.fixture(scope="module")
def blind(xml_path):
    return blind_variant(xml_path)


# -- the blind variant ------------------------------------------------------


def test_blind_variant_turns_the_servos_into_activations(plant, blind) -> None:
    """``na`` 0 -> 3 and ``nx`` 8 -> 11, sensor block untouched.

    The three activations are what the filter estimates in place of the control
    it never receives. Without them, ``ctrl = 0`` makes every position servo
    pull toward zero and the filter "knows" the arm is returning home.
    """
    model, _ = plant
    model_blind, _ = blind
    assert model_blind.na == 3
    assert state_dim_of(model_blind) == 11
    assert state_dim(model_blind) == state_dim_of(model_blind)
    assert model_blind.nsensordata == model.nsensordata


def test_the_plant_model_fails_the_same_contract(plant) -> None:
    """Falsification: the un-edited model has no activations at all.

    Stated precisely, because the difference is the whole point of the phase:
    the plant has ``na = 0`` and ``nx = 8``, so an ``na == 3`` assert against it
    fails outright rather than merely giving worse numbers. A test that passed
    against both models would not be checking that ``blind_variant`` did
    anything.
    """
    model, _ = plant
    assert model.na == 0
    assert state_dim_of(model) == 8
    assert model.nu == 3, "still three actuators -- only their dyntype changed"


def test_blind_variant_preserves_gains_and_sensors(plant, blind) -> None:
    """Only ``dyntype`` changes; ``h(x)`` must stay comparable to the plant log."""
    model, _ = plant
    model_blind, _ = blind
    assert sensor_layout(model_blind) == sensor_layout(model)
    assert np.allclose(model_blind.actuator_gainprm, model.actuator_gainprm)
    assert np.allclose(model_blind.actuator_forcerange, model.actuator_forcerange)


# -- the sensor layout ------------------------------------------------------


def test_sensor_layout_matches_rows_of(plant) -> None:
    """The two ways of addressing ``sensordata`` must agree.

    ``sensor_layout`` gives a slice per sensor, ``rows_of`` the concatenated
    indices; the filter uses the second and the plots the first.
    """
    model, _ = plant
    layout = sensor_layout(model)
    for name in IMU_SENSORS:
        sl = layout[name]
        assert np.array_equal(rows_of(model, name), np.arange(sl.start, sl.stop))


def test_sensor_layout_is_the_documented_one(plant) -> None:
    """12 IMU channels then ``efector_pos``, which is never measured.

    Written out because ``rows 12..14`` being output-only is load-bearing:
    ``make_R`` assigns those channels variance 1.0 by substring fallback, so
    feeding them to an update would quietly use sigma = 1 m.
    """
    model, _ = plant
    layout = sensor_layout(model)
    assert list(layout) == [*IMU_SENSORS, "efector_pos"]
    assert layout["link1_acc"] == slice(0, 3)
    assert layout["link2_gyro"] == slice(9, 12)
    assert layout["efector_pos"] == slice(12, 15)


def test_load_model_names_the_keyframes_it_has(xml_path) -> None:
    with pytest.raises(KeyError, match="home"):
        load_model(xml_path, keyframe="nonexistent")


# -- the servoed equilibrium ------------------------------------------------


def test_warmup_to_rest_settles_under_gravity(plant, blind) -> None:
    """At rest both accelerometers read g, and the arm has sunk a known amount.

    The sink is the measured 0.09 deg on link1 and 0.20 deg on link2 -- small,
    but it is the difference between an accelerometer reading 9.81 and one
    reading 4.14 (see the falsification below).
    """
    model, data = plant
    model_blind, data_blind = blind
    x_rest = np.r_[warmup_to_rest(model, data, Q_CMD_REST), Q_CMD_REST]
    assert x_rest.shape == (11,)

    layout = sensor_layout(model_blind)
    h = h_dyn(x_rest, np.zeros(model_blind.nu), model_blind, data_blind)
    assert np.linalg.norm(h[layout["link1_acc"]]) == pytest.approx(9.81, abs=0.02)
    assert np.linalg.norm(h[layout["link2_acc"]]) == pytest.approx(9.81, abs=0.02)
    assert np.linalg.norm(h[layout["link1_gyro"]]) < 1e-3, "at rest the gyros read zero"

    sink_deg = np.rad2deg(x_rest[1:3] - Q_CMD_REST[1:3])
    assert sink_deg == pytest.approx([0.0935, 0.2015], abs=5e-3)


def test_skipping_the_warmup_misses_gravity_by_more_than_5g(blind) -> None:
    """Falsification: initialising at the keyframe instead of the equilibrium.

    Precisely how it fails: with ``act = 0`` and no warmup the servos make no
    force, so ``h(x0)`` gives ``link2_acc_x`` 4.14 m/s^2 where the equilibrium
    gives 9.81 and the real IMU measured 9.65 at rest. A 5.7 m/s^2 constant
    offset on an accelerometer channel whose sigma is 0.05 is a 114-sigma bias
    -- the kind no ``Q`` or ``R`` tuning repairs, only a correct ``x0``.

    Note this is *not* the same as zeroing the whole state: ``qpos[3]`` is the
    passive parallelogram joint, and zeroing it breaks the tendon equality and
    gives 345 m/s^2 instead. The keyframe value is kept here so the failure
    isolates the missing warmup.
    """
    model_blind, data_blind = blind
    layout = sensor_layout(model_blind)

    x_no_warmup = np.zeros(state_dim_of(model_blind))
    x_no_warmup[3] = 1.5693      # the `home` value for the passive joint
    h = h_dyn(x_no_warmup, np.zeros(model_blind.nu), model_blind, data_blind)

    acc_x = float(h[layout["link2_acc"]][0])
    assert acc_x == pytest.approx(4.144, abs=0.01)
    assert abs(acc_x - 9.81) > 5.0, "the whole reason x0 is an equilibrium, not a pose"
