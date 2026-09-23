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


# ------------------------------------------------------------------ scenery


def test_the_scenery_is_index_safe(plant, blind) -> None:
    """`scene.xml` must add the ground without renumbering anything.

    MuJoCo numbers geoms by BODY and the world is always body 0, so a floor hung
    straight off `<worldbody>` takes geom id 0 and shifts every geom of the arm
    up by one -- no matter where in the file it is written. That would move the
    mesh ids `erp.viz.ghost` looks up through `geom_dataid`, which is how a
    purely decorative change breaks the estimate's overlay.

    `scene.xml` avoids it by putting the floor inside its own static
    `<body name="scenery">`, created after the arm, so it collects the last body
    id and the last geom ids. This is the guard: move the include, or lift the
    geom out of that body, and the numbers below move while everything else
    still passes.
    """
    import mujoco as mj

    model, _ = plant
    model_blind, _ = blind

    floor = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "floor")
    assert floor != -1, "scene.xml should be included by the arm XML"

    # The property that matters is that the arm's geoms come FIRST and keep a
    # contiguous block starting at 0 -- not that the floor happens to be last.
    # Asserting `floor == ngeom - 1` would have been tighter and wrong: it fails
    # the first time someone adds a second decor body, which is a supported
    # thing to do and breaks nothing.
    def body_of(i: int) -> str:
        return str(mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])))

    arm = {"base_link", "rot", "link1", "link2", "act"}
    arm_geoms = [i for i in range(model.ngeom) if body_of(i) in arm]
    assert arm_geoms == list(range(len(arm_geoms))), "the arm's geoms must start at 0"
    assert floor > max(arm_geoms), "the scenery must come AFTER the arm, never before"

    # The arm's own geoms, unmoved. These indices are quoted in ghost.py and in
    # the /goal skill's notes, and `geom_dataid` is what picks the mesh.
    mesh = int(mj.mjtGeom.mjGEOM_MESH)
    mesh_idx = [i for i in range(model.ngeom) if int(model.geom_type[i]) == mesh]
    assert mesh_idx == [0, 1, 2, 10, 18]
    assert [int(model.geom_dataid[i]) for i in mesh_idx] == [0, 1, 2, 3, 4]

    # Visual only: no state, no sensor channels, no contacts.
    assert (model.nq, model.nv, model.na) == (4, 4, 0)
    assert model.nsensordata == 15
    assert int(model.geom_contype[floor]) == 0
    assert int(model.geom_conaffinity[floor]) == 0

    # The blind variant is compiled from the same XML through MjSpec, and it is
    # the model the ghost is drawn from, so it has to agree.
    assert model_blind.ngeom == model.ngeom
    blind_mesh = [i for i in range(model_blind.ngeom) if int(model_blind.geom_type[i]) == mesh]
    assert blind_mesh == mesh_idx


def test_the_scenery_does_not_move_the_physics(xml_path) -> None:
    """Stepping with the ground present must match stepping without it, exactly.

    The falsification for "purely visual". The scenery body is deleted from a
    second spec and the two are stepped side by side for the golden run's own
    length; anything but an exact zero means the frozen fixture is at risk and
    `make_golden_run.py --check` is the next thing to run.
    """
    import mujoco as mj

    with_scenery, _ = load_model(xml_path)
    data = mj.MjData(with_scenery)

    spec = mj.MjSpec.from_file(str(xml_path))
    spec.delete(spec.body("scenery"))
    bare = spec.compile()
    bare_data = mj.MjData(bare)
    assert bare.ngeom == with_scenery.ngeom - 1, "the scenery body was not removed"

    for m, d in ((with_scenery, data), (bare, bare_data)):
        mj.mj_resetDataKeyframe(m, d, m.key("home").id)

    cmd = np.array([0.4, 0.5, 0.3])
    worst = 0.0
    for _ in range(1848):        # the golden run's length
        data.ctrl[:3] = cmd
        bare_data.ctrl[:3] = cmd
        mj.mj_step(with_scenery, data)
        mj.mj_step(bare, bare_data)
        worst = max(worst, float(np.abs(data.sensordata - bare_data.sensordata).max()))
    assert worst == 0.0, f"the scenery moved the physics by {worst:.3e}"
