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
from erp.sim.plant import (
    blind_variant,
    distancias,
    geoms_por_clase,
    load_model,
    sensor_layout,
    state_dim_of,
    warmup_to_rest,
)

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

    # No state and no sensor channels. Contacts are deliberately split: the
    # PLANT's floor is solid (`scene.xml`, so the `ee_collider` rests on it),
    # the FILTER's is not (`blind_variant`), which makes touching the floor
    # unmodelled dynamics. `test_blind_floor_does_not_collide` pins both halves.
    assert (model.nq, model.nv, model.na) == (4, 4, 0)
    assert model.nsensordata == 15

    # The blind variant is compiled from the same XML through MjSpec, and it is
    # the model the ghost is drawn from, so it has to agree.
    assert model_blind.ngeom == model.ngeom
    blind_mesh = [i for i in range(model_blind.ngeom) if int(model_blind.geom_type[i]) == mesh]
    assert blind_mesh == mesh_idx


def test_blind_floor_does_not_collide(xml_path, plant, blind) -> None:
    """The filter's floor is contact-free; the plant's floor and the opt-out are not.

    Checked by contact, not just by flags: the effector is pushed below z = 0 in
    every model, and only the solid-floor ones may report a contact with the
    floor. The `floor_collides=True` build is the broken variant -- it is
    `blind_variant` before the floor was removed -- and must fail the
    blind-model assertion.
    """
    import mujoco as mj

    def floor_contacts(model) -> int:
        data = mj.MjData(model)
        mj.mj_resetDataKeyframe(model, data, model.key("home").id)
        floor = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "floor")
        # link1 at its upper stop (1.57 rad) and link2 at 0 put the
        # `ee_collider` below the floor plane (found by a grid sweep over both).
        for name, q in (("link1", 1.57), ("link2", 0.0)):
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[jid]] = q
        mj.mj_forward(model, data)
        hits = 0
        for c in data.contact[: data.ncon]:
            if floor in (int(c.geom1), int(c.geom2)):
                hits += 1
        return hits

    plant_model, _ = plant
    blind_model, _ = blind
    modelled_floor, _ = blind_variant(xml_path, floor_collides=True)

    floor = mj.mj_name2id(blind_model, mj.mjtObj.mjOBJ_GEOM, "floor")
    assert int(blind_model.geom_contype[floor]) == 0
    assert int(blind_model.geom_conaffinity[floor]) == 0
    assert floor_contacts(blind_model) == 0

    # The halves that must still collide, or the check above proves nothing.
    assert floor_contacts(plant_model) > 0, "pose does not reach the floor"
    assert floor_contacts(modelled_floor) > 0
    assert blind_model.ngeom == plant_model.ngeom, "the floor geom must stay (index safety)"


FLOOR_POSE = {"link1": 1.57, "link2": 0.0}   # puts `ee_collider` ~67 mm into the floor


def _pose(model, data, pose):
    import mujoco as mj

    mj.mj_resetDataKeyframe(model, data, model.key("home").id)
    for name, q in pose.items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = q
    return np.array(data.qpos, dtype=np.float64)


def test_distancias_is_mujocos_own_contact_depth(plant) -> None:
    """Signed tool-floor distance equals the deepest `contact.dist` of mj_collision.

    MuJoCo is the oracle, not a hand-computed depth. The broken variant is an
    unsigned distance: it reports the same magnitude, so only the sign test
    below can tell it apart, and it must fail it.
    """
    import mujoco as mj

    model, _ = plant
    herr, obs = geoms_por_clase(model)

    oracle = mj.MjData(model)
    qpos = _pose(model, oracle, FLOOR_POSE)
    mj.mj_forward(model, oracle)
    floor = model.geom("floor").id
    deepest = min(float(c.dist) for c in oracle.contact[: oracle.ncon]
                  if floor in (int(c.geom1), int(c.geom2)) and
                  {int(c.geom1), int(c.geom2)} & set(herr.tolist()))

    dist, normal = distancias(model, mj.MjData(model), qpos, herr, obs)
    k = int(np.flatnonzero(obs == floor)[0])
    assert dist[k] == pytest.approx(deepest, abs=1e-9)
    assert dist[k] < -0.01, "the pose should be well inside the floor"
    # The broken variant, an unsigned distance, fails both checks above: it has
    # the right magnitude and the wrong sign.
    unsigned = abs(dist[k])
    assert unsigned != pytest.approx(deepest, abs=1e-9)
    assert not unsigned < -0.01
    # Out of the floor is +z, inside it as well as above it.
    np.testing.assert_allclose(normal[k], [0.0, 0.0, 1.0], atol=1e-6)
    # Close enough to be inside `dist_max` (50 mm), so a direction exists.
    near = _pose(model, mj.MjData(model), {"link1": 1.0, "link2": 0.0})   # ~20 mm up
    above, n_above = distancias(model, mj.MjData(model), near, herr, obs)
    assert 0.0 < above[k] < 0.05
    np.testing.assert_allclose(n_above[k], [0.0, 0.0, 1.0], atol=1e-6)


def test_geoms_por_clase_finds_tool_and_solids_without_names() -> None:
    """Collision class, not a name list: the stair wrapper's six blocks come in alone."""
    xml = resolve_repo_path(*XML[:-1], "MyPalletizer260_escaleras.xml")
    model, _ = load_model(xml)
    herr, obs = geoms_por_clase(model)
    assert [model.geom(int(g)).name for g in herr] == ["ee_collider"]
    names = {model.geom(int(g)).name for g in obs}
    assert names == {"floor", "esc_A1", "esc_A2", "esc_A3", "esc_B1", "esc_B2", "esc_B3"}


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


# -- the FSR model and its contact geometry ---------------------------------

FSR_XML = ("mechanical", "mujoco_assets", "MyPalletizer260", "MyPalletizer260_fsr.xml")
FSR_STAIRS_XML = ("mechanical", "mujoco_assets", "MyPalletizer260",
                  "MyPalletizer260_fsr_escaleras.xml")


def _x_of(q: list[float]) -> np.ndarray:
    """Blind-filter state (nx = 11) at joint angles q, `act` closing the tendon."""
    x = np.zeros(11)
    x[:3] = q
    x[3] = 1.5693 - q[1] - q[2]
    return x


def test_fsr_blind_model_steps_exactly_like_the_bare_one(blind) -> None:
    """The adapter is massless, so the filter may run on either XML.

    `MyPalletizer260_fsr.xml` is a hand-kept COPY of the base arm; this is also
    what catches the two drifting apart (a limit or a damping changed in one).
    """
    from erp.sim.mujoco import step_with_jacobian

    mb, db = blind
    mf, df = blind_variant(resolve_repo_path(*FSR_XML))
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = np.r_[warmup_to_rest(mb, db, Q_CMD_REST), Q_CMD_REST] + rng.normal(0, 0.05, 11)
        xa, Fa = step_with_jacobian(x, np.zeros(3), mb, db)
        xb, Fb = step_with_jacobian(x, np.zeros(3), mf, df)
        np.testing.assert_allclose(xb, xa, rtol=0, atol=1e-12)
        np.testing.assert_allclose(Fb, Fa, rtol=0, atol=1e-12)


@pytest.mark.parametrize("q", [[0.0, 1.1, 0.0], [0.0, 1.0, 0.2], [0.5, 1.1, -0.3],
                               [-0.4, 0.9, 0.4]])
def test_contact_jacobian_matches_finite_differences(q) -> None:
    """Analytic dd/dx (mj_jac along the normal) against central differences.

    Poses outside and inside solids, including one inside a stair block where J1
    has a gradient. d itself must equal `distancias` restricted to `fsr_tip`.
    """
    import mujoco as mj

    from erp.sim.dynamics import MujocoContactGeometry

    model, data = load_model(resolve_repo_path(*FSR_STAIRS_XML))
    geo = MujocoContactGeometry(model, data, nx=11)
    x = _x_of(q)
    d, D = geo.distance(x)
    fd = np.zeros(11)
    for k in range(11):
        e = np.zeros(11)
        e[k] = 1e-6
        fd[k] = (geo.distance(x + e)[0] - geo.distance(x - e)[0]) / 2e-6
    np.testing.assert_allclose(D, fd, atol=1e-8)
    assert not D[4:].any(), "velocity and activation do not move the distance"

    tip = np.array([model.geom("fsr_tip").id])
    _, obs = geoms_por_clase(model)
    ref, _ = distancias(model, mj.MjData(model), x[:4], tip, obs, dist_max=0.5)
    assert d == pytest.approx(float(ref.min()), abs=1e-12)


def test_fsr_force_and_the_geometry_agree_on_contact() -> None:
    """The ON signal (touch > 1 N) is true exactly when `fsr_tip` is in contact.

    Driven by the servo into the floor at several depths: clear of the floor the
    pad reads 0 N, and in contact it reads tens of newtons with d < 0 (the
    adapter's stiff solref keeps it under ~1 mm).
    """
    import mujoco as mj

    from erp.sensors.mujoco import rows_of
    from erp.sim.dynamics import MujocoContactGeometry

    model, data = load_model(resolve_repo_path(*FSR_XML))
    geo = MujocoContactGeometry(model, mj.MjData(model), nx=11)
    row = rows_of(model, "fsr_force")
    for cmd in (0.9, 1.0, 1.2, 1.57):
        mj.mj_resetDataKeyframe(model, data, model.key("home").id)
        for _ in range(1500):
            data.ctrl[:3] = [0.0, cmd, 0.0]
            mj.mj_step(model, data)
        force = float(data.sensordata[row][0])
        d, _ = geo.distance(np.r_[data.qpos, np.zeros(7)])
        assert (force > 1.0) == (d < 0.0), (cmd, force, d)
        if d < 0.0:
            assert -0.002 < d, "the stiff adapter contact stays within ~1 mm"


def test_contact_geometry_is_validated() -> None:
    from erp.sim.dynamics import MujocoContactGeometry

    model, data = load_model(resolve_repo_path(*FSR_XML))
    with pytest.raises(KeyError, match="fsr_tip"):
        bare, bare_data = load_model(resolve_repo_path(*XML))
        MujocoContactGeometry(bare, bare_data, nx=11)
    with pytest.raises(ValueError, match="nx"):
        MujocoContactGeometry(model, data, nx=4)
    with pytest.raises(ValueError, match="solidos"):
        blind_model, blind_data = blind_variant(resolve_repo_path(*FSR_XML))
        MujocoContactGeometry(blind_model, blind_data, nx=11)
