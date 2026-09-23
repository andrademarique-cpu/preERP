"""The two halves of the live demo's visuals that can be wrong numerically.

Same split as `test_viz.py`: what has arithmetic gets real assertions, what
needs a screen gets nothing. The strip buffer runs everywhere; the ghost needs
the model out of Git LFS, so it carries the `mujoco` marker.

The live pyqtgraph widgets in `erp.viz.live` are tested by neither, on purpose
-- they need a display and an event loop. What they *do* is push arrays from a
`RingBuffer` into a curve, and the buffer is pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from erp.viz.strips import RingBuffer

# ------------------------------------------------------------- RingBuffer


def test_an_unfilled_buffer_returns_what_it_was_given() -> None:
    buf = RingBuffer(capacity=10, width=2)
    for i in range(3):
        buf.append(float(i), [i, -i])
    t, Y = buf.view()
    np.testing.assert_array_equal(t, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(Y, [[0, 0], [1, -1], [2, -2]])
    assert len(buf) == 3
    assert not buf.full


def test_a_wrapped_buffer_returns_oldest_first_not_storage_order() -> None:
    """The one thing in this class that is easy to get wrong and hard to see.

    Once it wraps, the storage holds newest-at-the-front. Handing that to a
    plot draws a sawtooth: the trace jumps back in time at the write cursor,
    which looks like a data glitch rather than an ordering bug.
    """
    buf = RingBuffer(capacity=4, width=1)
    for i in range(6):
        buf.append(float(i), [i])
    t, Y = buf.view()
    assert buf.full
    np.testing.assert_array_equal(t, [2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(Y.ravel(), [2, 3, 4, 5])
    assert np.all(np.diff(t) > 0), "a wrapped view must still be monotone in time"


def test_the_view_is_a_copy_so_a_later_append_cannot_rewrite_a_drawn_point() -> None:
    buf = RingBuffer(capacity=3, width=1)
    for i in range(3):
        buf.append(float(i), [i])
    t, Y = buf.view()
    buf.append(99.0, [99.0])
    np.testing.assert_array_equal(t, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(Y.ravel(), [0, 1, 2])


def test_a_zero_width_buffer_holds_bare_timestamps() -> None:
    """Used to mark update instants on a strip, which have no value of their own."""
    buf = RingBuffer(capacity=3, width=0)
    buf.append(1.5)
    t, Y = buf.view()
    np.testing.assert_array_equal(t, [1.5])
    assert Y.shape == (1, 0)


def test_the_wrong_number_of_channels_raises() -> None:
    buf = RingBuffer(capacity=3, width=2)
    with pytest.raises(ValueError, match="width is 2"):
        buf.append(0.0, [1.0])


def test_clear_forgets_the_history_but_not_the_shape() -> None:
    buf = RingBuffer(capacity=3, width=2)
    buf.append(0.0, [1.0, 2.0])
    buf.clear()
    assert len(buf) == 0
    assert buf.view()[0].size == 0
    buf.append(5.0, [1.0, 2.0])
    assert len(buf) == 1


@pytest.mark.parametrize(("cap", "width"), [(0, 1), (-1, 1), (3, -1)])
def test_invalid_shapes_are_refused(cap: int, width: int) -> None:
    with pytest.raises(ValueError):
        RingBuffer(capacity=cap, width=width)


# ------------------------------------------------------------------- ghost


@pytest.mark.mujoco
def test_the_ghost_covers_every_movable_geom_and_follows_the_pose() -> None:
    import mujoco as mj

    from erp.io.paths import resolve_repo_path
    from erp.sim.plant import blind_variant, load_model
    from erp.viz.ghost import draw_ghost, ghost_geom_ids

    xml = resolve_repo_path(
        "mechanical", "mujoco_assets", "MyPalletizer260", "MyPalletizer260.xml"
    )
    model, data = load_model(xml)
    mb, db = blind_variant(xml)

    db.qpos[:] = data.qpos
    db.qpos[1] += 0.3
    mj.mj_forward(mb, db)

    scene = mj.MjvScene(model, maxgeom=200)
    added = draw_ghost(scene, mb, db)

    drawn = ghost_geom_ids(mb)
    assert added == len(drawn)
    assert scene.ngeom == len(drawn)
    assert added < mb.ngeom, "the immovable geoms are supposed to be skipped"
    # The ghost is posed from db, not from data: if draw_ghost read the wrong
    # MjData the overlay would track the truth exactly and show zero error --
    # a demo that always looks perfect.
    assert float(np.abs(data.geom_xpos - db.geom_xpos).max()) > 0.1


_POSE = (0.4, 0.5, 0.3, 1.5693)
"""A non-home `qpos` for the ghost tests. Away from the keyframe on all three
commandable joints, so a geom that failed to follow the pose shows up."""


def _renderer_geoms(model: object, data: object) -> dict[int, tuple[int, int, object, object]]:
    """`{model geom index: (type, dataid, pos, mat)}` as MuJoCo itself builds it.

    The oracle for everything `draw_ghost` writes. Hand-written expectations are
    what let the `dataid` bug through: the old test asserted the field equalled
    `geom_dataid`, which proved it had been *set* and never that it had been set
    to what the renderer reads.

    Keyed by `objid` rather than by position, because `mjv_updateScene` emits
    more geoms than the model has (30 against 27 here -- it adds decor) so scene
    order is not model order. Values are copied out: the `MjvGeom`s belong to the
    scene, which goes out of scope with this call.
    """
    import mujoco as mj

    scene = mj.MjvScene(model, maxgeom=2000)
    mj.mjv_updateScene(
        model, data, mj.MjvOption(), mj.MjvPerturb(), mj.MjvCamera(),
        mj.mjtCatBit.mjCAT_ALL, scene,
    )
    out: dict[int, tuple[int, int, object, object]] = {}
    for k in range(int(scene.ngeom)):
        g = scene.geoms[k]
        if int(g.objtype) != int(mj.mjtObj.mjOBJ_GEOM):
            continue
        i = int(g.objid)
        if 0 <= i < int(model.ngeom):  # type: ignore[attr-defined]
            out[i] = (
                int(g.type),
                int(g.dataid),
                np.asarray(g.pos, dtype=np.float64).copy(),
                np.asarray(g.mat, dtype=np.float64).reshape(9).copy(),
            )
    return out


@pytest.mark.mujoco
def test_mesh_geoms_get_the_renderer_dataid() -> None:
    """`dataid` must be `2 * geom_dataid`, which is what the renderer writes.

    Two traps in one field, and this test used to catch only the first.

    1. `mjv_initGeom` leaves every geom at `dataid = -1`, and -1 on a mesh draws
       nothing -- 5 mesh geoms of 27, and they are the only ones that look like
       a robot.
    2. The render context holds *two* entries per mesh, so the index is
       `2 * geom_dataid`. Passing the undoubled value draws a different link's
       mesh on every link but the base, which reads as a rotation bug rather
       than an indexing one.

    Both halves are falsified below: the bare `mjv_initGeom` leaves -1, and the
    undoubled value is asserted *not* to match the renderer.
    """
    import mujoco as mj

    from erp.io.paths import resolve_repo_path
    from erp.sim.plant import blind_variant, load_model
    from erp.viz.ghost import draw_ghost, ghost_geom_ids

    xml = resolve_repo_path(
        "mechanical", "mujoco_assets", "MyPalletizer260", "MyPalletizer260.xml"
    )
    model, data = load_model(xml)
    mb, db = blind_variant(xml)
    data.qpos[:] = _POSE
    db.qpos[:] = _POSE
    mj.mj_forward(model, data)
    mj.mj_forward(mb, db)

    # Scene slot != model geom id since the immovable geoms are skipped, so go
    # through `ghost_geom_ids` rather than assuming the two line up.
    mesh = int(mj.mjtGeom.mjGEOM_MESH)
    drawn = ghost_geom_ids(mb)
    mesh_idx = [i for i in drawn if int(mb.geom_type[i]) == mesh]
    assert mesh_idx, "the model is expected to be drawn with meshes"

    ref = _renderer_geoms(model, data)
    want = [ref[i][1] for i in mesh_idx]

    scene = mj.MjvScene(mb, maxgeom=200)
    draw_ghost(scene, mb, db)
    got = [int(scene.geoms[drawn.index(i)].dataid) for i in mesh_idx]
    assert got == want
    assert -1 not in got

    # The falsification this test was missing. `geom_dataid` is a real field
    # holding a real mesh id -- it is simply not the index a scene geom carries,
    # so an assertion against it passes while the ghost renders wrong.
    raw = [int(mb.geom_dataid[i]) for i in mesh_idx]
    assert want == [2 * v for v in raw]
    assert raw != want, "if these agree the doubling is untested, not unnecessary"

    # The original falsification: initGeom alone leaves them unusable.
    bare = mj.MjvScene(mb, maxgeom=200)
    for n, i in enumerate(mesh_idx):
        mj.mjv_initGeom(
            bare.geoms[n], mesh, mb.geom_size[i], db.geom_xpos[i],
            db.geom_xmat[i].reshape(9), np.ones(4, dtype=np.float32),
        )
    assert all(int(bare.geoms[n].dataid) == -1 for n in range(len(mesh_idx)))


@pytest.mark.mujoco
def test_a_ghost_on_the_truth_is_indistinguishable_from_the_rendered_arm() -> None:
    """Posed at the plant's own state, every ghost geom matches the renderer's.

    The end-to-end form of the test above, and the one that would have caught
    the `dataid` bug without anyone knowing to look at `dataid`: type, position,
    orientation and mesh id all have to agree, because when the estimate equals
    the truth the overlay is supposed to disappear into the arm.
    """
    import mujoco as mj

    from erp.io.paths import resolve_repo_path
    from erp.sim.plant import blind_variant, load_model
    from erp.viz.ghost import draw_ghost, ghost_geom_ids

    xml = resolve_repo_path(
        "mechanical", "mujoco_assets", "MyPalletizer260", "MyPalletizer260.xml"
    )
    model, data = load_model(xml)
    mb, db = blind_variant(xml)
    data.qpos[:] = _POSE
    db.qpos[:] = _POSE
    mj.mj_forward(model, data)
    mj.mj_forward(mb, db)

    ref = _renderer_geoms(model, data)
    scene = mj.MjvScene(mb, maxgeom=200)
    draw_ghost(scene, mb, db)

    # Only the geoms the renderer actually emitted: a default MjvOption filters
    # by geom group, and a geom it dropped is not this function's business.
    assert ref, "the renderer emitted no geoms to compare against"
    compared = 0
    for slot, i in enumerate(ghost_geom_ids(mb)):
        if i not in ref:
            continue
        gtype, dataid, pos, mat = ref[i]
        compared += 1
        g = scene.geoms[slot]
        assert int(g.type) == gtype, f"geom {i}: type"
        assert int(g.dataid) == dataid, f"geom {i}: dataid"
        np.testing.assert_allclose(
            np.asarray(g.pos, dtype=np.float64), pos, atol=1e-6,
            err_msg=f"geom {i}: position",
        )
        np.testing.assert_allclose(
            np.asarray(g.mat, dtype=np.float64).reshape(9), mat, atol=1e-6,
            err_msg=f"geom {i}: orientation",
        )
    assert compared > 0, "nothing was compared -- the id mapping is wrong"


@pytest.mark.mujoco
def test_ghost_geom_ids_skips_exactly_what_cannot_move() -> None:
    """The rule, checked against the model rather than restated.

    `ghost_geom_ids` keeps geoms whose body is not welded to the world. On this
    arm that has to mean: every geom of `rot`, `link1`, `link2` and `act` in,
    and the ground plane plus `base_link` out. Naming them here is the point --
    the implementation says `body_weldid != 0`, and this says what that buys.
    """
    import mujoco as mj

    from erp.io.paths import resolve_repo_path
    from erp.sim.plant import blind_variant
    from erp.viz.ghost import ghost_geom_ids

    xml = resolve_repo_path(
        "mechanical", "mujoco_assets", "MyPalletizer260", "MyPalletizer260.xml"
    )
    mb, _ = blind_variant(xml)
    drawn = set(ghost_geom_ids(mb))

    def body_of(i: int) -> str:
        return str(mj.mj_id2name(mb, mj.mjtObj.mjOBJ_BODY, int(mb.geom_bodyid[i])))

    kept = {body_of(i) for i in drawn}
    dropped = {body_of(i) for i in range(mb.ngeom) if i not in drawn}
    assert kept == {"rot", "link1", "link2", "act"}
    assert dropped == {"base_link", "scenery"}

    # The floor specifically: without this the ghost paints a translucent plane
    # exactly on top of the real one.
    floor = mj.mj_name2id(mb, mj.mjtObj.mjOBJ_GEOM, "floor")
    assert floor != -1, "scene.xml should have been included"
    assert floor not in drawn


@pytest.mark.mujoco
def test_a_full_scene_truncates_rather_than_overruns() -> None:
    import mujoco as mj

    from erp.io.paths import resolve_repo_path
    from erp.sim.plant import blind_variant
    from erp.viz.ghost import draw_ghost

    xml = resolve_repo_path(
        "mechanical", "mujoco_assets", "MyPalletizer260", "MyPalletizer260.xml"
    )
    mb, db = blind_variant(xml)
    mj.mj_forward(mb, db)
    scene = mj.MjvScene(mb, maxgeom=5)
    assert draw_ghost(scene, mb, db) == 5
