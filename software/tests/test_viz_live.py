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
def test_the_ghost_covers_every_geom_and_follows_the_pose() -> None:
    import mujoco as mj

    from erp.io.paths import resolve_repo_path
    from erp.sim.plant import blind_variant, load_model
    from erp.viz.ghost import draw_ghost

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

    assert added == mb.ngeom
    assert scene.ngeom == mb.ngeom
    # The ghost is posed from db, not from data: if draw_ghost read the wrong
    # MjData the overlay would track the truth exactly and show zero error --
    # a demo that always looks perfect.
    assert float(np.abs(data.geom_xpos - db.geom_xpos).max()) > 0.1


@pytest.mark.mujoco
def test_mesh_geoms_keep_their_dataid() -> None:
    """The falsification for the trap in `draw_ghost`'s docstring.

    `mjv_initGeom` leaves every geom at `dataid = -1`, and -1 on a mesh draws
    nothing. This asserts both halves: that the helper repairs it, and that
    skipping the repair really does lose the arm -- 5 mesh geoms of 26, and
    they are the only ones that look like a robot.
    """
    import mujoco as mj

    from erp.io.paths import resolve_repo_path
    from erp.sim.plant import blind_variant
    from erp.viz.ghost import draw_ghost

    xml = resolve_repo_path(
        "mechanical", "mujoco_assets", "MyPalletizer260", "MyPalletizer260.xml"
    )
    mb, db = blind_variant(xml)
    mj.mj_forward(mb, db)
    mesh = int(mj.mjtGeom.mjGEOM_MESH)
    mesh_idx = [i for i in range(mb.ngeom) if int(mb.geom_type[i]) == mesh]
    assert mesh_idx, "the model is expected to be drawn with meshes"

    scene = mj.MjvScene(mb, maxgeom=200)
    draw_ghost(scene, mb, db)
    got = [int(scene.geoms[i].dataid) for i in mesh_idx]
    assert got == [int(mb.geom_dataid[i]) for i in mesh_idx]
    assert -1 not in got

    # Without the repair: initGeom alone leaves them unusable.
    bare = mj.MjvScene(mb, maxgeom=200)
    for n, i in enumerate(mesh_idx):
        mj.mjv_initGeom(
            bare.geoms[n], mesh, mb.geom_size[i], db.geom_xpos[i],
            db.geom_xmat[i].reshape(9), np.ones(4, dtype=np.float32),
        )
    assert all(int(bare.geoms[n].dataid) == -1 for n in range(len(mesh_idx)))


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
