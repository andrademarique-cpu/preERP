"""Draw a second pose of the same model into a viewer scene, as a ghost.

What this is for: showing the EKF's estimate *over* the plant it is estimating,
so the error is visible as separation between two arms rather than as a gap
between two windows. MuJoCo supports one passive viewer per process, so a
second window would mean a second process and state IPC that drifts; a
``user_scn`` overlay needs neither.

**This is the fourth module in the package that touches the mujoco API**, after
``sim/mujoco.py``, ``sim/plant.py`` and ``sensors/mujoco.py``, and that list is
stated in CLAUDE.md as closed. The reason it is opened here: the rule's purpose
is that no ``Any`` escapes into the estimator under strict mypy, because mujoco
ships no ``py.typed``. This module returns ``int`` and writes into a scene
object; nothing it produces reaches a filter. Every value read back is wrapped
the way ``erp.sim.mujoco`` wraps its own. Do not use this as precedent for
putting *dynamics* anywhere but ``erp.sim.mujoco``.

English, matching the rest of :mod:`erp.viz`.
"""

from __future__ import annotations

import mujoco as mj
import numpy as np

__all__ = ["GHOST_RGBA", "draw_ghost", "ghost_geom_ids"]

GHOST_RGBA = (0.25, 0.70, 1.00, 0.35)
"""Translucent blue. Alpha 0.35 reads as a ghost without hiding the truth
behind it; at 0.6 the two arms are hard to tell apart, and below ~0.2 the
estimate disappears against a light floor."""


def ghost_geom_ids(model: mj.MjModel) -> list[int]:
    """Model geom ids the ghost draws: those on a body that can actually move.

    ``body_weldid == 0`` means the body is welded rigidly to the world, so a
    geom on it has the same pose in the estimate as in the truth *by
    construction* -- its ghost can never show error, and drawing it is overdraw
    that tints the real thing blue for nothing.

    Two kinds of geom are excluded on the MyPalletizer260, and both are
    deliberate:

    * the **scenery** (``mechanical/mujoco_assets/scene.xml``: the ground
      plane). Without this the ghost paints a translucent blue plane exactly
      coplanar with the real floor -- z-fighting and a blue-tinted ground.
    * **base_link**, which is bolted to the world. Dropping it is a small
      improvement rather than a side effect: it used to get a blue tint that
      carried no information, and the moving links read more clearly without it.

    25 of 27 geoms on this model. The rule is stated against the model rather
    than against a list of names, so it stays right for any arm and for whatever
    else ends up in the scene.
    """
    return [
        i
        for i in range(int(model.ngeom))
        if int(model.body_weldid[int(model.geom_bodyid[i])]) != 0
    ]


def draw_ghost(
    scene: mj.MjvScene,
    model: mj.MjModel,
    data: mj.MjData,
    *,
    rgba: tuple[float, float, float, float] = GHOST_RGBA,
    append: bool = False,
) -> int:
    """Add one translucent geom per *movable* model geom at ``data``'s pose.

    Call :func:`mujoco.mj_forward` on ``data`` first -- this reads
    ``geom_xpos``/``geom_xmat``, which are outputs of the forward pass, so
    without it the ghost renders at whatever pose was last computed.

    Parameters
    ----------
    scene:
        The scene to draw into, normally a passive viewer's ``user_scn``.
    model, data:
        The model and the state to draw. ``data`` is the *estimate's*
        ``MjData``, separate from the one the viewer renders.
    rgba:
        Colour and alpha of every ghost geom.
    append:
        ``False`` (default) resets the scene's geom count first, which is what
        a per-frame redraw wants. ``True`` adds to what is already there.

    Returns
    -------
    The number of geoms added, which is ``len(ghost_geom_ids(model))`` unless
    the scene ran out of room -- see the note on ``maxgeom`` below. Note that
    it is **not** ``model.ngeom``: geoms that cannot move are skipped, so a
    caller mapping scene slots back to model geoms must go through
    :func:`ghost_geom_ids` and not assume the two indices agree.

    Notes
    -----
    **``mjv_initGeom`` does not set ``dataid``, and the value it wants is not
    ``geom_dataid``.** Two separate facts, and getting the first right while
    getting the second wrong is what this docstring used to do.

    The first: ``mjv_initGeom`` initialises the fields it is handed and defaults
    the rest, so every geom comes back with ``dataid = -1``. For a mesh geom that
    is not cosmetic -- ``dataid`` is which mesh to draw, and -1 draws none. The
    MyPalletizer260 has 27 geoms of which 5 are meshes, and those 5 are the
    visible arm, the other 22 being the ground plane plus spheres and cylinders.
    Four of the five meshes are ghosted -- base_link's is skipped, see
    :func:`ghost_geom_ids`. Omit the ``dataid`` line and the ghost
    half-appears, as a scattering of decorations where the arm should be.

    The second: the render context stores **two entries per mesh**, so the index
    a scene geom carries is ``2 * geom_dataid`` and not ``geom_dataid`` --
    that is what ``mjv_updateScene`` itself writes, and it is the only correct
    source. Measured on this model: ``geom_dataid`` is ``[0, 1, 2, 3, 4]`` over
    mesh geoms ``[0, 1, 2, 10, 18]`` while MuJoCo's own scene gives them
    ``[0, 2, 4, 6, 8]``. Passing the undoubled value makes geom ``k`` draw mesh
    ``k // 2``, so every link after the base gets a *different link's* mesh --
    base_link's hull on ``rot``, rot's mesh on ``link1``, rot's hull on
    ``link2``, link1's mesh on ``act``. Each mesh carries its own compiled frame
    (``mesh_quat`` is ~120 deg off identity on four of the five), so the result
    reads as a ghost rotated the wrong way rather than as a wrong index, which is
    why it survived review once already.

    ``mat`` needs no such repair: ``mjv_initGeom`` copies the 9-vector it is
    given, and MuJoCo's own scene geoms match ``geom_xmat`` exactly. Both halves
    are pinned in ``test_viz_live.py`` against ``mjv_updateScene`` as the oracle.

    **``maxgeom`` is a hard ceiling.** A scene silently stops accepting geoms
    when it is full, so the ghost would be truncated rather than refused. 25
    drawn geoms against a default ``maxgeom`` of 1000 is not close, but the
    count is returned so a caller can assert on it.
    """
    if not append:
        scene.ngeom = 0
    mesh = int(mj.mjtGeom.mjGEOM_MESH)
    rgba_a = np.asarray(rgba, dtype=np.float32)
    added = 0
    for i in ghost_geom_ids(model):
        if scene.ngeom >= scene.maxgeom:
            break
        g = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(
            g,
            int(model.geom_type[i]),
            np.asarray(model.geom_size[i], dtype=np.float64),
            np.asarray(data.geom_xpos[i], dtype=np.float64),
            np.asarray(data.geom_xmat[i], dtype=np.float64).reshape(9),
            rgba_a,
        )
        if int(model.geom_type[i]) == mesh:
            # 2 * geom_dataid, not geom_dataid: the render context holds two
            # entries per mesh and this is what mjv_updateScene writes. See the
            # second half of the dataid note above -- the undoubled value draws
            # the wrong link's mesh and looks like a rotation bug.
            g.dataid = 2 * int(model.geom_dataid[i])
        # Not pickable: without this the ghost intercepts the mouse and you
        # end up dragging an estimate around instead of the arm.
        g.objtype = int(mj.mjtObj.mjOBJ_UNKNOWN)
        g.objid = -1
        scene.ngeom += 1
        added += 1
    return added
