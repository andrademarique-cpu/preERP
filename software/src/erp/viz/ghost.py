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

__all__ = ["GHOST_RGBA", "draw_ghost"]

GHOST_RGBA = (0.25, 0.70, 1.00, 0.35)
"""Translucent blue. Alpha 0.35 reads as a ghost without hiding the truth
behind it; at 0.6 the two arms are hard to tell apart, and below ~0.2 the
estimate disappears against a light floor."""


def draw_ghost(
    scene: mj.MjvScene,
    model: mj.MjModel,
    data: mj.MjData,
    *,
    rgba: tuple[float, float, float, float] = GHOST_RGBA,
    append: bool = False,
) -> int:
    """Add one translucent geom per model geom at ``data``'s current pose.

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
    The number of geoms added, which is ``model.ngeom`` unless the scene ran
    out of room -- see the note on ``maxgeom`` below.

    Notes
    -----
    **``mjv_initGeom`` does not set ``dataid``, and this model is mostly
    meshes.** It initialises the fields it is handed and defaults the rest, so
    every geom comes back with ``dataid = -1``. For a mesh geom that is not a
    cosmetic problem: ``dataid`` is which mesh to draw, and -1 draws none. The
    MyPalletizer260 has 26 geoms of which 5 are meshes -- and those 5 are the
    visible arm, the other 21 being spheres and cylinders. Omit the ``dataid``
    line and the ghost half-appears, as a scattering of decorations where the
    arm should be, which reads as a rendering glitch rather than a missing
    assignment. It is verified in ``test_viz_ghost.py``.

    **``maxgeom`` is a hard ceiling.** A scene silently stops accepting geoms
    when it is full, so the ghost would be truncated rather than refused. 26
    geoms against a default ``maxgeom`` of 1000 is not close, but the count is
    returned so a caller can assert on it.
    """
    if not append:
        scene.ngeom = 0
    mesh = int(mj.mjtGeom.mjGEOM_MESH)
    rgba_a = np.asarray(rgba, dtype=np.float32)
    added = 0
    for i in range(int(model.ngeom)):
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
            g.dataid = int(model.geom_dataid[i])
        # Not pickable: without this the ghost intercepts the mouse and you
        # end up dragging an estimate around instead of the arm.
        g.objtype = int(mj.mjtObj.mjOBJ_UNKNOWN)
        g.objid = -1
        scene.ngeom += 1
        added += 1
    return added
