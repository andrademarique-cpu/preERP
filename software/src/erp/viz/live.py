"""Real-time strip charts for the interactive demo. pyqtgraph, not matplotlib.

Why a second backend rather than reusing :mod:`erp.viz.figures`: matplotlib
redraws the whole figure, and at a 30 Hz strip it does not keep up. The arm
then stutters in the viewer -- because the render tick is waiting on the draw
-- which looks like a physics or a filter problem and is neither.
``pyproject.toml`` still carries a comment about pyqtgraph having been a
dependency here once, for a finger viewer that no longer exists.

**One variable per plot, three windows.** The first version put two IMU
channels and three joints on shared axes, all joints in one colour, and the
curves could not be told apart. Now: one window per link with a plot per
sensor axis (plant / measured / ``h(x_hat)`` on each), and one window of
joint angles, one colour per joint -- plant solid, estimate dashed -- above
the NIS strip. Each sensor *kind* (``kinds``) is a column in the link window:
the simulated demo shows the accelerometer alone, the hardware demo
(``scripts/demo_teleop_real.py``) adds the gyro beside it, because on the real
arm the gyro is where the motion and the transport lag read most clearly.

**A fourth window, opt-in: FSR / contact** (``contact=``). Fingertip force
over the dome's signed distance to the nearest solid, plant against estimate,
with every contact correction ``EKFFSR`` applied marked where the estimate was
when it happened. Its buffers are :class:`~erp.viz.strips.ContactTrace`, which
is pure numpy and tested; this module only draws them. No mujoco here either:
the caller computes ``d(x)`` through its ``ContactGeometry`` and pushes numbers.

**The y axes are fixed, deliberately.** Auto-ranging zooms onto whatever is on
screen, so an arm at rest fills the plot with 0.05 m/s^2 of sensor noise and
looks like it is shaking. Fixed limits keep the scale of *motion*. The
defaults were measured on a scripted jog driving every joint end to end
(``scripts/demo_teleop.py``'s model, 2026-09-23): the plant's accelerations
stay within about +-15 m/s^2 and ``h(x_hat)`` reaches -30 on ``link2_acc_x``
for a moment during a reversal, so +-20 shows every free motion and clips
only that transient. With a prop, contact spikes reach +-60 and clip at the
edge, which still shows where the impact happens. The joint limits cover
``actuator_ctrlrange`` (rot +-2.79, link1 0..1.57, link2 -1.6055..1.04 rad;
link2 reaches negative angles since 2026-10-21).

**Nothing here is subclassed from Qt, deliberately.** PyQt5 ships ``.pyi``
stubs and pyqtgraph does not, and CI installs neither, so a Qt base class is
strictly typed locally and ``Any`` in CI -- and a ``type: ignore`` needed on
one leg is flagged as unused on the other. Composition sidesteps it: the
widgets are held as attributes, the same resolution :mod:`erp.viz.theme` uses
for ``_RC``. It also keeps the keyboard out of here, which belongs to the
MuJoCo viewer's ``key_callback``.

Gated behind the ``[live]`` extra. Withheld from ``erp/viz/__init__.py`` so
``import erp.viz`` stays numpy-only and CI can import it -- same shape as
``theme`` and ``figures``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg

from erp.viz.strips import ContactTrace, RingBuffer

__all__ = ["LivePlots"]

# Contact corrections. Okabe-Ito bluish green and vermillion: an ON pulls the
# estimate onto the surface, an OFF pushes it out of the solid. The script's 3D
# marker uses the same two colours so a strip marker and a dome flash match.
CONTACT_ON = (0, 158, 115)
CONTACT_OFF = (213, 94, 0)

# Fixed limits of the contact window. Force: the plant's dome reads 24-32 N
# resting on the floor under the servos (`scripts/medir_penetracion.py`), so
# 0..40 N shows a press with room. Distance: the estimate has been measured up
# to ~22 mm inside the floor with no FSR (`medir_penetracion.py`, piso/nada),
# and a jog lifts the tip a few cm, so -25..+40 mm.
_FORCE_RANGE = (-2.0, 40.0)
_DIST_RANGE_MM = (-25.0, 40.0)

# Plant / measured / estimate. Colour-blind-safe and legible on the dark
# background pyqtgraph defaults to.
_TRUTH = (120, 220, 140)     # green  -- what the plant did
_MEAS = (250, 200, 90)       # amber  -- what the IMU reported
_EST = (110, 180, 255)       # blue   -- what the filter believes (matches the ghost)
_TARGET = (200, 200, 200)

# One colour per joint (Okabe-Ito orange, sky blue, reddish purple). The joint
# window has its own palette because it is a different comparison: joint
# against joint, not plant against estimate.
_JOINT_COLOURS = [(230, 159, 0), (86, 180, 233), (204, 121, 167)]


class LivePlots:
    """One window per link (a column per sensor kind), plus joint angles + NIS.

    Every push is cheap (an array write); the drawing happens in :meth:`redraw`,
    which the caller calls at its own rate. Keeping those separate is what lets
    the physics run at 500 Hz while the plot runs at 30 -- pushing and drawing
    together would tie the strip's cost to the physics step.

    Sensor vectors are pushed flat, link-major, then kind, then axis:
    ``[link_0 kind_0 x y z, link_0 kind_1 x y z, link_1 kind_0 x y z, ...]``,
    each kind in its own units, sensor site frame. With the default single
    kind that is ``[link1 acc xyz, link2 acc xyz]`` in m/s^2.
    """

    def __init__(
        self,
        *,
        links: Sequence[str] = ("link1", "link2"),
        joint_labels: Sequence[str] = ("rot", "link1", "link2"),
        nis_target: float,
        kinds: Sequence[tuple[str, str, tuple[float, float]]] = (
            ("acc", "m/s^2", (-20.0, 20.0)),
        ),
        truth_label: str = "plant",
        joint_range: tuple[float, float] = (-3.0, 3.0),
        window_s: float = 10.0,
        push_hz: float = 30.0,
        title: str = "blind EKF vs plant",
        contact: tuple[float, float] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        links:
            One window per entry.
        joint_labels:
            Names of the joints on the joint strip, in model order.
        nis_target:
            The consistency target, ``nz`` -- drawn as a horizontal reference.
            A strip of NIS with no target on it cannot be read.
        kinds:
            ``(suffix, units, (y_lo, y_hi))`` per sensor kind: one column of
            three axis plots per kind in every link window, with FIXED y
            limits. The default is the accelerometer alone at +-20 m/s^2 --
            see the module docstring for where that comes from.
        truth_label:
            What the solid line is. ``"plant"`` in simulation, where it is the
            truth; on hardware there is no truth and the solid line is a model
            driven by the command, so the caller must say so.
        joint_range:
            Fixed y limits of the joint plot, rad.
        window_s:
            Seconds of history retained. Older samples fall off the left.
        push_hz:
            The rate the caller intends to push at; sets the buffer capacity.
        contact:
            ``(threshold_n, pen_on_m)`` to add the FSR / contact window, or
            ``None`` (default) for none. The threshold is the force above
            which the filter reads the FSR as ON (N); ``pen_on`` is how far
            the dome sinks into a surface it rests on (m), i.e. where an ON
            correction puts the estimate. Both are drawn as reference lines,
            because a strip of force or distance without them cannot be read.
            Fed through :attr:`contact` (a :class:`ContactTrace`).
        """
        self.links = list(links)
        self.kinds = [(str(s), str(u), (float(r[0]), float(r[1]))) for s, u, r in kinds]
        self.joint_labels = list(joint_labels)
        if not self.kinds:
            raise ValueError("kinds must name at least one sensor kind")
        cap = max(16, int(window_s * push_hz))
        nacc, njt = 3 * len(self.links) * len(self.kinds), len(self.joint_labels)

        self.truth = RingBuffer(cap, nacc)
        self.est = RingBuffer(cap, nacc)
        # Measurements arrive at ~20 Hz, below the push rate, so `cap` holds
        # at least as long a history as the other buffers.
        self.meas = RingBuffer(cap, nacc)
        self.q_true = RingBuffer(cap, njt)
        self.q_est = RingBuffer(cap, njt)
        self.nis = RingBuffer(cap, 1)

        pg.setConfigOptions(antialias=True)
        dash = pg.QtCore.Qt.PenStyle.DashLine
        self.windows: list[Any] = []

        # -- one window per link: a column per kind, a row per axis -------------
        # Curves are appended in the same link-major, kind, axis order as the
        # pushed vectors, so curve i always draws column i of the buffers.
        self._c_truth: list[Any] = []
        self._c_est: list[Any] = []
        self._c_meas: list[Any] = []
        kinds_title = " + ".join(s for s, _, _ in self.kinds)
        for k, link in enumerate(self.links):
            win: Any = pg.GraphicsLayoutWidget(
                show=True, title=f"{link}: {kinds_title} -- {title}"
            )
            win.resize(720 * len(self.kinds), 640)
            win.move(40 + 40 * k, 40 + 40 * k)
            self.windows.append(win)
            first: Any = None
            for c, (suffix, units, y_range) in enumerate(self.kinds):
                for a, axis in enumerate("xyz"):
                    p: Any = win.addPlot(row=a, col=c, title=f"{link}_{suffix}_{axis}")
                    self._setup(p, units, y_range, last=(a == 2))
                    if first is None:
                        first = p
                        p.addLegend(offset=(-10, 10))
                    else:
                        p.setXLink(first)
                    self._c_truth.append(
                        p.plot(pen=pg.mkPen(_TRUTH, width=2), name=truth_label)
                    )
                    self._c_est.append(
                        p.plot(pen=pg.mkPen(_EST, width=2, style=dash),
                               name="estimate h(x_hat)")
                    )
                    # Markers, not a line: there are ~25 predicts between
                    # updates, so a line would imply the filter is corrected
                    # continuously.
                    self._c_meas.append(
                        p.plot(pen=None, symbol="o", symbolSize=5, symbolBrush=_MEAS,
                               symbolPen=None, name="measured")
                    )

        # -- joints, one colour each, over the NIS strip ------------------------
        win_j: Any = pg.GraphicsLayoutWidget(show=True, title=f"joint angles -- {title}")
        win_j.resize(720, 640)
        win_j.move(40 + 40 * len(self.links), 40 + 40 * len(self.links))
        self.windows.append(win_j)
        self._p_joint: Any = win_j.addPlot(
            row=0, col=0, title=f"joint angle: {truth_label} (solid) vs estimate (dashed)"
        )
        self._setup(self._p_joint, "rad", joint_range, last=False)
        self._p_joint.addLegend(offset=(-10, 10))
        # Colour says WHICH joint, line style says plant or estimate -- the
        # same solid/dashed convention as the acceleration windows, so a pair
        # that separates is one colour pulling apart.
        colour = [_JOINT_COLOURS[j % len(_JOINT_COLOURS)] for j in range(njt)]
        self._c_qt = [
            self._p_joint.plot(pen=pg.mkPen(colour[j], width=2), name=f"{lb} {truth_label}")
            for j, lb in enumerate(self.joint_labels)
        ]
        self._c_qe = [
            self._p_joint.plot(pen=pg.mkPen(colour[j], width=2, style=dash), name=f"{lb} est")
            for j, lb in enumerate(self.joint_labels)
        ]

        self._p_nis: Any = win_j.addPlot(row=1, col=0, title="NIS per applied measurement")
        self._p_nis.showGrid(x=True, y=True, alpha=0.25)
        self._p_nis.setLabel("bottom", "t", units="s")
        self._p_nis.setLabel("left", "NIS")
        self._p_nis.setXLink(self._p_joint)
        self._c_nis = self._p_nis.plot(pen=None, symbol="o", symbolSize=5,
                                       symbolBrush=_MEAS, symbolPen=None)
        target = pg.InfiniteLine(pos=nis_target, angle=0,
                                 pen=pg.mkPen(_TARGET, style=dash))
        self._p_nis.addItem(target)
        self._p_nis.setLogMode(y=True)   # a degraded run reaches 1e4; linear hides the good one

        self.contact: ContactTrace | None = None
        if contact is not None:
            self.contact = ContactTrace(cap)
            self._build_contact_window(contact, truth_label, title, dash)

    def _build_contact_window(self, contact: tuple[float, float], truth_label: str,
                              title: str, dash: Any) -> None:
        """Force over signed distance, t linked to the joint strip."""
        threshold_n, pen_on_m = contact
        win: Any = pg.GraphicsLayoutWidget(show=True, title=f"FSR / contact -- {title}")
        win.resize(720, 640)
        k = len(self.windows)
        win.move(40 + 40 * k, 40 + 40 * k)
        self.windows.append(win)
        ref = pg.mkPen(_TARGET, style=dash)

        self._p_force: Any = win.addPlot(row=0, col=0, title="fingertip force (FSR)")
        self._setup(self._p_force, "N", _FORCE_RANGE, last=False)
        self._p_force.addLegend(offset=(-10, 10))
        self._p_force.setXLink(self._p_joint)
        # connect="finite": on hardware the plant force is NaN, and a gap is
        # the honest drawing of "no truth here".
        self._c_force = self._p_force.plot(pen=pg.mkPen(_TRUTH, width=2),
                                           name=truth_label, connect="finite")
        self._c_force_meas = self._p_force.plot(pen=None, symbol="o", symbolSize=5,
                                                symbolBrush=_MEAS, symbolPen=None,
                                                name="measured")
        self._p_force.addItem(pg.InfiniteLine(pos=threshold_n, angle=0, pen=ref,
                                              label=f"ON above {threshold_n:g} N",
                                              labelOpts={"position": 0.9, "color": _TARGET}))

        self._p_dist: Any = win.addPlot(
            row=1, col=0,
            title=f"dome to nearest solid: signed distance (< 0 inside; "
                  f"clipped at {_DIST_RANGE_MM[1]:g} mm = far)",
        )
        self._setup(self._p_dist, "mm", _DIST_RANGE_MM, last=True)
        self._p_dist.addLegend(offset=(-10, 10))
        self._p_dist.setXLink(self._p_joint)
        self._c_d_true = self._p_dist.plot(pen=pg.mkPen(_TRUTH, width=2),
                                           name=truth_label, connect="finite")
        self._c_d_est = self._p_dist.plot(pen=pg.mkPen(_EST, width=2, style=dash),
                                          name="estimate d(x_hat)")
        self._c_on = self._p_dist.plot(pen=None, symbol="t1", symbolSize=9,
                                       symbolBrush=CONTACT_ON, symbolPen=None,
                                       name="ON: pulled to surface")
        self._c_off = self._p_dist.plot(pen=None, symbol="t", symbolSize=9,
                                        symbolBrush=CONTACT_OFF, symbolPen=None,
                                        name="OFF: pushed out")
        self._p_dist.addItem(pg.InfiniteLine(pos=0.0, angle=0, pen=ref))
        self._p_dist.addItem(pg.InfiniteLine(pos=-1e3 * pen_on_m, angle=0,
                                             pen=pg.mkPen(CONTACT_ON, style=dash)))

    @staticmethod
    def _setup(p: Any, units: str, y_range: tuple[float, float], *, last: bool) -> None:
        p.showGrid(x=True, y=True, alpha=0.25)
        p.setLabel("left", units)
        if last:
            p.setLabel("bottom", "t", units="s")
        # `setYRange` alone turns y auto-range off, but the "A" button on the
        # plot turns it back on; disabling it explicitly makes the limit stick
        # until the user drags the axis on purpose.
        p.setYRange(*y_range, padding=0.0)
        p.enableAutoRange(axis="y", enable=False)

    # -- ingestion ----------------------------------------------------------

    def push_plant(self, t: float, y: npt.ArrayLike, q: npt.ArrayLike) -> None:
        """The solid line: sensor channels (flat, see the class docstring) and
        joint angles (rad). The plant in simulation, the command model on hardware."""
        self.truth.append(t, y)
        self.q_true.append(t, q)

    def push_estimate(self, t: float, y: npt.ArrayLike, q: npt.ArrayLike) -> None:
        """``h(x_hat)`` on the same channels (site frame), and the estimated
        joint angles (rad)."""
        self.est.append(t, y)
        self.q_est.append(t, q)

    def push_measurement(self, t: float, y: npt.ArrayLike) -> None:
        """One sample on the same channels, stamped at its sample instant.
        Drawn as markers."""
        self.meas.append(t, y)

    def push_nis(self, t: float, nis: float) -> None:
        self.nis.append(t, [nis])

    # -- drawing ------------------------------------------------------------

    def redraw(self) -> None:
        """Push every buffer into its curve. Call at ~30 Hz, not per physics step."""
        t_tr, Y_tr = self.truth.view()
        t_es, Y_es = self.est.view()
        t_me, Y_me = self.meas.view()
        for i in range(len(self._c_truth)):
            self._c_truth[i].setData(t_tr, Y_tr[:, i])
            self._c_est[i].setData(t_es, Y_es[:, i])
            self._c_meas[i].setData(t_me, Y_me[:, i])

        t_qt, Q_t = self.q_true.view()
        t_qe, Q_e = self.q_est.view()
        for j in range(len(self.joint_labels)):
            self._c_qt[j].setData(t_qt, Q_t[:, j])
            self._c_qe[j].setData(t_qe, Q_e[:, j])

        t_n, N = self.nis.view()
        self._c_nis.setData(t_n, np.clip(N.ravel(), 1e-3, None))

        if self.contact is not None:
            c = self.contact
            t_s, S = c.state.view()
            self._c_force.setData(t_s, S[:, 0])
            # Clipped at the top so a far-away dome rides the upper edge instead
            # of leaving the plot empty: at rest it is ~186 mm above the floor,
            # far outside a range chosen to resolve sub-mm contact. The title
            # says so. Not clipped below: a deep estimate must stay visible.
            top = _DIST_RANGE_MM[1]
            self._c_d_true.setData(t_s, np.minimum(1e3 * S[:, 1], top))
            self._c_d_est.setData(t_s, np.minimum(1e3 * S[:, 2], top))
            t_m, F = c.measured.view()
            self._c_force_meas.setData(t_m, F[:, 0])
            t_on, D_on = c.on.view()
            self._c_on.setData(t_on, 1e3 * D_on[:, 0])
            t_off, D_off = c.off.view()
            self._c_off.setData(t_off, 1e3 * D_off[:, 0])

    def close(self) -> None:
        for win in self.windows:
            win.close()
