"""Real-time strip charts for the interactive demo. pyqtgraph, not matplotlib.

Why a second backend rather than reusing :mod:`erp.viz.figures`: matplotlib
redraws the whole figure, and at a 30 Hz strip it does not keep up. The arm
then stutters in the viewer -- because the render tick is waiting on the draw
-- which looks like a physics or a filter problem and is neither.
``pyproject.toml`` still carries a comment about pyqtgraph having been a
dependency here once, for a finger viewer that no longer exists.

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

from erp.viz.strips import RingBuffer

__all__ = ["LivePlots"]

# Colour-blind-safe and legible on the dark background pyqtgraph defaults to.
_TRUTH = (120, 220, 140)     # green  -- what the plant did
_MEAS = (250, 200, 90)       # amber  -- what the IMU reported
_EST = (110, 180, 255)       # blue   -- what the filter believes (matches the ghost)
_TARGET = (200, 200, 200)


class LivePlots:
    """Three strips: a sensor channel, the joint angles, and NIS per update.

    Every push is cheap (an array write); the drawing happens in :meth:`redraw`,
    which the caller calls at its own rate. Keeping those separate is what lets
    the physics run at 500 Hz while the plot runs at 30 -- pushing and drawing
    together would tie the strip's cost to the physics step.
    """

    def __init__(
        self,
        *,
        channel_labels: Sequence[str],
        joint_labels: Sequence[str],
        nis_target: float,
        window_s: float = 10.0,
        push_hz: float = 30.0,
        title: str = "blind EKF vs plant",
    ) -> None:
        """
        Parameters
        ----------
        channel_labels:
            Names of the IMU channels plotted on the first strip. Two is
            readable; twelve is not, and the question they answer is the same.
        joint_labels:
            Names of the joints on the second strip, in model order.
        nis_target:
            The consistency target, ``nz`` -- drawn as a horizontal reference.
            A strip of NIS with no target on it cannot be read.
        window_s:
            Seconds of history retained. Older samples fall off the left.
        push_hz:
            The rate the caller intends to push at; sets the buffer capacity.
        """
        self.channel_labels = list(channel_labels)
        self.joint_labels = list(joint_labels)
        cap = max(16, int(window_s * push_hz))
        nch, njt = len(self.channel_labels), len(self.joint_labels)

        self.truth = RingBuffer(cap, nch)
        self.est = RingBuffer(cap, nch)
        self.meas = RingBuffer(cap, nch)
        self.q_true = RingBuffer(cap, njt)
        self.q_est = RingBuffer(cap, njt)
        self.nis = RingBuffer(cap, 1)

        pg.setConfigOptions(antialias=True)
        self.win: Any = pg.GraphicsLayoutWidget(show=True, title=title)
        self.win.resize(980, 760)

        self._p_sensor: Any = self.win.addPlot(
            row=0, col=0, title="IMU: plant / measured / h(x_hat)"
        )
        self._p_joint: Any = self.win.addPlot(row=1, col=0, title="joint angle: plant vs estimate")
        self._p_nis: Any = self.win.addPlot(row=2, col=0, title="NIS per applied measurement")
        for p in (self._p_sensor, self._p_joint, self._p_nis):
            p.showGrid(x=True, y=True, alpha=0.25)
            p.setLabel("bottom", "t", units="s")
        self._p_sensor.setLabel("left", "m/s^2 and rad/s, site frame")
        self._p_joint.setLabel("left", "rad")
        self._p_nis.setLabel("left", "NIS")
        self._p_sensor.addLegend(offset=(-10, 10))
        self._p_joint.addLegend(offset=(-10, 10))

        # Solid = plant, dashed = estimate, dots = the samples actually applied.
        self._c_truth = [
            self._p_sensor.plot(pen=pg.mkPen(_TRUTH, width=2), name=f"{lb} plant")
            for lb in self.channel_labels
        ]
        self._c_est = [
            self._p_sensor.plot(pen=pg.mkPen(_EST, width=2, style=pg.QtCore.Qt.PenStyle.DashLine),
                                name=f"{lb} est")
            for lb in self.channel_labels
        ]
        self._c_meas = [
            self._p_sensor.plot(pen=None, symbol="o", symbolSize=4,
                                symbolBrush=_MEAS, symbolPen=None, name=f"{lb} meas")
            for lb in self.channel_labels
        ]
        self._c_qt = [
            self._p_joint.plot(pen=pg.mkPen(_TRUTH, width=2), name=f"{lb} plant")
            for lb in self.joint_labels
        ]
        self._c_qe = [
            self._p_joint.plot(pen=pg.mkPen(_EST, width=2, style=pg.QtCore.Qt.PenStyle.DashLine),
                               name=f"{lb} est")
            for lb in self.joint_labels
        ]
        self._c_nis = self._p_nis.plot(pen=None, symbol="o", symbolSize=5,
                                       symbolBrush=_MEAS, symbolPen=None)
        target = pg.InfiniteLine(pos=nis_target, angle=0,
                                 pen=pg.mkPen(_TARGET, style=pg.QtCore.Qt.PenStyle.DashLine))
        self._p_nis.addItem(target)
        self._p_nis.setLogMode(y=True)   # a degraded run reaches 1e4; linear hides the good one

    # -- ingestion ----------------------------------------------------------

    def push_plant(self, t: float, y: npt.ArrayLike, q: npt.ArrayLike) -> None:
        """The plant's current channel readings and joint angles (rad)."""
        self.truth.append(t, y)
        self.q_true.append(t, q)

    def push_estimate(self, t: float, y: npt.ArrayLike, q: npt.ArrayLike) -> None:
        """``h(x_hat)`` on the same channels, and the estimated joint angles (rad)."""
        self.est.append(t, y)
        self.q_est.append(t, q)

    def push_measurement(self, t: float, z: npt.ArrayLike) -> None:
        """One applied sample. Drawn as markers, not a line.

        The markers are the point: there are ~25 predicts between updates, so a
        line here would imply the filter is being corrected continuously.
        """
        self.meas.append(t, z)

    def push_nis(self, t: float, nis: float) -> None:
        self.nis.append(t, [nis])

    # -- drawing ------------------------------------------------------------

    def redraw(self) -> None:
        """Push every buffer into its curve. Call at ~30 Hz, not per physics step."""
        t_tr, Y_tr = self.truth.view()
        t_es, Y_es = self.est.view()
        t_me, Y_me = self.meas.view()
        for i in range(len(self.channel_labels)):
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

    def close(self) -> None:
        self.win.close()
