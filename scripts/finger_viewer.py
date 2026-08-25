"""Live finger viewer: command vs. truth vs. EKF estimate, with tip uncertainty.

What this shows, and what each signal actually is:

* **U** -- the commanded joint angle. Comes from the two potentiometers when
  hardware is connected, otherwise from a synthetic sweep. It is an *input*,
  not a measurement: you turn the pots, the simulated finger follows.
* **Q true** -- MuJoCo's own ``qpos``. MuJoCo is the plant here, so this is
  ground truth, and nothing derived from it reaches the filter except through
  a noisy sensor.
* **Q est** -- what the EKF in ``erp`` reconstructs, fed only by noisy
  encoder and gyro samples arriving at two different rates.
* **Tip** -- ``fk_tip`` of the estimate, with the 2-sigma ellipse from
  ``tip_covariance``. The ellipse is the point: a position without a
  trustworthy covariance is a guess.

The gap between U and Q true is the servo's own lag and droop. The gap between
Q true and Q est is what the estimator costs you. They are different failures
and the plots keep them apart.

Expect the filter to TRACK well and to be OVERCONFIDENT
-------------------------------------------------------
Position tracking is good -- of order 0.1 deg on a 60 deg sweep -- but the
+/-2 sigma coverage runs near 10 %, not 95 %. That is not a bug in the plumbing,
and it is worth understanding before trusting the ellipse:

``erp``'s ``servoed_finger_model`` integrates the coupled ``(q, v, a)`` system
exactly, so over a step it effectively uses the *average* servo activation.
MuJoCo evaluates actuator force at a step endpoint instead -- the start, or with
``actearly="true"`` the end. The difference is ``kp * delta_act / inertia``,
about 26 rad/s^2 here, and it is **deterministic and time-correlated**, not
white.

So process noise cannot absorb it. Measured: shrinking the plant timestep from
2 ms to 0.25 ms cuts the one-step residual tenfold and moves coverage from 0.09
to 0.09. Raising ``--psd-alpha`` to ~400 does force coverage to 0.97, but that
is whitening a bias -- exactly the mistake ``docs/theory/finger_imu_ekf.md``
section 4 documents, where inflating the noise flattens NIS while leaving NEES
untouched.

The real fixes are the ones that report lists: match the model to the plant's
convention, or augment the state. Until then the status bar reports live
coverage so the inconsistency stays visible rather than implied.

Run::

    python scripts/finger_viewer.py                    # synthetic input
    python scripts/finger_viewer.py --pot --simulate   # pot GUI stack, no hardware
    python scripts/finger_viewer.py --pot --port COM5  # real potentiometers

Not part of the ``erp`` package: this is an application. The reusable parts
(kinematics, ellipse geometry, the estimator itself) live in ``erp``.
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "software" / "src"))
# The potentiometer stack already exists; reuse it rather than reimplementing
# the serial protocol. The module name carries a typo -- it is theirs, and
# renaming it would break the user's own launch commands.
sys.path.insert(0, str(_ROOT / "notebooks"))

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

import mujoco as mj  # noqa: E402
import pyqtgraph as pg  # noqa: E402
from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402

from erp.core.timeline import InputHistory  # noqa: E402
from erp.core.types import CalibrationResult, ControlInput, GaussianState, Measurement  # noqa: E402
from erp.estimators.ekf import ExtendedKalmanFilter  # noqa: E402
from erp.fusion.engine import FusionEngine  # noqa: E402
from erp.models.base import MeasurementModel  # noqa: E402
from erp.models.kinematics import FingerGeometry, fk_tip, tip_covariance  # noqa: E402
from erp.models.linear import JointParams, servoed_finger_model  # noqa: E402
from erp.models.measurement import joint_block_model  # noqa: E402
from erp.sensors.base import Sensor  # noqa: E402
from erp.viz.ellipse import covariance_ellipse  # noqa: E402

# ---------------------------------------------------------------------------
# Plant parameters -- single source of truth, mirroring the notebook.
# The XML is topology only; every physical number lives here.
# ---------------------------------------------------------------------------

GEOMETRY = dict(l0=0.030, w0=0.008, l1=0.035, w1=0.006, i1=0.0175,
                l2=0.030, w2=0.005, i2=0.0150, b=0.002)
SIM_OPTIONS = dict(timestep=0.002, gravity="0 0 -9.81", integrator="implicitfast")
JOINTS_CFG = {"joint_1": dict(range_deg=(-90.0, 90.0)),
              "joint_2": dict(range_deg=(-120.0, 120.0))}
ACTUATORS = {
    "act_joint_1": dict(joint="joint_1", kp=2.0, kv=0.025,
                        force_max=0.30, armature=4e-5, damping=1e-3, tau=0.004),
    "act_joint_2": dict(joint="joint_2", kp=0.8, kv=0.010,
                        force_max=0.15, armature=2e-5, damping=8e-4, tau=0.004),
}

SIG_ENC = math.radians(0.15)     # encoder noise, rad  (~0.15 deg, a 10-bit pot)
SIG_GYRO = 5e-3                  # rate gyro noise, rad/s
PSD_ALPHA = 4.0                  # angular-acceleration disturbance PSD, rad^2/s^3
PSD_ACT = 1e-6                   # servo activation jitter PSD, rad^2/s
F_INPUT, F_ENC, F_GYRO = 200.0, 100.0, 250.0

HISTORY_SECONDS = 12.0
UI_PERIOD_MS = 33
COLOR_U, COLOR_TRUE, COLOR_EST = "#888888", "#4c8fd8", "#e08a3c"


def relocate_actuator_inertia(xml: str) -> str:
    """Move ``armature``/``damping`` from ``<general>`` onto the driven ``<joint>``.

    MuJoCo 3.4 does not accept either attribute on an actuator -- it rejects the
    template in ``notebooks/assets/finger_2link.xml`` outright. Rather than edit
    that asset (the notebook builds its ``fields`` dict around the current
    layout), the equivalent model is produced here at load time.

    The transformation is physics-preserving *because the template uses
    ``gear="1"``*: with unity gear the actuator's reflected rotor inertia is
    exactly a joint armature, and gearbox friction simply adds to the joint's
    structural damping, which is what the template's own comments already say
    it does.
    """
    root = ET.fromstring(xml)
    joints = {j.get("name"): j for j in root.iter("joint")}
    for act in root.iter("general"):
        target = joints.get(act.get("joint", ""))
        if target is None:
            continue
        if act.get("gear", "1") != "1":
            raise ValueError("relocation assumes gear=1; reflected inertia would rescale")
        armature = act.attrib.pop("armature", None)
        damping = act.attrib.pop("damping", None)
        if armature is not None:
            target.set("armature", armature)
        if damping is not None:
            total = float(target.get("damping", "0")) + float(damping)
            target.set("damping", repr(total))
    return ET.tostring(root, encoding="unicode")


def build_xml(template: str, *, motor_lag: bool, gravity: str,
              timestep: float, actearly: bool) -> str:
    """Fill the MuJoCo template. ``motor_lag`` adds the servo activation state.

    ``actearly`` selects which end of the step MuJoCo evaluates actuator force
    at. Neither setting matches the estimator's exact continuous integration;
    ``False`` halves the discrepancy. See the module docstring.
    """
    fields = dict(GEOMETRY, **{**SIM_OPTIONS, "gravity": gravity, "timestep": timestep})
    for name, joint in JOINTS_CFG.items():
        lo, hi = joint["range_deg"]
        fields[f"{name}_range_deg"] = f"{lo} {hi}"
    for idx, (act_name, act) in enumerate(ACTUATORS.items(), start=1):
        lo, hi = np.deg2rad(JOINTS_CFG[act["joint"]]["range_deg"])
        fields[f"{act_name}_ctrlrange"] = f"{lo:.6f} {hi:.6f}"
        fields[f"{act_name}_forcerange"] = f"{-act['force_max']} {act['force_max']}"
        fields[f"{act_name}_armature"] = act["armature"]
        fields[f"{act_name}_damping"] = act["damping"]
        fields[f"{act_name}_gainprm"] = f"{act['kp']} 0 0"
        fields[f"{act_name}_biasprm"] = f"0 {-act['kp']} {-act['kv']}"
        fields[f"motor_dynamics_{idx}"] = (
            f'dyntype="filterexact" dynprm="{act["tau"]}" '
            f'actearly="{str(actearly).lower()}" actrange="{lo:.6f} {hi:.6f}"'
            if motor_lag else ""
        )
    return relocate_actuator_inertia(template.format(**fields))


def joint_params_from_plant(model: mj.MjModel, data: mj.MjData,
                            psd_alpha: float = PSD_ALPHA) -> list[JointParams]:
    """Derive the estimator's linear model from the compiled plant.

    Effective inertia is read out of MuJoCo's own mass matrix at the neutral
    pose rather than guessed, so the filter's model and the plant agree on the
    one number that is hardest to eyeball. Only the diagonal is taken: the
    linear model treats the joints as decoupled, which is a real approximation,
    but reflected rotor inertia (4e-5) dominates the inter-link coupling
    (~3e-6) by an order of magnitude here.
    """
    # MuJoCo >= 3.11 takes the MjData itself and writes into ``dst``; the older
    # form passed the sparse ``data.qM`` as the third argument instead.
    M = np.zeros((model.nv, model.nv))
    mj.mj_fullM(model, data, M)
    out = []
    for i, act in enumerate(ACTUATORS.values()):
        out.append(JointParams(
            kp=act["kp"], kv=act["kv"],
            damping=GEOMETRY["b"] + act["damping"],   # structural + gearbox, summed
            inertia=float(M[i, i]),
            tau=act["tau"], psd_alpha=psd_alpha, psd_act=PSD_ACT,
        ))
    return out


# ---------------------------------------------------------------------------
# Input sources -- these produce U, the command
# ---------------------------------------------------------------------------


class SyntheticInput:
    """Raised-cosine sweep, so angle and rate both start at zero."""

    label = "synthetic sweep"

    def __init__(self) -> None:
        self._amp = np.deg2rad([45.0, 60.0])
        self._freq = np.array([0.25, 0.16])

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def command(self, t: float) -> np.ndarray:
        return self._amp / 2 * (1 - np.cos(2 * np.pi * self._freq * t))

    def status(self) -> str:
        return "synthetic"


class PotentiometerInput:
    """Two potentiometers driving the command, via the existing logger stack.

    Reuses ``notebooks/Potenctiometerlogging.py`` for the serial protocol,
    micros() unwrapping and the counts-to-degrees calibration, so there is one
    implementation of that wire format rather than two that can drift apart.
    """

    label = "potentiometers"

    def __init__(self, port: str | None, baud: int, simulate: bool,
                 cal_span_deg: tuple[float, float]) -> None:
        import Potenctiometerlogging as potlog  # noqa: N813 - user's module name

        self._potlog = potlog
        lo, hi = cal_span_deg
        self._cal = (potlog.ChannelCal("theta1 (A0)", angle_min=lo, angle_max=hi),
                     potlog.ChannelCal("theta2 (A1)", angle_min=lo, angle_max=hi))
        self._source = (potlog.SimSource() if simulate
                        else potlog.SerialSource(port or "", baud))
        self._queue: queue.Queue = queue.Queue()
        self._reader = potlog.ReaderThread(self._source, self._queue)
        self._latest = np.zeros(2)
        self._count = 0

    def start(self) -> None:
        self._reader.start()

    def stop(self) -> None:
        self._reader.stop()
        self._reader.join(timeout=2.0)

    def command(self, t: float) -> np.ndarray:
        """Latest pot reading, held. ``t`` is ignored: the pots set their own pace."""
        try:
            while True:
                _, a0, a1 = self._queue.get_nowait()
                self._latest = np.deg2rad([self._cal[0].to_angle(a0),
                                           self._cal[1].to_angle(a1)])
                self._count += 1
        except queue.Empty:
            pass
        return self._latest

    def status(self) -> str:
        if self._reader.error:
            return f"serial error: {self._reader.error}"
        return f"{self._count} pot samples"


# ---------------------------------------------------------------------------
# Sensors -- synthesised from the plant, delivered through the Sensor contract
# ---------------------------------------------------------------------------


class QueueSensor(Sensor):
    """Sensor fed by the simulation loop, drained by the FusionEngine.

    A real driver would fill the same queue from a serial port. The engine
    cannot tell the difference, which is the point of the Adapter.
    """

    def __init__(self, model: MeasurementModel, dim: int, frame_id: str) -> None:
        self._model, self._dim, self._frame_id = model, dim, frame_id
        self._pending: list[Measurement] = []

    def emit(self, z: np.ndarray, t: float) -> None:
        self._pending.append(Measurement(z=z, timestamp=t, frame_id=self._frame_id))

    def read(self) -> Measurement | None:
        return self._pending.pop(0) if self._pending else None

    def drain(self) -> list[Measurement]:
        out, self._pending = self._pending, []
        return out

    def calibrate(self) -> CalibrationResult:
        return CalibrationResult(np.zeros(self._dim), np.ones(self._dim),
                                 np.zeros((self._dim, self._dim)), True, "synthetic")

    @property
    def measurement_model(self) -> MeasurementModel:
        return self._model


# ---------------------------------------------------------------------------
# Session: plant + estimator, advanced together
# ---------------------------------------------------------------------------


@dataclass
class Trace:
    """Rolling history for plotting."""

    t: list[float] = field(default_factory=list)
    u: list[np.ndarray] = field(default_factory=list)
    q_true: list[np.ndarray] = field(default_factory=list)
    q_est: list[np.ndarray] = field(default_factory=list)
    sigma: list[np.ndarray] = field(default_factory=list)

    def append(self, t: float, u: np.ndarray, q_true: np.ndarray,
               q_est: np.ndarray, sigma: np.ndarray) -> None:
        self.t.append(t)
        self.u.append(u.copy())
        self.q_true.append(q_true.copy())
        self.q_est.append(q_est.copy())
        self.sigma.append(sigma.copy())

    def trim(self, horizon: float) -> None:
        while self.t and self.t[-1] - self.t[0] > horizon:
            for seq in (self.t, self.u, self.q_true, self.q_est, self.sigma):
                seq.pop(0)

    def arrays(self) -> tuple[np.ndarray, ...]:
        return (np.asarray(self.t), np.asarray(self.u), np.asarray(self.q_true),
                np.asarray(self.q_est), np.asarray(self.sigma))


class FingerSession:
    """Drives the MuJoCo plant and the erp estimator on one clock."""

    def __init__(self, source: SyntheticInput | PotentiometerInput,
                 *, gravity: bool, buffer_horizon: float, seed: int,
                 psd_alpha: float = PSD_ALPHA, plant_dt: float = 0.002,
                 actearly: bool = True) -> None:
        template = (_ROOT / "notebooks" / "assets" / "finger_2link.xml").read_text(
            encoding="utf-8")
        grav = SIM_OPTIONS["gravity"] if gravity else "0 0 0"
        self.model = mj.MjModel.from_xml_string(build_xml(
            template, motor_lag=True, gravity=grav,
            timestep=plant_dt, actearly=actearly))
        self.data = mj.MjData(self.model)
        mj.mj_forward(self.model, self.data)

        self.geom = FingerGeometry(GEOMETRY["l0"], GEOMETRY["l1"], GEOMETRY["l2"])
        self.source = source
        self.rng = np.random.default_rng(seed)
        self.dt = float(self.model.opt.timestep)

        joints = joint_params_from_plant(self.model, self.data, psd_alpha)
        process = servoed_finger_model(joints)
        P0 = np.diag([1e-6, 1e-6, 1e-4, 1e-4, 1e-6, 1e-6])
        self.ekf = ExtendedKalmanFilter(process, GaussianState(np.zeros(6), P0))

        self.enc = QueueSensor(joint_block_model(2, "angle", SIG_ENC), 2, "joint")
        self.gyro = QueueSensor(joint_block_model(2, "rate", SIG_GYRO), 2, "joint")

        self.inputs = InputHistory()
        self.inputs.push(ControlInput(u=source.command(0.0).copy(), timestamp=0.0))
        self.engine = FusionEngine(
            self.ekf, {"enc": self.enc, "gyro": self.gyro}, self.inputs,
            buffer_horizon=buffer_horizon, t0=0.0,
        )

        self.trace = Trace()
        self._next = {"input": 1.0 / F_INPUT, "enc": 1.0 / F_ENC, "gyro": 1.0 / F_GYRO}
        self._u = self.inputs.u_at(0.0).copy()

    @property
    def sim_time(self) -> float:
        return float(self.data.time)

    def advance(self, wall_dt: float) -> None:
        """Step the plant forward by ``wall_dt`` seconds of simulated time."""
        target = self.sim_time + wall_dt
        while self.sim_time < target:
            t = self.sim_time

            if t >= self._next["input"]:
                self._u = np.clip(
                    self.source.command(t),
                    self.model.actuator_ctrlrange[:, 0],
                    self.model.actuator_ctrlrange[:, 1],
                )
                self.inputs.push(ControlInput(u=self._u.copy(), timestamp=t))
                self._next["input"] += 1.0 / F_INPUT

            # Sensors sample the state *before* integration, so the timestamp
            # and the value refer to the same instant. Logging after mj_step
            # while reusing the previous ctrl desynchronises by a whole step,
            # which then masquerades as model error.
            if t >= self._next["enc"]:
                z = self.data.qpos[:2] + self.rng.normal(0, SIG_ENC, 2)
                self.enc.emit(z, t)
                self._next["enc"] += 1.0 / F_ENC
            if t >= self._next["gyro"]:
                z = self.data.qvel[:2] + self.rng.normal(0, SIG_GYRO, 2)
                self.gyro.emit(z, t)
                self._next["gyro"] += 1.0 / F_GYRO

            self.data.ctrl[:] = self._u
            mj.mj_step(self.model, self.data)

        self.engine.step(self.sim_time)
        self._record()
        self.inputs.prune_before(self.engine.time - 1.0)

    def _record(self) -> None:
        state = self.ekf.state
        self.trace.append(
            self.sim_time, self._u, np.asarray(self.data.qpos[:2]),
            state.x[:2], np.sqrt(np.diag(state.P)[:2]),
        )
        self.trace.trim(HISTORY_SECONDS)

    def tip(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(true_tip_yz, est_tip_yz, tip_covariance)``."""
        state = self.ekf.state
        true_p = fk_tip(np.asarray(self.data.qpos[:2]), self.geom)[1:]
        est_p = fk_tip(state.x[:2], self.geom)[1:]
        return true_p, est_p, tip_covariance(state.x[:2], state.P[:2, :2], self.geom)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


class Window(QtWidgets.QMainWindow):
    def __init__(self, session: FingerSession, speed: float, ellipse_gain: float) -> None:
        super().__init__()
        self.session = session
        self.speed = speed
        self.ellipse_gain = ellipse_gain
        self.setWindowTitle("ERP finger - command vs. truth vs. EKF estimate")
        self.resize(1500, 880)

        pg.setConfigOptions(antialias=True, background="w", foreground="k")
        self.renderer = mj.Renderer(session.model, height=440, width=560)

        self.view = QtWidgets.QLabel()
        self.view.setFixedSize(560, 440)
        self.view.setStyleSheet("background: #202020;")

        self.plot_tip = self._make_plot("tip z", "tip y", legend=True)
        self.plot_tip.setAspectLocked(True)
        self.curve_tip_true = self.plot_tip.plot(
            pen=None, symbol="o", symbolSize=9, symbolBrush=COLOR_TRUE, name="true tip")
        self.curve_tip_est = self.plot_tip.plot(
            pen=None, symbol="x", symbolSize=11, symbolPen=pg.mkPen(COLOR_EST, width=2),
            name="EKF tip")
        self.curve_ellipse = self.plot_tip.plot(
            pen=pg.mkPen(COLOR_EST, width=2, style=QtCore.Qt.DashLine), name="2-sigma")
        self.curve_link = self.plot_tip.plot(pen=pg.mkPen("#cccccc", width=6))

        self.joint_plots, self.joint_curves = [], []
        for j in (1, 2):
            plot = self._make_plot(f"joint {j}", "time", units="deg", legend=True)
            curves = {
                "u": plot.plot(pen=pg.mkPen(COLOR_U, width=2,
                                            style=QtCore.Qt.DashLine), name="U command"),
                "true": plot.plot(pen=pg.mkPen(COLOR_TRUE, width=2), name="Q true"),
                "est": plot.plot(pen=pg.mkPen(COLOR_EST, width=2), name="Q est"),
            }
            self.joint_plots.append(plot)
            self.joint_curves.append(curves)
        self.joint_plots[1].setXLink(self.joint_plots[0])

        self.plot_err = self._make_plot("Q est - Q true", "time", units="deg", legend=True)
        self.plot_err.setXLink(self.joint_plots[0])
        self.err_curves, self.band_curves = [], []
        for j, color in enumerate((COLOR_TRUE, COLOR_EST)):
            self.err_curves.append(
                self.plot_err.plot(pen=pg.mkPen(color, width=2), name=f"joint {j + 1}"))
            self.band_curves.append((
                self.plot_err.plot(pen=pg.mkPen(color, width=1, style=QtCore.Qt.DotLine)),
                self.plot_err.plot(pen=pg.mkPen(color, width=1, style=QtCore.Qt.DotLine)),
            ))

        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.view)
        left.addWidget(self.plot_tip, 1)
        right = QtWidgets.QVBoxLayout()
        for widget in (*self.joint_plots, self.plot_err):
            right.addWidget(widget, 1)

        root = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(root)
        row.addLayout(left)
        row.addLayout(right, 1)
        self.setCentralWidget(root)

        self.status = self.statusBar()
        self._wall_prev = time.perf_counter()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(UI_PERIOD_MS)

    @staticmethod
    def _make_plot(left: str, bottom: str, units: str | None = None,
                   legend: bool = False) -> pg.PlotWidget:
        plot = pg.PlotWidget()
        plot.setLabel("left", left)
        plot.setLabel("bottom", bottom, units=units)
        plot.showGrid(x=True, y=True, alpha=0.3)
        if legend:
            plot.addLegend(offset=(-10, 10))
        return plot

    def _tick(self) -> None:
        now = time.perf_counter()
        # Cap the catch-up so a stalled frame cannot spiral into a long
        # simulation burst that stalls the next frame in turn.
        wall_dt = min(now - self._wall_prev, 0.10) * self.speed
        self._wall_prev = now
        try:
            self.session.advance(wall_dt)
        except Exception as exc:  # keep the window alive to show the message
            self.timer.stop()
            QtWidgets.QMessageBox.critical(self, "Simulation stopped", repr(exc))
            return
        self._draw_scene()
        self._draw_plots()

    def _draw_scene(self) -> None:
        self.renderer.update_scene(self.session.data, camera="vista_plana")
        frame = np.ascontiguousarray(self.renderer.render())
        h, w, _ = frame.shape
        image = QtGui.QImage(frame.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self.view.setPixmap(QtGui.QPixmap.fromImage(image))

    def _draw_plots(self) -> None:
        t, u, q_true, q_est, sigma = self.session.trace.arrays()
        if t.size < 2:
            return
        deg = np.degrees

        for j, curves in enumerate(self.joint_curves):
            curves["u"].setData(t, deg(u[:, j]))
            curves["true"].setData(t, deg(q_true[:, j]))
            curves["est"].setData(t, deg(q_est[:, j]))

        err = deg(q_est - q_true)
        band = 2.0 * deg(sigma)
        for j, curve in enumerate(self.err_curves):
            curve.setData(t, err[:, j])
            lo, hi = self.band_curves[j]
            lo.setData(t, -band[:, j])
            hi.setData(t, band[:, j])

        true_p, est_p, cov = self.session.tip()
        # Plotted z-horizontal / y-vertical to match the camera's view of the
        # finger, and the ellipse is magnified: at these noise levels it is a
        # few tens of microns against a ~55 mm travel, invisible at 1:1.
        self.curve_tip_true.setData([true_p[1]], [true_p[0]])
        self.curve_tip_est.setData([est_p[1]], [est_p[0]])
        pts = covariance_ellipse(cov * self.ellipse_gain**2, est_p, n_sigma=2.0)
        self.curve_ellipse.setData(pts[:, 1], pts[:, 0])

        q = self.session.data.qpos[:2]
        g = self.session.geom
        joints = np.array([
            [0.0, 0.0],
            [0.0, g.l0],
            [-g.l1 * np.sin(q[0]), g.l0 + g.l1 * np.cos(q[0])],
        ])
        tip3 = fk_tip(np.asarray(q), g)
        self.curve_link.setData(
            np.append(joints[:, 1], tip3[2]), np.append(joints[:, 0], tip3[1]))

        major = float(np.sqrt(np.linalg.eigvalsh(cov).max()))
        minor = float(np.sqrt(np.linalg.eigvalsh(cov).min()))
        # Coverage is reported live because it is the number that says whether
        # the ellipse above can be believed. Near 0.95 the filter is honest;
        # near 0.10 it is tracking well and lying about how well.
        coverage = float((np.abs(err) <= band).mean())
        self.status.showMessage(
            f"sim {self.session.sim_time:7.2f} s  |  filter {self.session.engine.time:7.2f} s"
            f"  |  tip err {1e6 * np.linalg.norm(est_p - true_p):6.1f} um"
            f"  |  2-sigma {2e6 * major:5.1f} x {2e6 * minor:4.1f} um"
            f" ({major / max(minor, 1e-18):.0f}:1)"
            f"  |  +/-2sig coverage {coverage:4.2f} (want 0.95)"
            f"  |  {self.session.source.status()}"
            f"  |  dropped {sum(self.session.engine.discarded.values())}"
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.timer.stop()
        self.session.source.stop()
        super().closeEvent(event)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pot", action="store_true",
                    help="drive the command from the potentiometers")
    ap.add_argument("--port", help="serial port, e.g. COM5 (implies --pot)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--simulate", action="store_true",
                    help="with --pot, use the logger's synthetic source instead of hardware")
    ap.add_argument("--pot-range", type=float, nargs=2, default=(-60.0, 60.0),
                    metavar=("MIN_DEG", "MAX_DEG"),
                    help="joint angle the pot endpoints map to (default: -60 60)")
    ap.add_argument("--speed", type=float, default=1.0, help="simulated seconds per real second")
    ap.add_argument("--buffer-horizon", type=float, default=0.0,
                    help="seconds the fusion engine holds measurements before processing")
    ap.add_argument("--no-gravity", action="store_true", help="disable gravity")
    ap.add_argument("--psd-alpha", type=float, default=PSD_ALPHA,
                    help="angular-acceleration disturbance PSD, rad^2/s^3. ~1.3 matches the "
                         "measured one-step residual; ~400 forces 2-sigma coverage to 0.95 but "
                         "does so by whitening a deterministic bias (see module docstring)")
    ap.add_argument("--plant-dt", type=float, default=0.002,
                    help="MuJoCo timestep, s. Smaller shrinks the one-step residual but, as "
                         "measured, barely moves coverage")
    ap.add_argument("--no-actearly", action="store_true",
                    help="evaluate actuator force at the start of the step; halves the "
                         "plant/model discrepancy")
    ap.add_argument("--ellipse-gain", type=float, default=200.0,
                    help="magnification of the tip 2-sigma ellipse, for visibility")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    source: SyntheticInput | PotentiometerInput
    if args.pot or args.port or args.simulate:
        source = PotentiometerInput(args.port, args.baud, args.simulate, tuple(args.pot_range))
    else:
        source = SyntheticInput()

    app = QtWidgets.QApplication(sys.argv)
    session = FingerSession(source, gravity=not args.no_gravity,
                            buffer_horizon=args.buffer_horizon, seed=args.seed,
                            psd_alpha=args.psd_alpha, plant_dt=args.plant_dt,
                            actearly=not args.no_actearly)
    source.start()
    window = Window(session, args.speed, args.ellipse_gain)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
