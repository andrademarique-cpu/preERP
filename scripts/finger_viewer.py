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

Reading the numbers: NEES, not the trajectory
---------------------------------------------
The status bar reports median NEES against the full 6-state MuJoCo truth
(``qpos``, ``qvel``, ``act``), target 6. That is the diagnostic to trust. A
trajectory that looks right and a badly inconsistent filter are the same
picture, which is why coverage and NEES are on screen at all.

**Most of the apparent inconsistency used to be a bug in the diagnostic, not in
the filter.** The belief is valid at ``engine.time``, which trails the plant by
the buffer horizon plus the gap to the last processed event. Scoring it against
truth sampled at ``sim_time`` compared two different instants, and at these
joint rates that mismatch alone exceeds the 2-sigma band it is judged against.
Measured over a 10 s sweep: median NEES **1145 before** the belief was
extrapolated to ``sim_time``, **20.3 after** -- a factor of 56, none of it
physics. With ``--buffer-horizon 0.010`` the old number reached 22587 while the
aligned one stayed at 19.8, i.e. it was measuring lag and nothing else. See
``FusionEngine.belief_at``.

What remains is real, and smaller than previously believed: 20.3 against a
target of 6, so the filter is still roughly 3x overconfident. Contributors,
measured by turning them off:

* **Unmodelled gravity** -- ``servoed_finger_model`` has no gravity term at all.
  ``--no-gravity`` takes NEES from 20.3 to 13.7.
* **Actuator-force convention.** ``erp`` integrates the coupled ``(q, v, a)``
  system exactly, so over a step it uses the *average* servo activation; MuJoCo
  evaluates actuator force at a step endpoint. The difference is
  ``kp * delta_act / inertia``, about 26 rad/s^2, and it is deterministic and
  time-correlated, not white -- so process noise cannot absorb it.
  ``--no-actearly`` halves it; ``--plant-dt 0.00025`` cuts the one-step residual
  tenfold and barely moves coverage, which is what tells you it is a bias.
* **Decoupled joints** -- only ``diag(M)`` is taken; see
  ``joint_params_from_plant``.

Raising ``--psd-alpha`` to ~400 does force coverage to 0.97, but that is
whitening a bias -- the mistake ``docs/theory/finger_imu_ekf.md`` section 4
documents, where inflating the noise flattens NIS while leaving NEES untouched.
Note the same trap exists on the ``R`` side: enlarging ``SIG_GYRO`` widens the
band and flatters coverage without improving the estimate at all.

By default the filter's model IS the plant
------------------------------------------
``joint_params_from_plant`` reads inertia out of MuJoCo's own mass matrix and
copies the servo gains from the dict that generated the XML, the filter starts
from exact truth, and 4 of the 6 states are measured directly with an ``R`` that
is exactly right. Nothing here can go badly wrong, so good tracking is evidence
about the plumbing, not about the estimator. The ``error injection`` flags exist
to remove that guarantee -- ``--model-error 20`` roughly doubles NEES and
``--model-error 50`` takes it to 230 with coverage collapsing to 0.11.

``--unknown-input``: the deployment case
----------------------------------------
The default also hands the filter the commanded servo angle, which real hardware
does not -- the setpoint lives inside the servo's own controller. With
``--unknown-input`` the estimator never sees ``u``. The activation stops being
driven and becomes a random walk recovered from the observed motion, which works
because it is observable from the encoder and gyro alone.

Since ``tau`` is short next to the command's timescale, the estimated activation
*is* the recovered command: the dotted ``U est`` trace on the joint plots is a
command the filter was never given. Measured over a 10 s sweep it sits 0.21 deg
from the true command, against a 60 deg travel.

The surprise is that it comes out **more** consistent, not less: median NEES 4.3
against the input-driven filter's 20.3, with coverage 0.88 against 0.37. The
random walk's process noise also absorbs the unmodelled gravity and
actuator-convention bias that makes the input-driven filter overconfident.
Accuracy still degrades, which is the honest cost -- joint error 0.078 deg
against 0.047 deg.

``--psd-act`` is load-bearing here and defaults differently per mode. Carrying
the known-input jitter value into this mode is the specific trap: at 1e-6 the
blind filter reads NEES 77910, coverage 0.01, and the recovered command lags the
real one by 238 ms.

Run::

    python scripts/finger_viewer.py                    # synthetic input
    python scripts/finger_viewer.py --pot --simulate   # pot GUI stack, no hardware
    python scripts/finger_viewer.py --pot --port COM5  # real potentiometers
    python scripts/finger_viewer.py --model-error 20   # make the filter earn it
    python scripts/finger_viewer.py --unknown-input    # as deployed: u withheld
    python scripts/finger_viewer.py --sensor-latency 0.005 0.001 --buffer-horizon 0.01

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
from dataclasses import dataclass, field, replace
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

from erp.core.linalg import make_spd, nees  # noqa: E402
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
SIG_GYRO = 9e-2                  # rate gyro noise, rad/s
PSD_ALPHA = 4.0                  # angular-acceleration disturbance PSD, rad^2/s^3
PSD_ACT = 1e-6                   # servo activation jitter PSD, rad^2/s
# With --unknown-input, psd_act stops being jitter around a known setpoint and
# has to cover the whole unknown actuation, so it is ~3 orders larger. Inheriting
# PSD_ACT there leaves the filter 959x overconfident; see test_consistency.
PSD_ACT_BLIND = 8e-4             # activation random-walk PSD, rad^2/s
F_INPUT, F_ENC, F_GYRO = 200.0, 100.0, 250.0

HISTORY_SECONDS = 12.0
UI_PERIOD_MS = 33
COLOR_U, COLOR_TRUE, COLOR_EST = "#888888", "#4c8fd8", "#e08a3c"
COLOR_UEST = "#2e8b57"           # recovered command, when the filter is not told it


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
    return template.format(**fields)


def joint_params_from_plant(model: mj.MjModel, data: mj.MjData,
                            psd_alpha: float = PSD_ALPHA,
                            psd_act: float = PSD_ACT) -> list[JointParams]:
    """Derive the estimator's linear model from the compiled plant.

    Effective inertia is read out of MuJoCo's own mass matrix at the neutral
    pose rather than guessed, so the filter's model and the plant agree on the
    one number that is hardest to eyeball. Only the diagonal is taken: the
    linear model treats the joints as decoupled, which is a real approximation,
    but reflected rotor inertia (4e-5) dominates the inter-link coupling
    (~3e-6) by an order of magnitude here.

    That rotor inertia is declared on the ``<general>`` actuator, not on the
    joint, so ``dof_armature`` reads zero -- but MuJoCo folds actuator armature
    (through ``gear^2``) into the mass matrix, which is why ``M`` is the right
    place to read it from and ``dof_armature`` is not.
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
            tau=act["tau"], psd_alpha=psd_alpha, psd_act=psd_act,
        ))
    return out


def perturb_joint_params(joints: list[JointParams], pct: float,
                         rng: np.random.Generator) -> list[JointParams]:
    """Give the *filter* a model that differs from the plant by +/- ``pct`` percent.

    Without this the viewer cannot fail. ``joint_params_from_plant`` reads
    inertia out of MuJoCo's own mass matrix and copies kp/kv/tau/damping from
    the dict that generated the XML, so the filter's ``A_c`` equals the plant's
    to machine precision -- a luxury no real system has, where inertia is
    rarely known better than ~10 %. Tracking that looks excellent under a
    perfectly matched model is evidence about the plumbing, not the estimator.

    Perturbs inertia, kp, kv and damping independently. ``tau`` is left alone:
    the servo time constant is the one parameter a datasheet actually pins down.
    """
    if pct <= 0.0:
        return joints
    def jitter() -> float:
        return 1.0 + float(rng.uniform(-pct, pct)) / 100.0
    return [
        replace(jp, inertia=jp.inertia * jitter(), kp=jp.kp * jitter(),
                kv=jp.kv * jitter(), damping=jp.damping * jitter())
        for jp in joints
    ]


def sample_gaussian(rng: np.random.Generator, cov: np.ndarray) -> np.ndarray:
    """Draw a zero-mean sample from ``cov``; robust where it is near-singular."""
    return np.asarray(np.linalg.cholesky(make_spd(cov)) @ rng.normal(size=cov.shape[0]))


def quantize(z: np.ndarray, span: float, bits: int) -> np.ndarray:
    """Round ``z`` onto a ``bits``-resolution grid spanning ``span`` radians.

    A potentiometer's dominant error is quantization, which is uniform and
    correlated with the signal -- not the white Gaussian the filter's ``R``
    assumes. Turning this on is therefore a way to make ``R`` wrong on purpose,
    in the direction real hardware is wrong.
    """
    step = span / (2**bits - 1)
    return np.asarray(np.round(z / step) * step)


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

    ``latency`` models transport delay: a sample taken at ``t`` is stamped ``t``
    -- the timestamp is always the *sample* instant -- but does not become
    readable until ``t + latency``. That split is the whole reason
    ``Measurement.timestamp`` and arrival time are separate concepts, and with
    a non-zero value here the engine's ``buffer_horizon`` and ``discarded``
    counters finally do something.
    """

    def __init__(self, model: MeasurementModel, dim: int, frame_id: str,
                 latency: float = 0.0) -> None:
        self._model, self._dim, self._frame_id = model, dim, frame_id
        self._latency = float(latency)
        self._pending: list[tuple[float, Measurement]] = []
        self._now = 0.0

    def set_clock(self, t: float) -> None:
        """Tell the sensor what time it is, so ``drain`` can withhold late samples."""
        self._now = float(t)

    def emit(self, z: np.ndarray, t: float) -> None:
        self._pending.append(
            (t + self._latency, Measurement(z=z, timestamp=t, frame_id=self._frame_id))
        )

    def read(self) -> Measurement | None:
        for i, (arrival, m) in enumerate(self._pending):
            if arrival <= self._now:
                del self._pending[i]
                return m
        return None

    def drain(self) -> list[Measurement]:
        ready = [m for arrival, m in self._pending if arrival <= self._now]
        self._pending = [(a, m) for a, m in self._pending if a > self._now]
        return ready

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
    nees: list[float] = field(default_factory=list)
    # Estimated servo activation. In unknown-input mode this is the recovered
    # command: `a` tracks `u` through a tau=4ms lag, so plotting it against the
    # true U shows the filter reconstructing an input it was never given.
    u_est: list[np.ndarray] = field(default_factory=list)

    def append(self, t: float, u: np.ndarray, q_true: np.ndarray,
               q_est: np.ndarray, sigma: np.ndarray, nees_val: float,
               u_est: np.ndarray) -> None:
        self.t.append(t)
        self.u.append(u.copy())
        self.q_true.append(q_true.copy())
        self.q_est.append(q_est.copy())
        self.sigma.append(sigma.copy())
        self.nees.append(nees_val)
        self.u_est.append(u_est.copy())

    def trim(self, horizon: float) -> None:
        while self.t and self.t[-1] - self.t[0] > horizon:
            for seq in (self.t, self.u, self.q_true, self.q_est, self.sigma,
                        self.nees, self.u_est):
                seq.pop(0)

    def arrays(self) -> tuple[np.ndarray, ...]:
        return (np.asarray(self.t), np.asarray(self.u), np.asarray(self.q_true),
                np.asarray(self.q_est), np.asarray(self.sigma), np.asarray(self.u_est))


class FingerSession:
    """Drives the MuJoCo plant and the erp estimator on one clock."""

    def __init__(self, source: SyntheticInput | PotentiometerInput,
                 *, gravity: bool, buffer_horizon: float, seed: int,
                 psd_alpha: float = PSD_ALPHA, plant_dt: float = 0.002,
                 actearly: bool = True, model_error: float = 0.0,
                 init_error: bool = False, enc_bits: int = 0,
                 gyro_bias: float = 0.0,
                 sensor_latency: tuple[float, float] = (0.0, 0.0),
                 pot_span: float = 120.0, known_input: bool = True,
                 psd_act: float | None = None) -> None:
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
        self.enc_bits = enc_bits
        self.gyro_bias = gyro_bias
        self.pot_span = np.deg2rad(pot_span)

        self.known_input = known_input
        if psd_act is None:
            psd_act = PSD_ACT if known_input else PSD_ACT_BLIND

        # The plant's own parameters; the filter gets a perturbed copy so that a
        # matched model is a deliberate choice rather than an accident.
        # Drawn from a SEPARATE stream: sharing self.rng would make the sensor
        # noise realisation depend on whether --model-error was passed, so the
        # knob would move two things at once and nothing could be attributed.
        self.plant_joints = joint_params_from_plant(self.model, self.data, psd_alpha, psd_act)
        self.process = servoed_finger_model(
            perturb_joint_params(self.plant_joints, model_error,
                                 np.random.default_rng(seed + 1)),
            known_input=known_input)
        P0 = np.diag([1e-6, 1e-6, 1e-4, 1e-4, 1e-6, 1e-6])
        x0 = sample_gaussian(self.rng, P0) if init_error else np.zeros(6)
        self.ekf = ExtendedKalmanFilter(self.process, GaussianState(x0, P0))

        # Per-sensor latency, not one shared value: equal latencies preserve
        # arrival order, so nothing ever arrives late relative to the filter and
        # engine.discarded stays empty no matter how large they are. It is the
        # *difference* between sensors that produces reordering, which is the
        # thing buffer_horizon exists to absorb.
        lat_enc, lat_gyro = sensor_latency
        self.enc = QueueSensor(joint_block_model(2, "angle", SIG_ENC), 2, "joint",
                               latency=lat_enc)
        self.gyro = QueueSensor(joint_block_model(2, "rate", SIG_GYRO), 2, "joint",
                                latency=lat_gyro)

        # Two histories when the command is withheld. `self.inputs` is the real
        # one -- it drives the plant and feeds the U trace. `filter_inputs` is
        # what the estimator is allowed to see: a single zero, held forever.
        # Not an empty history, because FusionEngine still queries u_at at every
        # event and u_at raises rather than fabricating actuation that never
        # happened. With B_c = 0 the value is ignored; what matters is that it
        # yields no breakpoints, so intervals split at measurements only.
        self.inputs = InputHistory()
        self.inputs.push(ControlInput(u=source.command(0.0).copy(), timestamp=0.0))
        if known_input:
            self.filter_inputs = self.inputs
        else:
            self.filter_inputs = InputHistory()
            self.filter_inputs.push(ControlInput(u=np.zeros(2), timestamp=0.0))
        self.engine = FusionEngine(
            self.ekf, {"enc": self.enc, "gyro": self.gyro}, self.filter_inputs,
            buffer_horizon=buffer_horizon, t0=0.0,
        )

        self.trace = Trace()
        self._next = {"input": 1.0 / F_INPUT, "enc": 1.0 / F_ENC, "gyro": 1.0 / F_GYRO}
        self._u = self.inputs.u_at(0.0).copy()
        self.belief = self.ekf.state.copy()

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
                if self.enc_bits:
                    z = quantize(np.asarray(self.data.qpos[:2]), self.pot_span, self.enc_bits)
                else:
                    z = self.data.qpos[:2] + self.rng.normal(0, SIG_ENC, 2)
                self.enc.emit(z, t)
                self._next["enc"] += 1.0 / F_ENC
            if t >= self._next["gyro"]:
                z = (self.data.qvel[:2] + self.gyro_bias
                     + self.rng.normal(0, SIG_GYRO, 2))
                self.gyro.emit(z, t)
                self._next["gyro"] += 1.0 / F_GYRO

            self.data.ctrl[:] = self._u
            mj.mj_step(self.model, self.data)

        now = self.sim_time
        for sensor in (self.enc, self.gyro):
            sensor.set_clock(now)
        self.engine.step(now)
        # The filter's belief is valid at engine.time, which trails `now` by the
        # buffer horizon plus the gap to the last processed event. Scoring it
        # against truth sampled at `now` would compare two different instants --
        # at these joint rates that mismatch alone can exceed the 2-sigma band
        # it is being judged against, and it reads as filter inconsistency while
        # actually being an error in the diagnostic.
        self.belief = self.engine.belief_at(now)
        self._record()
        self.inputs.prune_before(self.engine.time - 1.0)

    @property
    def lag(self) -> float:
        """Seconds the filter's own belief trails the plant by, before extrapolation."""
        return self.sim_time - self.engine.time

    def x_true(self) -> np.ndarray:
        """Full 6-state truth from the plant, in the model's blocked layout.

        MuJoCo carries all of it: ``qpos``/``qvel`` for the joints and ``act``
        for the servo activations, which exist because the actuator is compiled
        with ``dyntype="filterexact"``. The activation block *is* the model's
        ``a``: ``a' = (u - a)/tau`` matches ``filterexact`` with ``dynprm=tau``,
        and ``force = kp(a - q) - kv v`` matches ``A_c[iv, ia] = kp/inertia``.

        Having all six means NEES can be reported against the whole state, which
        is the diagnostic CLAUDE.md asks for -- unlike per-joint coverage it
        exercises the cross-terms, and unlike coverage it cannot be flattered by
        inflating R.
        """
        return np.concatenate([self.data.qpos[:2], self.data.qvel[:2], self.data.act[:2]])

    def _record(self) -> None:
        state = self.belief
        self.trace.append(
            self.sim_time, self._u, np.asarray(self.data.qpos[:2]),
            state.x[:2], np.sqrt(np.diag(state.P)[:2]),
            nees(self.x_true(), state.x, state.P),
            state.x[4:],                     # estimated activation = recovered command
        )
        self.trace.trim(HISTORY_SECONDS)

    def tip(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(true_tip_yz, est_tip_yz, tip_covariance)``, all at ``sim_time``."""
        state = self.belief
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
                # The estimated servo activation, in the same units as the
                # command. Only shown when the filter was never told U -- with
                # the command available it would just retrace the dashed line.
                "u_est": plot.plot(
                    pen=pg.mkPen(COLOR_UEST, width=2, style=QtCore.Qt.DotLine),
                    name="U est (recovered)"),
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
        t, u, q_true, q_est, sigma, u_est = self.session.trace.arrays()
        if t.size < 2:
            return
        deg = np.degrees

        for j, curves in enumerate(self.joint_curves):
            curves["u"].setData(t, deg(u[:, j]))
            curves["true"].setData(t, deg(q_true[:, j]))
            curves["est"].setData(t, deg(q_est[:, j]))
            if self.session.known_input:
                curves["u_est"].setData([], [])
            else:
                curves["u_est"].setData(t, deg(u_est[:, j]))

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
        # Median, not mean: NEES is heavy-tailed, driven by brief stretches where
        # P is small. Target is dim(x) = 6.
        med_nees = float(np.median(self.session.trace.nees))
        mode = "u known" if self.session.known_input else "u WITHHELD"
        self.status.showMessage(
            f"sim {self.session.sim_time:7.2f} s  |  {mode}"
            f"  |  lag {1e3 * self.session.lag:4.1f} ms"
            f"  |  NEES {med_nees:7.2f} (want 6)"
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
    ap.add_argument("--unknown-input", action="store_true",
                    help="withhold the servo command from the FILTER, as on real hardware "
                         "where the setpoint lives inside the servo's controller. The "
                         "activation becomes a random walk estimated from motion alone; the "
                         "'U est' trace is that estimate, i.e. the recovered command")
    ap.add_argument("--psd-act", type=float, default=None,
                    help=f"activation noise PSD, rad^2/s. Defaults to {PSD_ACT:g} with the "
                         f"command known and {PSD_ACT_BLIND:g} with --unknown-input, where it "
                         f"must cover the whole unknown actuation rather than jitter around a "
                         f"known setpoint. Inheriting the smaller value leaves the blind "
                         f"filter ~959x overconfident")

    # Error injection. All default to off, so the numbers quoted in the module
    # docstring stay reproducible. With every one of them off the filter's model
    # IS the plant and the initial state is exactly true, which is why the
    # defaults track so well -- see perturb_joint_params.
    err = ap.add_argument_group("error injection (default: none, i.e. a perfectly matched filter)")
    err.add_argument("--model-error", type=float, default=0.0, metavar="PCT",
                     help="perturb the FILTER's inertia/kp/kv/damping by +/- PCT percent "
                          "relative to the plant. The single biggest missing error source. "
                          "Measured from a 20.3 baseline: 20 roughly doubles NEES, 50 takes "
                          "it to ~230 and collapses coverage to 0.11")
    err.add_argument("--init-error", action="store_true",
                     help="start the filter from a sample of P0 instead of exact truth, "
                          "so there is an acquisition transient")
    err.add_argument("--enc-bits", type=int, default=0, metavar="N",
                     help="quantize the encoder to N bits over --pot-range instead of adding "
                          "Gaussian noise, making R wrong the way a real pot makes it wrong")
    err.add_argument("--gyro-bias", type=float, default=0.0, metavar="RAD_S",
                     help="constant gyro bias, rad/s -- the unmodelled error a rate sensor "
                          "actually has")
    err.add_argument("--sensor-latency", type=float, nargs=2, default=(0.0, 0.0),
                     metavar=("ENC_S", "GYRO_S"),
                     help="hold encoder / gyro samples this many seconds before the engine "
                          "can read them. config/estimation.yaml declares 0.005 and 0.001. "
                          "UNEQUAL values are what produce reordering, and so what makes "
                          "--buffer-horizon and the dropped counter mean anything")
    args = ap.parse_args()

    source: SyntheticInput | PotentiometerInput
    if args.pot or args.port or args.simulate:
        source = PotentiometerInput(args.port, args.baud, args.simulate, tuple(args.pot_range))
    else:
        source = SyntheticInput()

    app = QtWidgets.QApplication(sys.argv)
    lo, hi = args.pot_range
    session = FingerSession(source, gravity=not args.no_gravity,
                            buffer_horizon=args.buffer_horizon, seed=args.seed,
                            psd_alpha=args.psd_alpha, plant_dt=args.plant_dt,
                            actearly=not args.no_actearly,
                            model_error=args.model_error, init_error=args.init_error,
                            enc_bits=args.enc_bits, gyro_bias=args.gyro_bias,
                            sensor_latency=tuple(args.sensor_latency), pot_span=hi - lo,
                            known_input=not args.unknown_input, psd_act=args.psd_act)
    source.start()
    window = Window(session, args.speed, args.ellipse_gain)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
