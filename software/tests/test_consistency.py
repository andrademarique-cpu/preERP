"""Multi-rate NEES/NIS consistency, plus the falsifications that give it teeth.

Filter correctness here is judged by consistency statistics, never by eyeballing
a trajectory: the trajectories of a good and a badly-inconsistent filter look
equally plausible plotted.

Every deliberately-broken variant below must FAIL. A consistency test that
cannot fail is not evidence of anything, so the broken filters are asserted on
just as hard as the correct one.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

from erp.core.linalg import make_spd, nees
from erp.core.timeline import InputHistory
from erp.core.types import Array, ControlInput, GaussianState, Measurement
from erp.estimators.ekf import ExtendedKalmanFilter
from erp.fusion.engine import FusionEngine
from erp.models.base import ProcessModel
from erp.models.linear import JointParams, servoed_finger_model
from erp.models.measurement import joint_block_model
from erp.sensors.replay import ReplaySensor

JOINTS = [
    JointParams(kp=2.0, kv=0.025, damping=0.003, inertia=4e-5, tau=0.004,
                psd_alpha=4.0, psd_act=1e-6),
    JointParams(kp=0.8, kv=0.010, damping=0.0028, inertia=2e-5, tau=0.004,
                psd_alpha=4.0, psd_act=1e-6),
]
NX, NJ = 6, 2
T_END = 0.30
F_INPUT, F_ENC, F_GYRO = 200.0, 97.0, 253.0
SIG_ENC, SIG_GYRO = 1.7e-3, 5e-3
P0 = np.diag([1e-6, 1e-6, 1e-4, 1e-4, 1e-6, 1e-6])
N_RUNS = 20

ENC = joint_block_model(NJ, "angle", SIG_ENC)
GYRO = joint_block_model(NJ, "rate", SIG_GYRO)
DT_NOMINAL = 1.0 / F_GYRO


def build_inputs() -> InputHistory:
    """Raised-cosine setpoints, so angle and rate both start at zero."""
    h = InputHistory()
    for t in np.arange(0.0, T_END, 1.0 / F_INPUT):
        u = np.array([
            np.deg2rad(45) / 2 * (1 - np.cos(2 * np.pi * 0.5 * t)),
            np.deg2rad(60) / 2 * (1 - np.cos(2 * np.pi * 0.3 * t)),
        ])
        h.push(ControlInput(u=u, timestamp=float(t)))
    return h


def sample_gaussian(rng: np.random.Generator, cov: Array) -> Array:
    """Draw a zero-mean sample; robust where cov is near-singular."""
    return np.asarray(np.linalg.cholesky(make_spd(cov)) @ rng.normal(size=cov.shape[0]))


def measurement_schedule() -> list[tuple[float, str]]:
    """Two sensors on deliberately non-harmonic rates, so segments vary widely."""
    enc = [(float(t), "enc") for t in np.arange(0.005, T_END, 1.0 / F_ENC)]
    gyro = [(float(t), "gyro") for t in np.arange(0.003, T_END, 1.0 / F_GYRO)]
    return sorted(enc + gyro)


def simulate(rng: np.random.Generator, inputs: InputHistory) -> tuple[
    list[tuple[float, str]], dict[int, Array], dict[str, list[Measurement]]
]:
    """Generate a truth path and the measurements taken along it.

    Truth is propagated along the same event timeline the filter will use.
    That is legitimate precisely because the process noise composes exactly:
    a path built from many short segments has the same law as one built from
    few long ones.
    """
    model = servoed_finger_model(JOINTS)
    schedule = measurement_schedule()

    x = sample_gaussian(rng, P0)
    t = 0.0
    truth: dict[int, Array] = {}
    meas: dict[str, list[Measurement]] = {"enc": [], "gyro": []}

    for i, (t_m, name) in enumerate(schedule):
        for edge in [*inputs.breakpoints_in(t, t_m), t_m]:
            dt = edge - t
            if dt <= 0.0:
                continue
            x = model.predict(x, inputs.u_at(t), dt) + sample_gaussian(rng, model.Q(dt))
            t = edge
        truth[i] = x.copy()
        mm = ENC if name == "enc" else GYRO
        z = mm.h(x, inputs.u_at(t_m)) + sample_gaussian(rng, mm.R)
        meas[name].append(Measurement(z=z, timestamp=t_m, frame_id="joint"))

    return schedule, truth, meas


def run_filter(model: ProcessModel, seed: int) -> tuple[list[float], list[float]]:
    """Run one Monte Carlo realisation; return per-measurement NEES and NIS."""
    rng = np.random.default_rng(seed)
    inputs = build_inputs()
    schedule, truth, meas = simulate(rng, inputs)

    ekf = ExtendedKalmanFilter(model, GaussianState(np.zeros(NX), P0.copy()))
    engine = FusionEngine(
        ekf,
        {"enc": ReplaySensor(meas["enc"], ENC), "gyro": ReplaySensor(meas["gyro"], GYRO)},
        inputs,
        buffer_horizon=0.0,
    )

    nees_vals: list[float] = []
    nis_vals: list[float] = []
    for i, (t_m, _) in enumerate(schedule):
        engine.step(t_m)
        nees_vals.append(nees(truth[i], engine.state.x, engine.state.P))
        if ekf.last_nis is not None:
            nis_vals.append(ekf.last_nis)
    assert engine.discarded == {}, "replay must never drop a measurement"
    return nees_vals, nis_vals


def median_nees(model: ProcessModel) -> float:
    """Median NEES pooled over independent runs.

    The median, not the mean: the distribution is heavy-tailed, driven by brief
    stretches where P is small. A serious ANEES needs Monte Carlo, which is
    what the run loop provides.
    """
    pooled: list[float] = []
    for seed in range(N_RUNS):
        pooled.extend(run_filter(model, seed)[0])
    return float(np.median(pooled))


class ConstantQModel(ProcessModel):
    """Returns one fixed Q regardless of interval.

    This is exactly what a dt-less ``Q`` property would force an implementation
    to do, so this test is the concrete justification for ADR-0001 D1.
    """

    def __init__(self, base: ProcessModel, dt_nominal: float) -> None:
        self._base, self._Q = base, base.Q(dt_nominal)

    def predict(self, x: Array, u: Array, dt: float) -> Array:
        return self._base.predict(x, u, dt)

    def jacobian(self, x: Array, u: Array, dt: float) -> Array:
        return self._base.jacobian(x, u, dt)

    def Q(self, dt: float) -> Array:
        return self._Q


class DwnaQModel(ProcessModel):
    """Discrete white-noise acceleration, tuned to match at one nominal rate.

    Represents the reasonable-looking mistake: calibrate the noise at the
    design tick, then run it on a multi-rate schedule.
    """

    def __init__(self, base: ProcessModel, dt_nominal: float, psd: float) -> None:
        self._base = base
        self._sigma2 = psd / dt_nominal      # matches CWNA velocity variance at dt_nominal

    def predict(self, x: Array, u: Array, dt: float) -> Array:
        return self._base.predict(x, u, dt)

    def jacobian(self, x: Array, u: Array, dt: float) -> Array:
        return self._base.jacobian(x, u, dt)

    def Q(self, dt: float) -> Array:
        G = np.zeros((NX, NJ))
        G[:NJ, :] = np.eye(NJ) * dt**2 / 2
        G[NJ : 2 * NJ, :] = np.eye(NJ) * dt
        return G @ (self._sigma2 * np.eye(NJ)) @ G.T + np.eye(NX) * 1e-18


# Band on the pooled median NEES, as a fraction of the chi-squared median for
# nx degrees of freedom. Calibrated empirically: over five disjoint 20-run
# blocks the reference filter lands in 0.967-1.016 of target, so 0.85-1.20
# leaves comfortable margin without being so loose it stops discriminating.
CONSISTENT_BAND = (0.85 * chi2.median(NX), 1.20 * chi2.median(NX))


def test_multirate_filter_is_consistent() -> None:
    """The reference filter must sit inside the chi-squared band."""
    med = median_nees(servoed_finger_model(JOINTS))
    lo, hi = CONSISTENT_BAND
    assert lo <= med <= hi, f"median NEES {med:.3f} outside [{lo:.3f}, {hi:.3f}] for nx={NX}"


def test_multirate_innovations_are_consistent() -> None:
    """NIS targets dim(z) = 2; needs no ground truth, so it survives onto hardware."""
    pooled: list[float] = []
    for seed in range(5):
        pooled.extend(run_filter(servoed_finger_model(JOINTS), seed)[1])
    med = float(np.median(pooled))
    assert 0.4 * chi2.median(NJ) <= med <= 2.5 * chi2.median(NJ), f"median NIS {med:.3f}"


def test_constant_process_noise_fails_consistency() -> None:
    """Falsification: a dt-independent Q, which is what a Q property forces.

    It fails, but it is worth being precise about *how*. On this schedule it
    lands around 0.69 of target: biased low, i.e. conservative, because the
    nominal dt it was tuned at exceeds the median segment. That is a much
    milder failure than the DWNA case below, and the size of it depends on the
    sensor mix rather than on anything intrinsic.

    So the argument for ADR-0001 D1 is that a constant Q cannot be *correct*,
    only conservatively wrong by a schedule-dependent amount -- not that it
    blows up. Overstating this would be the easiest way to get the ADR's
    reasoning quietly disbelieved later.
    """
    med = median_nees(ConstantQModel(servoed_finger_model(JOINTS), DT_NOMINAL))
    lo, hi = CONSISTENT_BAND
    assert not (lo <= med <= hi), (
        f"constant-Q filter scored median NEES {med:.3f}, inside the consistent band "
        f"[{lo:.3f}, {hi:.3f}] -- the consistency test is too weak to detect it"
    )
    assert med < lo, f"expected constant-Q to be conservative (low), got {med:.3f}"


def test_dwna_process_noise_fails_catastrophically() -> None:
    """Falsification: schedule-dependent Q, the failure ADR-0001 D3 exists to stop.

    This is the severe one: the accumulated noise scales with how finely the
    interval happens to be chopped, so the filter grows overconfident for
    reasons unrelated to the physics. Lands five orders of magnitude above
    target, matching the mismatched-model result in
    ``docs/theory/finger_imu_ekf.md`` section 4.
    """
    med = median_nees(DwnaQModel(servoed_finger_model(JOINTS), DT_NOMINAL, JOINTS[0].psd_alpha))
    assert med > 100 * NX, f"expected catastrophic overconfidence, got median NEES {med:.3f}"


@pytest.mark.parametrize("scale", [1e-3, 1e3])
def test_mis_scaled_process_noise_fails_consistency(scale: float) -> None:
    """Sanity floor: grossly wrong Q must be caught in both directions."""

    class ScaledQ(ProcessModel):
        def __init__(self, base: ProcessModel) -> None:
            self._base = base

        def predict(self, x: Array, u: Array, dt: float) -> Array:
            return self._base.predict(x, u, dt)

        def jacobian(self, x: Array, u: Array, dt: float) -> Array:
            return self._base.jacobian(x, u, dt)

        def Q(self, dt: float) -> Array:
            return np.asarray(self._base.Q(dt) * scale)

    med = median_nees(ScaledQ(servoed_finger_model(JOINTS)))
    lo, hi = CONSISTENT_BAND
    assert not (lo <= med <= hi), f"scale={scale} scored median NEES {med:.3f}"
