"""Process models must be exact under arbitrary segmentation.

These target the ProcessModel ABC rather than any one implementation, so the
same checks run against every model registered below.
"""

from __future__ import annotations

import numpy as np
import pytest

from erp.models.base import ProcessModel
from erp.models.linear import ConstantAccelModel, JointParams, servoed_finger_model

JOINTS = [
    JointParams(kp=2.0, kv=0.025, damping=0.003, inertia=4e-5, tau=0.004, psd_alpha=1e-2,
                psd_act=1e-8),
    JointParams(kp=0.8, kv=0.010, damping=0.0028, inertia=2e-5, tau=0.004, psd_alpha=1e-2,
                psd_act=1e-8),
]


def finger() -> ProcessModel:
    return servoed_finger_model(JOINTS)


def double_integrator() -> ProcessModel:
    return ConstantAccelModel(n_axes=2, psd=0.5)


MODELS = {"servoed_finger": (finger, 6, 2), "constant_accel": (double_integrator, 4, 1)}


@pytest.fixture(params=sorted(MODELS))
def model_case(request: pytest.FixtureRequest) -> tuple[ProcessModel, int, int]:
    build, nx, nu = MODELS[request.param]
    return build(), nx, nu


def test_mean_composes_across_a_split(model_case: tuple[ProcessModel, int, int]) -> None:
    """predict(a) then predict(b) equals predict(a+b) for a held input.

    Exact rather than approximate because the fusion layer guarantees u is
    constant across each segment, which is what a zero-order-hold
    discretisation assumes.
    """
    model, nx, nu = model_case
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=nx)
    u = rng.normal(size=nu)
    a, b = 3.1e-3, 1.17e-2

    split = model.predict(model.predict(x0, u, a), u, b)
    single = model.predict(x0, u, a + b)
    assert np.allclose(split, single, rtol=1e-11, atol=1e-14)


def test_covariance_composes_across_a_split(model_case: tuple[ProcessModel, int, int]) -> None:
    """The sharpest check available: full belief propagation, both moments.

    If Q were schedule dependent this fails immediately, which is the whole
    reason ProcessModel.Q takes dt rather than being a property.
    """
    model, nx, _ = model_case
    a, b = 3.1e-3, 1.17e-2
    P0 = np.diag(np.linspace(1e-6, 1e-3, nx))
    x = np.zeros(nx)

    F_a, F_b = model.jacobian(x, np.zeros(1), a), model.jacobian(x, np.zeros(1), b)
    split = F_b @ (F_a @ P0 @ F_a.T + model.Q(a)) @ F_b.T + model.Q(b)

    F_ab = model.jacobian(x, np.zeros(1), a + b)
    single = F_ab @ P0 @ F_ab.T + model.Q(a + b)
    assert np.allclose(split, single, rtol=1e-9, atol=1e-20)


def test_schedule_invariance(model_case: tuple[ProcessModel, int, int]) -> None:
    """Same elapsed time, wildly different segmentations, same belief.

    This is the property that lets a sensor be added or re-rated without
    silently retuning the process noise.
    """
    model, nx, nu = model_case
    rng = np.random.default_rng(7)
    # 20 ms: a realistic worst-case gap, set by the slowest sensor in the
    # config (100 Hz encoders) plus margin. See the stiffness note below.
    total = 0.02
    x0, u = rng.normal(size=nx), rng.normal(size=nu)
    P0 = np.diag(np.linspace(1e-6, 1e-3, nx))

    def propagate(cuts: list[float]) -> tuple[np.ndarray, np.ndarray]:
        x, P, t = x0.copy(), P0.copy(), 0.0
        for edge in [*cuts, total]:
            dt = edge - t
            F = model.jacobian(x, u, dt)
            x = model.predict(x, u, dt)
            P = F @ P @ F.T + model.Q(dt)
            t = edge
        return x, P

    x_one, P_one = propagate([])
    x_even, P_even = propagate(list(np.linspace(0, total, 11)[1:-1]))
    # Deliberately lopsided: a 0.1 ms sliver next to a 12 ms stride, which is
    # what an unlucky alignment of a 1 kHz IMU against a 97 Hz encoder produces.
    x_ragged, P_ragged = propagate([0.001, 0.0011, 0.012, 0.0199])

    # Compared normwise, not elementwise: individual covariance entries span
    # ten orders of magnitude here, so elementwise relative error on the
    # smallest of them measures round-off, not correctness.
    #
    # Stiffness note: the servoed finger carries A_c entries of order
    # kp/inertia = 5e4, so ||A_c dt|| grows fast with the step. Up to ~20 ms
    # this composes at machine precision (~1e-16); by 50 ms, scaling-and-
    # squaring inside expm has degraded it to ~1e-7. Prediction gaps are
    # bounded by the slowest sensor, so this is comfortable today -- but a
    # sensor slower than ~20 Hz would need revisiting. See ADR-0001 section 7.
    for x, P in ((x_even, P_even), (x_ragged, P_ragged)):
        assert np.allclose(x, x_one, rtol=1e-9, atol=1e-12)
        assert np.abs(P - P_one).max() <= 1e-12 * np.abs(P_one).max()


def test_process_noise_is_positive_semidefinite(model_case: tuple[ProcessModel, int, int]) -> None:
    model, _, _ = model_case
    for dt in (1e-4, 2e-3, 5e-2):
        Q = model.Q(dt)
        assert np.allclose(Q, Q.T, atol=1e-18)
        assert np.linalg.eigvalsh(Q).min() >= -1e-18


def test_servo_activation_block_matches_the_exact_lag() -> None:
    """The activation is a first-order lag, exactly integrable at any step.

    Carrying it as state is what removes the accelerometer bias, and it costs
    nothing in variable-step accuracy because of this closed form.
    """
    model = finger()
    n, tau = 2, JOINTS[0].tau
    x0 = np.zeros(3 * n)
    x0[2 * n :] = [0.3, -0.2]
    u = np.array([1.0, 0.5])

    for dt in (1e-4, 4e-3, 0.1):
        got = model.predict(x0, u, dt)[2 * n :]
        want = u + (x0[2 * n :] - u) * np.exp(-dt / tau)
        assert np.allclose(got, want, rtol=1e-10, atol=1e-14)


def test_rejects_degenerate_joint_parameters() -> None:
    bad = JointParams(kp=1.0, kv=0.0, damping=0.0, inertia=0.0, tau=0.004, psd_alpha=1.0)
    with pytest.raises(ValueError, match="inertia and tau must be > 0"):
        servoed_finger_model([bad])
    with pytest.raises(ValueError, match="at least one joint"):
        servoed_finger_model([])
