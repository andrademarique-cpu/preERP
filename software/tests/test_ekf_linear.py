"""ADR-0002 P4: the EKF against a Kalman filter written out by hand.

This is the test the whole phase was for. While the EKF held an `MjModel` the
only evidence it was right was that its plots looked plausible on a model
nobody can solve by hand. Handed a linear dynamics its linearisation is exact,
so it *is* a Kalman filter, and a KF's recursion is four lines anyone can write
down and compare against.

No mujoco is imported here, and one test enforces that in a subprocess rather
than trusting `sys.modules` in a session where other files import it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from erp.core.linalg import make_spd, nees_of
from erp.estimators import EKF
from erp.models.linear import LinearDynamics

RNG = np.random.default_rng(20260920)


def closed_form_kf(A, C, Q, R, x0, P0, zs):
    """The textbook recursion, written out. No `erp` import anywhere in here.

    Deliberately NOT Joseph and NOT `make_spd`: this is the independent
    implementation, so it must not share the choices under test. On a
    well-conditioned problem the two forms agree to round-off, which is what
    lets this stand as an oracle.
    """
    x, P = np.asarray(x0, float).copy(), np.asarray(P0, float).copy()
    xs, Ps = [], []
    for z in zs:
        x = A @ x
        P = A @ P @ A.T + Q
        S = C @ P @ C.T + R
        K = P @ C.T @ np.linalg.inv(S)
        x = x + K @ (np.asarray(z, float) - C @ x)
        P = (np.eye(len(x)) - K @ C) @ P
        xs.append(x.copy())
        Ps.append(P.copy())
    return np.array(xs), np.array(Ps)


@pytest.fixture
def constant_velocity():
    """A 1-D constant-velocity model: position observed, velocity inferred.

    Well conditioned on purpose. The badly conditioned case is the Joseph
    falsification further down, and mixing the two would make a failure here
    ambiguous.
    """
    dt = 0.1
    A = np.array([[1.0, dt], [0.0, 1.0]])
    C = np.array([[1.0, 0.0]])
    Q = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]) * 0.01
    R = np.array([[0.25]])
    x0 = np.array([0.0, 1.0])
    P0 = np.diag([1.0, 1.0])
    truth = np.array([[0.3 * k * dt, 0.3] for k in range(40)])
    zs = truth[:, :1] + RNG.normal(0.0, 0.5, size=(40, 1))
    return dt, A, C, Q, R, x0, P0, zs


# --------------------------------------- the obligation

def test_the_ekf_equals_a_closed_form_kf_on_a_linear_model(constant_velocity) -> None:
    dt, A, C, Q, R, x0, P0, zs = constant_velocity
    ekf = EKF(x0, P0, Q, R, LinearDynamics(A, C=C, dt=dt))

    xs, Ps = [], []
    rows = np.array([0])
    for z in zs:
        ekf.predict()
        ekf.update(z, rows)
        xs.append(ekf.x.copy())
        Ps.append(ekf.P.copy())

    xs_ref, Ps_ref = closed_form_kf(A, C, Q, R, x0, P0, zs)
    assert np.allclose(xs, xs_ref, rtol=1e-10, atol=1e-12)
    assert np.allclose(Ps, Ps_ref, rtol=1e-10, atol=1e-12)


def test_the_innovation_and_nis_match_the_hand_computation(constant_velocity) -> None:
    dt, A, C, Q, R, x0, P0, zs = constant_velocity
    ekf = EKF(x0, P0, Q, R, LinearDynamics(A, C=C, dt=dt))
    rows = np.array([0])

    ekf.predict()
    x_pred, P_pred = ekf.x.copy(), ekf.P.copy()
    y, nis = ekf.update(zs[0], rows)

    y_ref = zs[0] - C @ x_pred
    S_ref = C @ P_pred @ C.T + R
    assert y == pytest.approx(y_ref, rel=1e-12)
    assert nis == pytest.approx(float(y_ref @ np.linalg.solve(S_ref, y_ref)), rel=1e-10)


def test_a_perfect_measurement_pins_the_observed_state(constant_velocity) -> None:
    """Sanity with an independent meaning: tiny R must collapse that variance."""
    dt, A, C, Q, _, x0, P0, _ = constant_velocity
    ekf = EKF(x0, P0, Q, np.array([[1e-10]]), LinearDynamics(A, C=C, dt=dt))
    ekf.predict()
    ekf.update(np.array([5.0]), np.array([0]))
    assert ekf.x[0] == pytest.approx(5.0, abs=1e-4)
    assert ekf.P[0, 0] < 1e-9


def test_no_mujoco_is_needed_to_run_the_filter() -> None:
    """Run in a subprocess: other test files import mujoco into this session.

    Checking `sys.modules` inline would pass or fail depending on collection
    order, which is not a check at all.
    """
    script = textwrap.dedent("""
        import sys
        import numpy as np
        from erp.estimators import EKF
        from erp.models.linear import LinearDynamics
        dyn = LinearDynamics(np.eye(2), C=np.array([[1.0, 0.0]]))
        ekf = EKF(np.zeros(2), np.eye(2), np.eye(2) * 0.01, np.array([[0.5]]), dyn)
        ekf.predict()
        ekf.update(np.array([1.0]), np.array([0]))
        assert "mujoco" not in sys.modules, sorted(m for m in sys.modules if "mujoco" in m)
        print("OK", float(ekf.x[0]))
    """)
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("OK")


# --------------------------------------- the falsification

def test_the_plain_covariance_update_loses_positive_definiteness() -> None:
    """Why the EKF uses Joseph, measured rather than asserted by tradition.

    Two states observed through their sum, with a posterior spanning eight
    orders of magnitude and a near-perfect measurement. On this one update:

        plain  (I - KH) P          min eigenvalue  -3.6e-17   <- not a covariance
        Joseph (I-KH)P(I-KH)' + KRK'               +5.0e-19

    Be precise about the size of this: it is one update at `sigma_R = 1e-9`,
    not a filter visibly diverging, and iterating the plain form on this
    problem does *not* blow up. What it produces is a covariance with a
    negative eigenvalue, and that is the failure that matters here, because
    `nees_of` on such a P returns a **negative** NEES -- so the consistency
    diagnostic silently reports something impossible instead of raising.
    """
    P = np.array([[1.0, 0.0], [0.0, 1e-8]])
    H = np.array([[1.0, 1.0]])
    R = np.array([[1e-18]])

    S = H @ P @ H.T + R
    K = np.linalg.solve(S, H @ P).T
    I_KH = np.eye(2) - K @ H
    plain = I_KH @ P
    joseph = I_KH @ P @ I_KH.T + K @ R @ K.T

    eig_plain = float(np.linalg.eigvalsh((plain + plain.T) / 2).min())
    eig_joseph = float(np.linalg.eigvalsh((joseph + joseph.T) / 2).min())

    assert eig_plain < 0.0, "the plain form was supposed to fail here"
    assert eig_plain == pytest.approx(-3.6e-17, rel=0.3)
    assert eig_joseph > 0.0
    assert eig_joseph == pytest.approx(5.0e-19, rel=0.3)

    # The consequence, spelled out: a negative NEES from a negative eigenvalue.
    err = np.linalg.eigh((plain + plain.T) / 2)[1][:, 0] * 1e-8
    assert nees_of(err, plain) < 0.0
    assert nees_of(err, joseph) > 0.0

    # And what the filter actually ships: Joseph *and* make_spd, so even the
    # plain form's output would be floored back to PSD before anyone saw it.
    assert np.linalg.eigvalsh(make_spd(plain)).min() >= 0.0


def test_the_filter_keeps_P_symmetric_and_psd_through_the_bad_case() -> None:
    """The same problem driven through the real `EKF`, which must survive it.

    The symmetry bound is *relative*: `make_spd` symmetrises first but then
    rebuilds as `(V * w) @ V.T`, and that reconstruction is not exactly
    symmetric. Measured here the residual asymmetry is 1.65e-16 of `max|P|` --
    one machine epsilon -- so an absolute bound would only be testing how
    large the entries happen to be.
    """
    eps = np.finfo(float).eps
    dyn = LinearDynamics(np.eye(2), C=np.array([[1.0, 1.0]]))
    ekf = EKF(np.zeros(2), np.diag([1.0, 1e-8]), np.zeros((2, 2)), np.array([[1e-18]]), dyn)
    for _ in range(50):
        ekf.predict()
        ekf.update(np.array([1.0]), np.array([0]))
        scale = float(np.abs(ekf.P).max())
        assert float(np.abs(ekf.P - ekf.P.T).max()) <= 4 * eps * scale
        assert np.linalg.eigvalsh(ekf.P).min() >= 0.0


# --------------------------------------- blindness and the per-update R

def test_the_filter_is_blind_by_signature_not_by_convention() -> None:
    """A project decision, so it is asserted structurally, not trusted.

    Neither `predict` nor `update` takes a control. That is the guarantee: not
    that callers refrain from passing one, but that there is nowhere to put it.
    """
    import inspect

    assert "u" not in inspect.signature(EKF.predict).parameters
    assert "u" not in inspect.signature(EKF.update).parameters
    assert list(inspect.signature(EKF.predict).parameters) == ["self"]

    dyn = LinearDynamics(np.eye(2), B=np.eye(2), C=np.eye(2))
    ekf = EKF(np.zeros(2), np.eye(2), np.eye(2) * 0.01, np.eye(2), dyn)
    assert ekf.u_blind.shape == (2,)
    assert not ekf.u_blind.any()
    before = ekf.u_blind
    ekf.predict()
    assert ekf.u_blind is before, "allocated once, never replaced"
    assert not ekf.u_blind.any()


def test_a_per_update_R_overrides_the_default_block() -> None:
    """ADR-0002 3.2: the sensor's R now reaches the filter."""
    dyn = LinearDynamics(np.eye(2), C=np.eye(2))
    R_default = np.diag([1.0, 1.0])
    rows = np.array([0])

    loose = EKF(np.zeros(2), np.eye(2), np.eye(2) * 0.01, R_default, dyn)
    loose.predict()
    _, nis_default = loose.update(np.array([1.0, 0.0]), rows)

    tight = EKF(np.zeros(2), np.eye(2), np.eye(2) * 0.01, R_default, dyn)
    tight.predict()
    _, nis_tight = tight.update(np.array([1.0, 0.0]), rows, R=np.array([[0.01]]))

    assert nis_tight > nis_default, "a tighter R must make the same residual less likely"
    assert tight.P[0, 0] < loose.P[0, 0]


def test_omitting_R_reproduces_the_pre_p4_behaviour() -> None:
    """The default path must be exactly the old `self.R[ix_(rows, rows)]`."""
    dyn = LinearDynamics(np.eye(2), C=np.eye(2))
    R_full = np.diag([0.25, 4.0])
    rows = np.array([1])

    implicit = EKF(np.zeros(2), np.eye(2), np.eye(2) * 0.01, R_full, dyn)
    implicit.predict()
    y_i, nis_i = implicit.update(np.array([0.0, 2.0]), rows)

    explicit = EKF(np.zeros(2), np.eye(2), np.eye(2) * 0.01, R_full, dyn)
    explicit.predict()
    y_e, nis_e = explicit.update(np.array([0.0, 2.0]), rows, R=R_full[np.ix_(rows, rows)])

    assert np.array_equal(y_i, y_e)
    assert nis_i == nis_e
    assert np.array_equal(implicit.P, explicit.P)


def test_a_wrongly_shaped_R_is_refused() -> None:
    dyn = LinearDynamics(np.eye(2), C=np.eye(2))
    ekf = EKF(np.zeros(2), np.eye(2), np.eye(2) * 0.01, np.eye(2), dyn)
    with pytest.raises(ValueError, match=r"R tiene que ser \(1, 1\)"):
        ekf.update(np.array([1.0, 0.0]), np.array([0]), R=np.eye(2))


def test_a_state_of_the_wrong_size_is_refused() -> None:
    dyn = LinearDynamics(np.eye(3), C=np.eye(3))
    with pytest.raises(ValueError, match="la dinamica pide 3"):
        EKF(np.zeros(2), np.eye(2), np.eye(2), np.eye(2), dyn)
