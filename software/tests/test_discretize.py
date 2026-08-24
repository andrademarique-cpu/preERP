"""Discretisation must be exact and schedule invariant.

The double integrator is used as the reference because its zero-order-hold
discretisation is known in closed form, so these compare against analysis
rather than against a second implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from erp.models.discretize import van_loan, zoh_input

# dx/dt = [[0,1],[0,0]] x + [[0],[1]] u,  noise enters on the velocity row.
A_C = np.array([[0.0, 1.0], [0.0, 0.0]])
B_C = np.array([[0.0], [1.0]])
G = np.array([[0.0], [1.0]])
PSD = 7.3


def analytic_F(dt: float) -> np.ndarray:
    return np.array([[1.0, dt], [0.0, 1.0]])


def analytic_Q(dt: float) -> np.ndarray:
    """Continuous white-acceleration noise, integrated exactly over dt."""
    return PSD * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])


@pytest.mark.parametrize("dt", [2e-3, 1.37e-2, 5e-2, 0.25])
def test_van_loan_matches_closed_form(dt: float) -> None:
    F, Q = van_loan(A_C, G, PSD * np.eye(1), dt)
    assert np.allclose(F, analytic_F(dt), rtol=0, atol=1e-15)
    assert np.allclose(Q, analytic_Q(dt), rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("dt", [2e-3, 1.37e-2, 5e-2])
def test_zoh_input_matches_closed_form(dt: float) -> None:
    """B_d for a double integrator is the familiar [dt^2/2, dt]."""
    B_d = zoh_input(A_C, B_C, dt)
    assert np.allclose(B_d.ravel(), [dt**2 / 2, dt], rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize(("a", "b"), [(3e-3, 11e-3), (1e-4, 9.9e-2), (7e-3, 7e-3)])
def test_process_noise_composes_across_a_split(a: float, b: float) -> None:
    """Q(a) then Q(b) must equal Q(a+b).

    This is the property that makes variable-step prediction sound: the
    accumulated noise depends on elapsed time, never on how the interval
    happened to be chopped up by the sensor schedule.
    """
    F_a, Q_a = van_loan(A_C, G, PSD * np.eye(1), a)
    F_b, Q_b = van_loan(A_C, G, PSD * np.eye(1), b)
    _, Q_ab = van_loan(A_C, G, PSD * np.eye(1), a + b)
    assert np.allclose(F_b @ Q_a @ F_b.T + Q_b, Q_ab, rtol=1e-11, atol=1e-20)
    assert np.allclose(F_b @ F_a, van_loan(A_C, G, PSD * np.eye(1), a + b)[0], atol=1e-14)


def test_dwna_does_not_compose() -> None:
    """Guards the reason van_loan exists at all.

    A discrete-white-noise construction accumulates a velocity variance that
    scales with the number of sub-steps, so splitting one interval N ways
    divides it by N. If this ever starts passing, someone has changed what
    "DWNA" means here and the ADR-0001 rationale needs revisiting.
    """
    def dwna_Q(dt: float) -> np.ndarray:
        G_d = np.array([[dt**2 / 2], [dt]])
        return PSD * (G_d @ G_d.T)

    total = 0.02
    variances = []
    for n in (1, 2, 5, 10):
        h = total / n
        P = np.zeros((2, 2))
        for _ in range(n):
            P = analytic_F(h) @ P @ analytic_F(h).T + dwna_Q(h)
        variances.append(P[1, 1])

    assert variances[0] == pytest.approx(10 * variances[-1], rel=1e-9)
    # ... whereas the continuous-time form is invariant to the same split.
    cwna = []
    for n in (1, 2, 5, 10):
        h = total / n
        P = np.zeros((2, 2))
        for _ in range(n):
            F_h, Q_h = van_loan(A_C, G, PSD * np.eye(1), h)
            P = F_h @ P @ F_h.T + Q_h
        cwna.append(P[1, 1])
    assert np.allclose(cwna, cwna[0], rtol=1e-12)


@pytest.mark.parametrize("bad_dt", [0.0, -1e-3])
def test_rejects_nonpositive_dt(bad_dt: float) -> None:
    with pytest.raises(ValueError, match="dt must be > 0"):
        van_loan(A_C, G, PSD * np.eye(1), bad_dt)
    with pytest.raises(ValueError, match="dt must be > 0"):
        zoh_input(A_C, B_C, bad_dt)


def test_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="G must have 2 rows"):
        van_loan(A_C, np.zeros((3, 1)), np.eye(1), 1e-3)
    with pytest.raises(ValueError, match="B_c must have 2 rows"):
        zoh_input(A_C, np.zeros((3, 1)), 1e-3)
