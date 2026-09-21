"""ADR-0002 P4: `MujocoDynamics` is the f/F/h/H path, paired and unchanged.

The phase's promise is that the EKF stops holding an `MjModel` without any
number moving. That rests entirely on one claim -- that pairing `(x_next, F)`
and `(z, H)` into a single `MjData` load computes exactly what two loads did --
so it is asserted with `array_equal`, not `allclose`.

It also pins the trap. `mjd_transitionFD` restores the *state* but leaves
`sensordata` at its last perturbation, so the obvious implementation of
`observe` is wrong by an amount small enough to look like a tolerance problem.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from erp.io.paths import resolve_repo_path
from erp.models.base import DiscreteDynamics
from erp.sim.dynamics import MujocoDynamics
from erp.sim.mujoco import F_dyn, H_dyn, f_dyn, h_dyn, state_dim
from erp.sim.plant import blind_variant, warmup_to_rest

pytestmark = pytest.mark.mujoco

XML = "mechanical/mujoco_assets/MyPalletizer260/MyPalletizer260.xml"


@pytest.fixture(scope="module")
def blind() -> tuple[Any, Any]:
    return blind_variant(resolve_repo_path(XML))


@pytest.fixture(scope="module")
def dyn(blind) -> MujocoDynamics:
    return MujocoDynamics(*blind)


@pytest.fixture(scope="module")
def x_rest(blind) -> np.ndarray:
    model, data = blind
    q = np.deg2rad([10.0, 20.0, 30.0])
    return np.r_[warmup_to_rest(model, data, q), q]


# ------------------------------------------------- the contract

def test_mujoco_dynamics_satisfies_the_protocol(dyn: MujocoDynamics, blind) -> None:
    model, _ = blind
    checked: DiscreteDynamics = dyn  # a type error here is the real assertion
    assert checked.nx == state_dim(model) == 11
    assert checked.nz == model.nsensordata == 15
    assert checked.nu == model.nu == 3
    assert checked.dt == pytest.approx(0.002)


def test_the_shapes_are_what_the_filter_expects(dyn: MujocoDynamics, x_rest) -> None:
    u = np.zeros(dyn.nu)
    x_next, F = dyn.step(x_rest, u)
    z, H = dyn.observe(x_rest, u)
    assert x_next.shape == (dyn.nx,)
    assert F.shape == (dyn.nx, dyn.nx)
    assert z.shape == (dyn.nz,)
    assert H.shape == (dyn.nz, dyn.nx)
    for a in (x_next, F, z, H):
        assert a.dtype == np.float64


# ---------------------------------- the claim the whole phase rests on

@pytest.mark.parametrize("q_deg", [(0.0, 0.0, 0.0), (10.0, 20.0, 30.0), (-30.0, 5.0, 45.0)])
def test_step_is_bit_identical_to_F_dyn_then_f_dyn(blind, q_deg) -> None:
    """Not `allclose` -- equal. A moved digit moves the golden run."""
    model, data = blind
    q = np.deg2rad(q_deg)
    x = np.r_[warmup_to_rest(model, data, q), q]
    u = np.zeros(model.nu)

    F_ref = F_dyn(x, u, model, data)
    x_ref = f_dyn(x, u, model, data)
    x_next, F = MujocoDynamics(model, data).step(x, u)

    assert np.array_equal(F, F_ref)
    assert np.array_equal(x_next, x_ref)


@pytest.mark.parametrize("q_deg", [(0.0, 0.0, 0.0), (10.0, 20.0, 30.0), (-30.0, 5.0, 45.0)])
def test_observe_is_bit_identical_to_H_dyn_then_h_dyn(blind, q_deg) -> None:
    model, data = blind
    q = np.deg2rad(q_deg)
    x = np.r_[warmup_to_rest(model, data, q), q]
    u = np.zeros(model.nu)

    H_ref = H_dyn(x, u, model, data)
    z_ref = h_dyn(x, u, model, data)
    z, H = MujocoDynamics(model, data).observe(x, u)

    assert np.array_equal(H, H_ref)
    assert np.array_equal(z, z_ref)


def test_the_jacobian_comes_from_the_state_before_the_step(dyn: MujocoDynamics, x_rest) -> None:
    """`F` is df/dx at the *input* x, which is what makes it the right F.

    Pairing exists partly to make this unmissable: with separate calls nothing
    stops `f` being applied first and `F` then evaluated at the new state. The
    filter keeps running either way and diverges slowly.
    """
    u = np.zeros(dyn.nu)
    x_next, F = dyn.step(x_rest, u)
    assert np.array_equal(F, F_dyn(x_rest, u, dyn.model, dyn.data))
    assert not np.array_equal(F, F_dyn(x_next, u, dyn.model, dyn.data))


# ------------------------------------------------- the trap

def test_reading_sensordata_straight_after_transitionFD_is_wrong(blind, x_rest) -> None:
    """Why `observe` runs a second `mj_forward`, measured rather than asserted.

    `mjd_transitionFD` restores qpos/qvel/act -- which is why the `step`
    pairing works at all -- but leaves `sensordata` holding whatever its last
    finite-difference perturbation computed. Reading it directly gives a `z`
    wrong by ~4.3e-4. Against `sig_acc = 0.05` that is about 1% of one sigma:
    too small to look like a bug, big enough to move the frozen run.
    """
    import mujoco as mj

    from erp.sim.mujoco import _load

    model, data = blind
    u = np.zeros(model.nu)
    z_ref = h_dyn(x_rest, u, model, data)

    _load(model, data, x_rest, u)
    mj.mj_forward(model, data)
    H = np.zeros((model.nsensordata, state_dim(model)))
    mj.mjd_transitionFD(model, data, 1e-6, True, None, None, H, None)
    z_naive = np.array(data.sensordata, dtype=np.float64)  # <-- no restoring forward

    err = float(np.abs(z_naive - z_ref).max())
    assert err == pytest.approx(4.3e-4, rel=0.3)
    assert err > 0.0
    # And the state really was restored -- it is only the sensors that were not.
    assert np.allclose(np.r_[data.qpos, data.qvel, data.act][: model.nq], x_rest[: model.nq])

    # What the implementation does instead.
    z_fixed, H_fixed = MujocoDynamics(model, data).observe(x_rest, u)
    assert np.array_equal(z_fixed, z_ref)
    assert np.array_equal(H_fixed, H)


# ------------------------------------------------- construction

def test_a_nonpositive_eps_is_refused(blind) -> None:
    with pytest.raises(ValueError, match="eps"):
        MujocoDynamics(*blind, eps=0.0)


def test_repr_names_the_dimensions(dyn: MujocoDynamics) -> None:
    assert "nx=11" in repr(dyn)
    assert "nz=15" in repr(dyn)
