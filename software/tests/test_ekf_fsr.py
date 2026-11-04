"""``EKFFSR``: the blind EKF plus a binary FSR contact measurement.

Fast, no mujoco: the filter runs on ``LinearDynamics`` and a linear contact
geometry, d(x) = a.x + b, so every update has a closed form to compare against.
The MuJoCo-backed geometry is tested in ``test_plant.py``.

The case that matters most is paired with its broken variant: OFF is an
INEQUALITY, so with the estimate outside the solid it must leave x and P
bit-identical. Treating OFF as an equality (always z = 0) pulls a free estimate
onto the surface; that variant is written out below and asserted to fail.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from erp.estimators import EKFFSR
from erp.models.linear import LinearDynamics

NX = 2
A_D = np.array([1.0, 0.0])       # d(x) = x[0] + B_D: x[0] is "height above the surface"
B_D = 0.0
SIG = 1e-3


class LinearGeometry:
    """d(x) = a.x + b, with a constant Jacobian. Satisfies ``ContactGeometry``."""

    nx = NX

    def __init__(self, a: np.ndarray = A_D, b: float = B_D) -> None:
        self.a, self.b = a, b

    def distance(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        return float(self.a @ x + self.b), self.a.copy()


def make(x0, P0=None, **kw) -> EKFFSR:
    dyn = LinearDynamics(np.eye(NX), C=np.eye(NX))
    P0 = np.diag([4e-4, 1e-4]) if P0 is None else P0
    kw.setdefault("sig_contacto", SIG)
    return EKFFSR(np.asarray(x0, float), P0, np.eye(NX) * 1e-6, np.eye(NX) * 0.01, dyn,
                  LinearGeometry(), **kw)


def closed_form(x, P, a, z, d, r):
    """Scalar KF update written out, no erp: K = P a / (a P a + r)."""
    s = a @ P @ a + r
    k = P @ a / s
    return x + k * (z - d), (np.eye(len(x)) - np.outer(k, a)) @ P, (z - d) ** 2 / s


@pytest.mark.parametrize("x0", [[0.010, 0.3], [-0.004, -0.1]])
def test_on_is_the_closed_form_scalar_update(x0) -> None:
    ekf = make(x0, pen_on=0.0008)
    x, P = ekf.x.copy(), ekf.P.copy()
    y, nis = ekf.update_contacto(True)
    x_ref, P_ref, nis_ref = closed_form(x, P, A_D, -0.0008, A_D @ x + B_D, SIG**2)
    np.testing.assert_allclose(ekf.x, x_ref, atol=1e-12)
    np.testing.assert_allclose(ekf.P, P_ref, atol=1e-12)
    assert nis == pytest.approx(nis_ref, rel=1e-9)
    assert y == pytest.approx(-0.0008 - (A_D @ x + B_D))


def test_off_outside_the_solid_changes_nothing_bit_for_bit() -> None:
    ekf = make([0.010, 0.3])           # 10 mm above the surface
    x, P = ekf.x.copy(), ekf.P.copy()
    assert ekf.update_contacto(False) is None
    assert np.array_equal(ekf.x, x)
    assert np.array_equal(ekf.P, P)

    # Broken variant: OFF as an EQUALITY (z = 0 regardless). Same arithmetic,
    # same geometry; it must move a free estimate, so it must fail the test above.
    broken = make([0.010, 0.3])
    d, D = broken.geometria.distance(broken.x)
    broken._corregir(np.array([0.0 - d]), D[None, :], broken.R_contacto)
    assert not np.array_equal(broken.x, x)
    assert abs(broken.x[0]) < 0.010 * 0.5, "the equality drags the estimate to the surface"


def test_off_inside_the_solid_pushes_the_estimate_out() -> None:
    """The stuck-after-lift-off case: FSR OFF while the estimate is inside."""
    ekf = make([-0.004, 0.0])
    P_before = ekf.P.copy()
    y, nis = ekf.update_contacto(False)
    assert y == pytest.approx(0.004)
    # Prior sigma along d is 20 mm, so 4 mm inside is a SMALL innovation: y^2 / (P00 + r).
    assert nis == pytest.approx(0.004**2 / (P_before[0, 0] + SIG**2), rel=1e-9)
    d_after, _ = ekf.geometria.distance(ekf.x)
    assert -0.004 < d_after <= 0.0
    assert abs(d_after) < 0.004 * 0.2           # most of the way out, sigma = 1 mm vs 20 mm
    assert ekf.P[0, 0] < P_before[0, 0]         # less uncertain along D
    assert ekf.P[1, 1] == pytest.approx(P_before[1, 1])   # D has no component there


def test_tol_off_tolerates_shallow_penetration() -> None:
    ekf = make([-0.0005, 0.0], tol_off=0.001)
    x = ekf.x.copy()
    assert ekf.update_contacto(False) is None
    assert np.array_equal(ekf.x, x)
    assert make([-0.002, 0.0], tol_off=0.001).update_contacto(False) is not None


def test_the_filter_stays_blind() -> None:
    for meth in (EKFFSR.predict, EKFFSR.update, EKFFSR.update_contacto):
        assert "u" not in inspect.signature(meth).parameters
    assert list(inspect.signature(EKFFSR.update_contacto).parameters) == ["self", "en_contacto"]


def test_construction_is_validated() -> None:
    dyn = LinearDynamics(np.eye(3), C=np.eye(3))
    with pytest.raises(ValueError, match="nx"):
        EKFFSR(np.zeros(3), np.eye(3), np.eye(3), np.eye(3), dyn, LinearGeometry())
    with pytest.raises(ValueError):
        make([0.0, 0.0], sig_contacto=0.0)
    with pytest.raises(ValueError):
        make([0.0, 0.0], pen_on=-1e-3)


def test_no_mujoco_is_needed_for_the_fsr_filter() -> None:
    """Subprocess, as in test_ekf_linear: other files import mujoco into this session."""
    script = textwrap.dedent("""
        import sys
        import numpy as np
        from erp.estimators import EKFFSR
        from erp.models.linear import LinearDynamics
        class G:
            nx = 2
            def distance(self, x):
                return float(x[0]), np.array([1.0, 0.0])
        ekf = EKFFSR(np.array([-0.01, 0.0]), np.eye(2) * 1e-4, np.eye(2) * 1e-6,
                     np.eye(2), LinearDynamics(np.eye(2), C=np.eye(2)), G())
        ekf.predict()
        assert ekf.update_contacto(False) is not None
        assert "mujoco" not in sys.modules, sorted(m for m in sys.modules if "mujoco" in m)
        print("OK")
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("OK")
