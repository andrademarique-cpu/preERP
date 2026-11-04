"""``erp.analysis.penetracion``: how far the estimate sits inside a solid.

Pure numpy over synthetic distance series, so no mujoco marker. Every metric is
paired with the broken variant it exists to reject, in the shape of
``test_clock.py``: the correct version passes a bound and the broken one is
asserted to miss it.
"""

from __future__ import annotations

import numpy as np
import pytest

from erp.analysis.penetracion import penetracion, resumen_penetracion

DT = 0.01
T = np.arange(0.0, 4.0, DT)
CONTACT = (T >= 1.0) & (T < 2.0)


def _true_series() -> np.ndarray:
    """10 mm clearance, then 1.0 s of 2 mm soft-contact sink, then clearance again."""
    return np.where(CONTACT, -0.002, 0.010)


def test_penetration_is_zero_outside_and_positive_inside() -> None:
    d = np.array([0.02, 0.0, -0.003])
    np.testing.assert_allclose(penetracion(d), [0.0, 0.0, 0.003])
    # Broken variant: abs(d) reports 20 mm of clearance as 20 mm of clipping.
    assert np.abs(d)[0] == pytest.approx(0.02)
    assert penetracion(d)[0] == 0.0


def test_an_estimate_that_follows_truth_is_charged_nothing() -> None:
    """Soft contact belongs to the plant, so an exact estimate has zero excess."""
    d = _true_series()
    r = resumen_penetracion(T, d, d)
    assert r.max_pen_true_mm == pytest.approx(2.0)
    assert r.max_exceso_mm == 0.0
    assert r.integral_mm_s == 0.0
    assert r.frac_recorte == 0.0
    assert r.n_contactos == 1
    assert r.recuperacion_s == 0.0
    # Broken variant: measured against ZERO instead of truth, the same exact
    # estimate is charged the plant's own 2 mm, for the whole contact.
    against_zero = resumen_penetracion(T, d, np.full_like(d, 0.01))
    assert against_zero.max_exceso_mm == pytest.approx(2.0)
    assert against_zero.frac_recorte == pytest.approx(CONTACT.mean())


def test_excess_depth_and_integral_have_the_injected_size() -> None:
    """5 mm deeper than truth for 0.5 s: excess 5 mm, integral ~2.5 mm*s."""
    d_true = _true_series()
    extra = (T >= 1.2) & (T < 1.7)
    d_est = np.where(extra, d_true - 0.005, d_true)
    r = resumen_penetracion(T, d_est, d_true)
    assert r.max_exceso_mm == pytest.approx(5.0)
    # Trapezoids smear one sample at each edge: 0.5 s +- one dt.
    assert r.integral_mm_s == pytest.approx(2.5, abs=5.0 * DT)
    assert r.frac_recorte == pytest.approx(extra.mean())


def test_recovery_measures_an_estimate_stuck_in_the_solid_after_release() -> None:
    """The modelled-floor failure: truth lifts off, the estimate stays inside 0.3 s."""
    d_true = _true_series()
    stuck = (T >= 2.0) & (T < 2.3)
    d_est = np.where(stuck, -0.002, d_true)
    r = resumen_penetracion(T, d_est, d_true)
    assert r.recuperacion_s == pytest.approx(0.3, abs=1.5 * DT)
    assert r.max_exceso_mm == pytest.approx(2.0)

    # Never comes out before the end of the run: infinite, not "the run length".
    never = np.where(T >= 1.0, -0.002, d_true)
    assert np.isinf(resumen_penetracion(T, never, d_true).recuperacion_s)


def test_sigma_normalised_excess_separates_honest_from_overconfident() -> None:
    d_true = _true_series()
    d_est = np.where(CONTACT, d_true - 0.004, d_true)
    honest = resumen_penetracion(T, d_est, d_true, sigma_n=np.full_like(T, 0.003))
    overconfident = resumen_penetracion(T, d_est, d_true, sigma_n=np.full_like(T, 0.0001))
    assert honest.max_exceso_sigma == pytest.approx(4.0 / 3.0)
    assert overconfident.max_exceso_sigma == pytest.approx(40.0)
    # No excess above threshold -> no ratio to report.
    assert np.isnan(resumen_penetracion(T, d_true, d_true, np.ones_like(T)).max_exceso_sigma)


def test_shapes_must_agree() -> None:
    with pytest.raises(ValueError):
        resumen_penetracion(T, T[:-1], T)
    with pytest.raises(ValueError):
        resumen_penetracion(T, T, T, sigma_n=T[:-1] + 1.0)
