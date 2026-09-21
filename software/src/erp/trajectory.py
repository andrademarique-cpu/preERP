"""Setpoint generation: the profile the arm is asked to follow.

Command path, beside :mod:`erp.robot`, and documented in English for the same
reason. Nothing here knows about an arm, a port or a model -- a trajectory is
``(t, q, qd)`` in model radians, and :meth:`erp.robot.JointMap.validate` is
what decides whether a given arm may be asked to follow it.

A single module rather than a package: after P2 took ``validate`` onto
``JointMap``, what ADR-0002 4.1 listed under ``trajectory/`` is two functions.

Why these two are here at all, rather than in the notebook cell that used to
hold them: the sine formula existed in **five** places -- two notebook cells
(three occurrences), ``test_robot.py`` and ``scripts/make_golden_run.py``.
That last one is the reason it mattered. The golden pipeline *regenerates* the
trajectory and drives the plant with it, so the frozen fixture's ``lag_imu``
depends on a hand-copied formula; changing the notebook's period without
changing the script's would leave the fixture quietly describing a run the
notebook no longer performs, and ``--check`` would keep passing, because it
re-runs the same script instead of comparing against the notebook. The
artifact whose job is catching drift had a drift channel of its own.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from erp.core.types import Array

__all__ = ["resample", "sine_sweep"]


def sine_sweep(
    amplitudes_deg: npt.ArrayLike, period_s: float, dt: float
) -> tuple[Array, Array, Array]:
    """One full sine cycle per joint, starting and ending at zero.

    ``q = A * (1 + sin(w*t - pi/2))``, so each joint leaves 0, reaches ``2A``
    at the half period and comes back. Starting at zero is what lets the
    trajectory be commanded from the ``home`` keyframe without a step.

    Returns ``(t, q, qd)`` with ``t`` in seconds, ``q`` in model radians
    (N, n) and ``qd`` in rad/s.

    **``period_s`` is the speed knob, and the only one.** Amplitude does not
    change with it, so position limits keep reporting OK while peak speed
    grows as ``A * 2*pi / period_s``. The range guard checks velocity for
    exactly this reason -- see :meth:`erp.robot.JointMap.validate`. Do not
    confuse it with the setpoint rate: streaming faster sends more points of
    the same trajectory, it does not make the arm move faster.

    ``qd`` carries the chain-rule factor ``w``. Without it the curve keeps its
    shape and only its scale is wrong, which no plot reveals. Note what does
    and does not depend on this: ``qd`` is a **diagnostic**. The streaming
    loop commands positions, the plant is driven by positions, and the range
    guard differentiates ``q`` numerically rather than reading ``qd``. A
    missing ``w`` here produces a wrong velocity plot and nothing else.

    Parameters
    ----------
    amplitudes_deg:
        (n,) per-joint amplitude in degrees. Travel is twice this.
    period_s:
        Seconds per cycle.
    dt:
        Used only to choose the sample count, ``int(period_s / dt)`` -- see
        the note below.

    Notes
    -----
    The sample spacing is **not** ``dt``. ``n = int(period_s / dt)`` points
    spanning ``[0, period_s]`` inclusive are spaced ``period_s / (n - 1)``, so
    at ``period_s = 3`` and ``dt = 2 ms`` the trajectory steps 2.0013 ms while
    the physics steps 2.0000 ms: the commanded profile runs 0.067% slow and
    ends one timestep behind the simulation clock.

    This is carried over unchanged and deliberately not repaired. It is small
    next to the arm's own ~395 ms transport lag, which is presumably why it
    was never noticed -- and the frozen golden run was generated with it, so
    correcting it here would move the fixture rather than fix a bug. It is
    recorded so that whoever does change it knows what will move.
    """
    if period_s <= 0.0:
        raise ValueError("period_s must be > 0")
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    amplitudes = np.deg2rad(np.asarray(amplitudes_deg, dtype=np.float64))
    if amplitudes.ndim != 1 or amplitudes.size == 0:
        raise ValueError(f"amplitudes_deg must be 1-D and non-empty, got {amplitudes.shape}")
    w = 2 * np.pi / period_s
    n_samples = int(period_s / dt)
    if n_samples < 2:
        raise ValueError(f"period_s / dt gives {n_samples} samples; need at least 2")
    t = np.linspace(0.0, period_s, n_samples)
    q = np.sin(w * t - np.pi / 2)[:, None] * amplitudes + amplitudes
    qd = amplitudes * w * np.cos(w * t - np.pi / 2)[:, None]
    return (
        np.asarray(t, dtype=np.float64),
        np.asarray(q, dtype=np.float64),
        np.asarray(qd, dtype=np.float64),
    )


def resample(t: npt.ArrayLike, q: npt.ArrayLike, rate_hz: float) -> tuple[Array, Array]:
    """Thin the trajectory down to something the serial port can carry.

    The 1500 points at a 2 ms step are not transmissible: at 115200 baud with
    a request/response protocol there is no way. ~25 Hz leaves ~76 setpoints.

    Endpoints are preserved exactly -- the resampled trajectory starts and
    ends where the original did, which is what keeps the arm's final pose the
    commanded one rather than wherever the last retained sample happened to
    fall.

    Returns ``(t_r, q_r)``; linear interpolation, column by column.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)  # cell 3 used to leave q as a list
    if t_arr.ndim != 1:
        raise ValueError(f"t must be 1-D, got {t_arr.shape}")
    if q_arr.ndim != 2 or len(q_arr) != len(t_arr):
        raise ValueError(f"q must be (len(t), n); got {q_arr.shape} for t {t_arr.shape}")
    if rate_hz <= 0.0:
        raise ValueError("rate_hz must be > 0")
    # The notebook wrote `int(round(...))`. `np.float64` subclasses `float`, so
    # `round` already returns an `int` and the cast was a no-op -- dropped
    # because ruff flags it (RUF046), and the bit-identity test covers it.
    n = max(2, round((t_arr[-1] - t_arr[0]) * rate_hz) + 1)
    t_r = np.linspace(t_arr[0], t_arr[-1], n)
    q_r = np.column_stack([np.interp(t_r, t_arr, q_arr[:, k]) for k in range(q_arr.shape[1])])
    return np.asarray(t_r, dtype=np.float64), np.asarray(q_r, dtype=np.float64)
