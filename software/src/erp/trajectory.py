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

__all__ = ["clamped_cubic_spline", "resample", "sine_sweep", "slew_limit"]


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


def clamped_cubic_spline(
    t_knots: npt.ArrayLike,
    q_knots: npt.ArrayLike,
    t: npt.ArrayLike,
    v_start: npt.ArrayLike | None = None,
    v_end: npt.ArrayLike | None = None,
) -> tuple[Array, Array]:
    """Cubic spline through ``(t_knots, q_knots)`` with the end velocities fixed.

    One cubic per interval, continuous in position, velocity **and**
    acceleration at every interior knot (C2), passing exactly through each
    knot. The two missing conditions are the velocities at the first and last
    knot -- "clamped" -- which default to zero: a move that starts and ends at
    rest. That is the property that matters on the arm. A natural spline
    (zero *acceleration* at the ends) leaves the endpoints with whatever
    velocity falls out of the fit, i.e. a step in commanded speed where the
    move begins and where it stops.

    Why not ``np.interp`` between waypoints: piecewise-linear joint motion has
    a velocity step at every waypoint, and a position servo turns a velocity
    step into a jerk. ``scipy.interpolate.CubicSpline(bc_type="clamped")`` is
    the same object; scipy is not a dependency (dropped at P0), and the
    tridiagonal system here is a few knots wide.

    Parameters
    ----------
    t_knots:
        (K,) knot times, s, strictly increasing, K >= 2.
    q_knots:
        (K, n) joint positions at the knots, any consistent unit (model
        radians on the command path).
    t:
        (N,) times to evaluate at, s. Outside ``[t_knots[0], t_knots[-1]]``
        the end cubic is extrapolated -- callers sample inside.
    v_start, v_end:
        (n,) velocities at the first and last knot, unit/s. Default zeros.

    Returns
    -------
    (q, qd):
        (N, n) position and (N, n) velocity, unit and unit/s.
    """
    tk = np.asarray(t_knots, dtype=np.float64)
    qk = np.asarray(q_knots, dtype=np.float64)
    tt = np.asarray(t, dtype=np.float64)
    if tk.ndim != 1 or len(tk) < 2:
        raise ValueError(f"t_knots must be 1-D with at least 2 knots, got {tk.shape}")
    if qk.ndim == 1:
        qk = qk[:, None]
    if qk.ndim != 2 or len(qk) != len(tk):
        raise ValueError(f"q_knots must be (len(t_knots), n); got {qk.shape} for {tk.shape}")
    h = np.diff(tk)
    if np.any(h <= 0.0):
        raise ValueError("t_knots must be strictly increasing")
    n = qk.shape[1]
    v0 = np.zeros(n) if v_start is None else np.broadcast_to(np.asarray(v_start, float), (n,))
    v1 = np.zeros(n) if v_end is None else np.broadcast_to(np.asarray(v_end, float), (n,))

    # Second derivatives M at the knots: the standard clamped system,
    # tridiagonal, solved densely because K is a handful of waypoints.
    k = len(tk)
    a = np.zeros((k, k))
    rhs = np.zeros((k, n))
    slope = np.diff(qk, axis=0) / h[:, None]
    a[0, 0], a[0, 1] = 2.0 * h[0], h[0]
    rhs[0] = 6.0 * (slope[0] - v0)
    for i in range(1, k - 1):
        a[i, i - 1], a[i, i], a[i, i + 1] = h[i - 1], 2.0 * (h[i - 1] + h[i]), h[i]
        rhs[i] = 6.0 * (slope[i] - slope[i - 1])
    a[-1, -2], a[-1, -1] = h[-1], 2.0 * h[-1]
    rhs[-1] = 6.0 * (v1 - slope[-1])
    m = np.linalg.solve(a, rhs)

    seg = np.clip(np.searchsorted(tk, tt, side="right") - 1, 0, k - 2)
    hi = h[seg][:, None]
    lo_gap = (tk[seg + 1] - tt)[:, None]  # time to the right knot
    hi_gap = (tt - tk[seg])[:, None]  # time since the left knot
    m0, m1, y0, y1 = m[seg], m[seg + 1], qk[seg], qk[seg + 1]
    c0 = y0 / hi - m0 * hi / 6.0
    c1 = y1 / hi - m1 * hi / 6.0
    q = (m0 * lo_gap**3 + m1 * hi_gap**3) / (6.0 * hi) + c0 * lo_gap + c1 * hi_gap
    qd = (-m0 * lo_gap**2 + m1 * hi_gap**2) / (2.0 * hi) - c0 + c1
    return np.asarray(q, dtype=np.float64), np.asarray(qd, dtype=np.float64)


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


def slew_limit(current: npt.ArrayLike, target: npt.ArrayLike, max_step: npt.ArrayLike) -> Array:
    """One tick of a rate-limited setpoint: move ``current`` toward ``target``.

    Each component moves by at most ``max_step`` (same units as the setpoint,
    e.g. rad per tick = rad/s times the tick length), independently per joint.
    Once the remaining distance fits inside one step, the result **is**
    ``target``, exactly.

    Why a setpoint needs this at all: a position servo handed a step command
    drives as hard as it can toward it. On the simulated MyPalletizer the
    servos saturate above ~2 rad/s, which reads as a filter problem and is
    not, and against a prop a step is an impact. The interactive demo lets a
    target be typed in, so without this a typed angle is exactly such a step.

    Why "exactly", and how much it matters -- which is less than it looks.
    The obvious ``current + clip(target - current, -s, s)`` misses ``target``
    by one ULP when the final step spans a distance comparable to the values
    themselves: for about 8% of one-tick jumps between random points on
    [-3, 3] (``1.2 -> 0.1`` lands on ``0.10000000000000009``). Over a *slow*
    approach it does not happen: the last step starts close to the target, so
    ``target - current`` is exact (Sterbenz) and the sum lands. Measured, at a
    0.01 rad step: 0 misses in 20 000 random approaches, and 0 in 20 000 with
    targets within 0.01 of zero. The demo slews ~0.017 rad per tick, so there
    the guard is insurance, not a fix. It is kept because it costs nothing
    and makes the guarantee unconditional: a miss is 1e-16 rad and moves
    nothing physically, but it makes a ``ctrl == target`` test for "arrived"
    never come true.

    Parameters
    ----------
    current, target:
        (n,) setpoints, same units.
    max_step:
        Scalar or (n,), >= 0. Zero freezes ``current``.
    """
    cur = np.asarray(current, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if cur.shape != tgt.shape:
        raise ValueError(f"current {cur.shape} and target {tgt.shape} differ in shape")
    step = np.broadcast_to(np.asarray(max_step, dtype=np.float64), cur.shape)
    if not np.all(np.isfinite(step)) or np.any(step < 0.0):
        raise ValueError("max_step must be finite and >= 0")
    delta = tgt - cur
    within = np.abs(delta) <= step
    out = np.where(within, tgt, cur + np.sign(delta) * step)
    return np.asarray(out, dtype=np.float64)
