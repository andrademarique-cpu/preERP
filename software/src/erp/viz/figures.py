"""The three standard figures (ADR-0002 § 4.1).

Each takes plain arrays rather than a pipeline result dict, so the same function
draws a simulated run, a replayed log and a live session. Each returns its
``Figure`` instead of calling ``show`` — the caller decides whether it is going
on screen or to a file, and a function that shows cannot be used headless.

matplotlib is imported at module scope. ``erp/viz/__init__.py`` does not
re-export this module, so ``import erp.viz`` stays numpy-only and CI, which
installs ``.[dev]`` without ``[viz]``, can still import the package.

What these figures are for: reading a filter's behaviour, not judging it.
Correctness is judged by NEES and NIS — :mod:`erp.analysis.consistency` — never
by how well a line follows another line. An estimate that tracks truth closely
and believes itself ten times more precise than it is draws a perfect picture.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from erp.core.types import Array
from erp.viz.geometry import sigma_per_axis
from erp.viz.theme import COLORS, apply_theme

__all__ = ["effector_band", "sensor_compare", "three_way"]


def three_way(
    t_truth: Array,
    y_truth: Array,
    t_meas: Array,
    y_meas: Array,
    t_est: Array,
    y_est: Array,
    *,
    labels: Sequence[str],
    ylabel: str,
    title: str | None = None,
) -> Figure:
    """Plant truth, the noisy sensor, and the filter's estimate of the same channel.

    One row per channel, shared time axis. ``y_*`` are (n, k) with one column per
    channel and ``labels`` names them; ``ylabel`` carries the unit and frame, e.g.
    ``"m/s^2, link1 site frame"``.

    The three signals are *not* interchangeable and the figure is drawn to make
    that visible:

    - **truth** is what the un-edited plant did. It exists only in simulation.
    - **measured** is drawn as markers, not a line, because samples arrive at
      ~20 Hz against a 2 ms physics step — joining them implies a continuity the
      sensor never had, and hides that there are ~25 predicts between updates.
    - **estimated** is ``h(x_hat)`` at the filter's own rate.

    Missing truth is allowed: pass an empty ``y_truth`` and the row is drawn
    without it, which is the real-hardware case.
    """
    apply_theme()
    y_truth = np.asarray(y_truth, dtype=np.float64)
    y_meas = np.asarray(y_meas, dtype=np.float64)
    y_est = np.asarray(y_est, dtype=np.float64)
    n_ch = len(labels)
    fig, axes = plt.subplots(n_ch, 1, figsize=(9, 2.1 * n_ch), sharex=True, squeeze=False)
    for i, label in enumerate(labels):
        ax = axes[i, 0]
        if y_truth.size:
            ax.plot(t_truth, y_truth[:, i], color=COLORS["truth"], lw=1.0, label="plant (truth)")
        ax.plot(
            t_meas, y_meas[:, i], linestyle="none", marker="o", ms=3.0, alpha=0.75,
            color=COLORS["measured"], label="sensor (with noise)",
        )
        ax.plot(t_est, y_est[:, i], color=COLORS["estimated"], label="EKF estimate")
        ax.set_ylabel(f"{label}\n{ylabel}")
        if i == 0:
            ax.legend(loc="upper right", ncol=3)
    axes[-1, 0].set_xlabel("t [s], host monotonic base")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def sensor_compare(
    t_update: Array,
    nis: Array,
    *,
    nis_target: float,
    residual: Array | None = None,
    residual_label: str = "innovation  z - h(x_pred)",
    title: str | None = None,
) -> Figure:
    """NIS per applied measurement, optionally over its residual.

    ``t_update`` is (m,) seconds and ``nis`` is (m,) dimensionless, matching
    :class:`~erp.fusion.History`'s ``t_update`` and ``nis`` — one entry per
    measurement the filter actually *applied*. Dropped ones are absent here by
    construction and live in ``History.discarded``; a NIS trace that looks
    healthy because most samples never reached the filter is exactly the
    failure that counter exists to expose, so print it beside this figure.

    ``residual`` is optional and (m, k) in the sensor's units. The canonical
    choice is the innovation ``z - h(x_pred)``, which is what ``nis`` is built
    from; a posterior residual is also informative but is a different quantity,
    so pass ``residual_label`` and say which one it is rather than letting the
    axis imply the other.

    The target line is drawn because a NIS trace without it is unreadable: 29
    and 12 look alike on a log axis and the whole question is the ratio. The
    axis is logarithmic since NIS has a heavy right tail — brief moments where
    ``P`` is small and the linearisation is poor — which is also why
    :mod:`erp.analysis.consistency` reports the median rather than the mean.
    """
    apply_theme()
    nis = np.asarray(nis, dtype=np.float64)
    if residual is None:
        fig, ax_nis = plt.subplots(1, 1, figsize=(9, 3.0))
    else:
        fig, (ax_res, ax_nis) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        ax_res.plot(t_update, np.asarray(residual, dtype=np.float64), lw=0.9, alpha=0.8)
        ax_res.set_ylabel(residual_label)
    ax_nis.semilogy(t_update, nis, color=COLORS["estimated"], marker="o", ms=2.5, lw=0.8)
    ax_nis.axhline(
        nis_target, color=COLORS["target"], ls="--", lw=1.0,
        label=f"target {nis_target:g} = len(rows)",
    )
    ax_nis.set_ylabel("NIS")
    ax_nis.set_xlabel("t [s], host monotonic base")
    ax_nis.legend(loc="upper right")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def effector_band(
    t: Array,
    p_est: Array,
    C_site: Array,
    *,
    p_true: Array | None = None,
    n_sigma: float = 2.0,
    title: str | None = None,
) -> Figure:
    """End-effector position per world axis with its ``n_sigma`` band.

    ``p_est`` and ``p_true`` are (n, 3) in m, world frame; ``C_site`` is
    (n, 3, 3) in m^2 — the **full** block, not a per-axis sigma, because that is
    what :func:`~erp.analysis.propagate_to_site` returns and what an honest band
    needs.

    The band is still per-axis, since the figure has one axis per row, and that
    is the figure's limitation rather than the data's: over the golden run the
    ellipsoid's major axis sits a median 24.9 deg off the nearest world axis, so
    these three bands understate the worst direction by a median 9.6%. Use
    :func:`erp.viz.geometry.principal_tilt_deg` when that matters.
    """
    apply_theme()
    p_est = np.asarray(p_est, dtype=np.float64)
    sig = sigma_per_axis(C_site)
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    for i, axis_name in enumerate("xyz"):
        ax = axes[i]
        ax.fill_between(
            t, (p_est[:, i] - n_sigma * sig[:, i]) * 1e3,
            (p_est[:, i] + n_sigma * sig[:, i]) * 1e3,
            color=COLORS["band"], alpha=0.18, lw=0,
            label=f"+-{n_sigma:g} sigma" if i == 0 else None,
        )
        ax.plot(t, p_est[:, i] * 1e3, color=COLORS["estimated"],
                label="EKF estimate" if i == 0 else None)
        if p_true is not None:
            ax.plot(t, np.asarray(p_true, dtype=np.float64)[:, i] * 1e3,
                    color=COLORS["truth"], lw=1.0,
                    label="plant (truth)" if i == 0 else None)
        ax.set_ylabel(f"{axis_name} [mm]")
    axes[0].legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("t [s], host monotonic base")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
