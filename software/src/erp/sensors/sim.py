"""Simulated sensor built from a recorded MuJoCo ``sensordata`` log.

Imports no mujoco: it takes the log as arrays (e.g. ``t`` and
``sensor_log_sim`` from the palletizer notebook), so the trajectory loop can be
exercised end to end with noise, decimation and latency but no hardware -- and
with the clean readings kept as ground truth for NEES/NIS work later.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from erp.core.types import Array
from erp.sensors.base import shared_rows_R, sqrt_psd
from erp.sensors.replay import ReplaySensor

__all__ = ["SimSensor"]


class SimSensor(ReplaySensor):
    """A :class:`ReplaySensor` over decimated, noise-corrupted sensordata.

    Noise is drawn once, up front, from N(0, R) with the declared ``R`` --
    the filter is told the exact noise it gets, which makes this a
    plumbing check, not evidence about the estimator on real data.
    """

    def __init__(
        self,
        t: npt.ArrayLike,
        sensordata: npt.ArrayLike,
        *,
        rows: npt.ArrayLike,
        R: npt.ArrayLike,
        name: str = "sim",
        rate_hz: float | None = None,
        latency_s: float = 0.0,
        seed: int = 0,
        noise: bool = True,
        t_shift: float = 0.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        t:
            (N,) simulation time of each row, seconds.
        sensordata:
            (N, nsensordata) full MuJoCo ``data.sensordata`` log; ``rows``
            selects the channels.
        rows, R:
            Channels to emit and their noise covariance (units of the MuJoCo
            sensors squared).
        rate_hz:
            Output rate. ``None`` keeps every row. Samples are taken at the
            first log row at or after each ``k / rate_hz``.
        latency_s:
            Delay before a sample is released when ``clock`` is given, seconds.
        seed:
            Noise RNG seed.
        noise:
            ``False`` emits the clean readings.
        t_shift:
            Added to every timestamp; pass the ``perf_counter()`` value at the
            start of the run to place samples in the host time base.
        clock:
            Host clock for paced release; ``None`` releases everything.
        """
        rows_a, R_a = shared_rows_R(rows, R)
        t_a = np.asarray(t, dtype=np.float64).reshape(-1)
        S = np.atleast_2d(np.asarray(sensordata, dtype=np.float64))
        if S.shape[0] != t_a.size:
            raise ValueError(f"sensordata has {S.shape[0]} rows, t has {t_a.size}")
        if rows_a.size and rows_a.max() >= S.shape[1]:
            raise ValueError(
                f"rows reach index {rows_a.max()}, sensordata has {S.shape[1]} columns"
            )

        if rate_hz is not None and t_a.size:
            if rate_hz <= 0.0:
                raise ValueError(f"rate_hz must be > 0, got {rate_hz}")
            grid = np.arange(t_a[0], t_a[-1] + 1e-12, 1.0 / rate_hz)
            pick = np.unique(np.clip(np.searchsorted(t_a, grid - 1e-12), 0, t_a.size - 1))
        else:
            pick = np.arange(t_a.size)

        t_out = t_a[pick]
        clean = S[np.ix_(pick, rows_a)]
        Z = clean
        if noise and clean.size:
            rng = np.random.default_rng(seed)
            Z = clean + rng.standard_normal(clean.shape) @ _sqrt_psd(R_a).T

        self.truth_t: Array = t_out + t_shift
        """(n,) host-base times of the emitted samples, seconds."""
        self.truth_z: Array = clean
        """(n, k) noise-free readings at those times."""

        base = ReplaySensor.from_arrays(t_out, Z, rows=rows_a, R=R_a, name=name, t_shift=t_shift)
        super().__init__(
            base._measurements, rows=rows_a, R=R_a, name=name, clock=clock, latency_s=latency_s
        )


def _sqrt_psd(R: Array) -> Array:
    """Deprecated alias for :func:`erp.sensors.base.sqrt_psd`.

    The implementation moved to ``base`` so the live sensor could colour its
    noise the same way; it is byte-for-byte the same arithmetic, so every
    ``SimSensor`` draw is unchanged.
    """
    return sqrt_psd(R)
