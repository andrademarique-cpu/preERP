"""In-memory measurement log for the trajectory loop.

Replaces the notebook's list-of-rows logging and its start/stop events: the
loop appends everything it drains, and the trajectory window is cut afterwards
by timestamp. Appends are O(1) list operations with no numpy concatenation, so
logging costs the loop essentially nothing; arrays are built once, on demand.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np

from erp.core.types import Array, IntArray, Measurement

__all__ = ["MeasurementLog"]


class MeasurementLog:
    """Measurements grouped by ``source``, kept in the order they were added."""

    def __init__(self) -> None:
        self._t: dict[str, list[float]] = {}
        self._z: dict[str, list[Array]] = {}
        self._rows: dict[str, IntArray] = {}

    def extend(self, measurements: Iterable[Measurement]) -> None:
        """Append samples (typically the output of ``sensor.drain()``)."""
        for m in measurements:
            src = m.source
            t_list = self._t.get(src)
            if t_list is None:
                t_list = self._t[src] = []
                self._z[src] = []
                self._rows[src] = m.rows
            t_list.append(m.timestamp)
            self._z[src].append(m.z)

    @property
    def sources(self) -> list[str]:
        return list(self._t)

    def count(self, source: str) -> int:
        return len(self._t.get(source, ()))

    def __len__(self) -> int:
        return sum(len(v) for v in self._t.values())

    def to_arrays(
        self, source: str, t_start: float | None = None, t_end: float | None = None
    ) -> tuple[Array, Array, IntArray]:
        """``(t, Z, rows)`` for one source, sorted by time, optionally windowed.

        ``t`` is (n,) seconds in the host base, ``Z`` is (n, k), ``rows`` (k,).
        The window is closed: ``t_start <= t <= t_end``.
        """
        if source not in self._t:
            raise KeyError(f"no samples from {source!r}; have {self.sources}")
        t = np.asarray(self._t[source], dtype=np.float64)
        Z = np.stack(self._z[source]) if t.size else np.empty((0, self._rows[source].size))
        order = np.argsort(t, kind="stable")
        t, Z = t[order], Z[order]
        keep = np.ones(t.size, dtype=bool)
        if t_start is not None:
            keep &= t >= t_start
        if t_end is not None:
            keep &= t <= t_end
        return t[keep], np.asarray(Z[keep], dtype=np.float64), self._rows[source]

    def window(self, t_start: float, t_end: float) -> MeasurementLog:
        """New log holding only samples with ``t_start <= t <= t_end``."""
        out = MeasurementLog()
        for src in self._t:
            t, Z, rows = self.to_arrays(src, t_start, t_end)
            out._t[src] = t.tolist()
            out._z[src] = list(Z)
            out._rows[src] = rows
        return out

    def save_csv(
        self,
        path: str | Path,
        source: str,
        *,
        t_ref: float = 0.0,
        column_names: Sequence[str] | None = None,
        transform: Callable[[Array], Array] | None = None,
    ) -> Path:
        """Write ``timestamp,<columns>`` for one source; returns the path.

        Timestamps are written as ``t - t_ref``: pass the loop's ``t0`` so the
        file shares its time axis with ``run_trajectory``'s ``t_cmd``.

        Without ``transform`` the values are the calibrated ``z`` (reload with
        :meth:`erp.sensors.replay.ReplaySensor.from_log_csv`). With
        ``transform`` -- e.g. ``IMUDecoder.to_raw`` and ``column_names=keys`` --
        the (n, k) ``z`` block is mapped first, so the file stores raw device
        values that can be re-decoded after the configuration changes (reload
        with :meth:`~erp.sensors.replay.ReplaySensor.from_legacy_imu_csv`).
        """
        t, Z, rows = self.to_arrays(source)
        if transform is not None:
            Z = np.atleast_2d(np.asarray(transform(Z), dtype=np.float64))
            if not t.size:
                Z = np.empty((0, Z.shape[-1] if Z.ndim == 2 else 0))
        width = Z.shape[1] if Z.ndim == 2 else rows.size
        names = list(column_names) if column_names is not None else [f"z{i}" for i in range(width)]
        if len(names) != width:
            raise ValueError(f"{len(names)} column names for {width} columns")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = np.column_stack([t - t_ref, Z]) if t.size else np.empty((0, width + 1))
        np.savetxt(path, table, delimiter=",", header=",".join(["timestamp", *names]),
                   comments="", fmt="%.9g")
        return path
