"""Multi-rate sensor fusion: all timestamp handling lives here.

Keeping it in one class is deliberate. Timestamp logic bolted onto individual
``Sensor`` subclasses is how multi-rate handling becomes ad hoc and how
covariance corruption becomes untraceable.

See ``docs/adr/0001-multi-rate-fusion.md``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from erp.core.timeline import InputHistory
from erp.core.types import GaussianState, Measurement
from erp.estimators.base import StateEstimator
from erp.sensors.base import Sensor

__all__ = ["FusionEngine"]


class FusionEngine:
    """Drive an estimator from sensors running at different rates and latencies.

    The governing rule, generalised from measurements to inputs:

        Prediction advances to the next **event**, where the event set is the
        union of measurement timestamps and control-input change times.

    Never by a fixed ``dt``. Fixed-step prediction under out-of-order arrivals
    silently corrupts the covariance -- the trajectory still looks plausible,
    and only NEES/NIS catches it. Splitting at input breakpoints matters for
    the same reason: ``u`` is piecewise constant, so integrating across a
    change with one value is a modelling error, not a rounding one.
    """

    def __init__(
        self,
        estimator: StateEstimator,
        sensors: Mapping[str, Sensor],
        inputs: InputHistory,
        *,
        buffer_horizon: float = 0.05,
        t0: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        estimator:
            The filter to drive.
        sensors:
            Named measurement sources. Names appear in :attr:`discarded`.
        inputs:
            Control-input history, used both to split prediction intervals and
            to supply ``u`` at each measurement instant.
        buffer_horizon:
            Seconds to hold measurements before processing, absorbing transport
            latency and reordering between sensors. The estimate therefore
            trails ``t_now`` by roughly this much. Size it above the worst
            expected sensor latency.
        t0:
            Filter start time, seconds in the host time base.
        """
        if buffer_horizon < 0.0:
            raise ValueError(f"buffer_horizon must be >= 0, got {buffer_horizon}")
        self._estimator = estimator
        self._sensors = dict(sensors)
        self._inputs = inputs
        self._buffer_horizon = float(buffer_horizon)
        self._t = float(t0)
        self._pending: list[tuple[float, str, Measurement]] = []
        self._discarded: Counter[str] = Counter()

    @property
    def time(self) -> float:
        """Timestamp the current belief is valid at, seconds.

        Advances only to processed events, so it trails ``t_now`` by up to
        ``buffer_horizon``. Consumers needing an estimate at a later instant
        must extrapolate explicitly rather than assume this is "now".
        """
        return self._t

    @property
    def state(self) -> GaussianState:
        """Current belief, valid at :attr:`time`."""
        return self._estimator.state

    @property
    def discarded(self) -> dict[str, int]:
        """Count of measurements dropped for arriving too late, per sensor.

        Assert on this in tests. A silent discard path is how a sensor quietly
        stops contributing while every plot still looks correct.
        """
        return dict(self._discarded)

    @property
    def pending(self) -> int:
        """Number of buffered measurements not yet releasable."""
        return len(self._pending)

    def step(self, t_now: float) -> GaussianState:
        """Ingest available measurements and advance the filter.

        Drains every sensor, releases what is older than the buffer horizon,
        and processes it in timestamp order. Returns the belief at
        :attr:`time`, which is the last processed event, *not* ``t_now``.

        Measurements older than the current filter time are dropped and counted
        (see :attr:`discarded`) rather than applied out of sequence.
        """
        for name, sensor in self._sensors.items():
            for m in sensor.drain():
                self._pending.append((m.timestamp, name, m))

        self._pending.sort(key=lambda item: item[0])
        release_before = t_now - self._buffer_horizon

        keep: list[tuple[float, str, Measurement]] = []
        for ts, name, m in self._pending:
            if ts > release_before:
                keep.append((ts, name, m))
                continue
            if ts < self._t:
                self._discarded[name] += 1
                continue
            self._advance_to(ts)
            self._estimator.update(m, self._sensors[name].measurement_model,
                                   self._inputs.u_at(ts))
        self._pending = keep
        return self._estimator.state

    def _advance_to(self, t_target: float) -> None:
        """Predict forward to ``t_target``, splitting at input breakpoints.

        Each sub-interval sees a single constant ``u``, which is what makes the
        zero-order-hold discretisation in the process model exact.
        """
        if t_target < self._t:
            raise ValueError(f"cannot advance backwards: {t_target} < {self._t}")
        t = self._t
        for t_next in [*self._inputs.breakpoints_in(t, t_target), t_target]:
            if t_next > t:
                self._estimator.predict(self._inputs.u_at(t), t_next - t)
                t = t_next
        self._t = t
