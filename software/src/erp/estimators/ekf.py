"""Extended Kalman filter with Joseph-form covariance update."""

from __future__ import annotations

import numpy as np

from erp.core.linalg import mahalanobis, make_spd
from erp.core.types import Array, GaussianState, Measurement
from erp.estimators.base import StateEstimator
from erp.models.base import MeasurementModel, ProcessModel

__all__ = ["ExtendedKalmanFilter"]


class ExtendedKalmanFilter(StateEstimator):
    """EKF over an arbitrary :class:`~erp.models.base.ProcessModel`.

    Handed a linear process and measurement model this reduces exactly to the
    standard Kalman filter, so the linear case needs no separate class.

    Two numerical choices are load-bearing rather than stylistic:

    * **Joseph form** for the covariance update, which stays symmetric
      positive semi-definite under the round-off that the plain
      ``(I - KH) P`` form does not survive here.
    * **Eigenvalue flooring** via :func:`~erp.core.linalg.make_spd` after every
      operation on ``P``. Without it the NEES goes negative and the consistency
      diagnostic fails silently.
    """

    def __init__(self, process_model: ProcessModel, initial: GaussianState) -> None:
        super().__init__(process_model, initial)
        self._state.P = make_spd(self._state.P)
        self._last_innovation: Array | None = None
        self._last_nis: float | None = None

    @property
    def last_innovation(self) -> Array | None:
        """Innovation ``z - h(x, u)`` from the most recent update, or ``None``.

        Units are the measuring sensor's own.
        """
        return self._last_innovation

    @property
    def last_nis(self) -> float | None:
        """NIS of the most recent update, or ``None`` before the first.

        Chi-squared with ``dim(z)`` degrees of freedom when the filter is
        consistent. This is the diagnostic that survives onto hardware, since
        it needs no ground truth.
        """
        return self._last_nis

    def predict(self, u: Array, dt: float) -> None:
        """Advance the belief by ``dt`` seconds under constant input ``u``."""
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
        model = self._process_model
        x, P = self._state.x, self._state.P
        F = model.jacobian(x, u, dt)
        self._state.x = model.predict(x, u, dt)
        self._state.P = make_spd(F @ P @ F.T + model.Q(dt))

    def update(self, m: Measurement, model: MeasurementModel, u: Array) -> None:
        """Fold in measurement ``m``, taken under input ``u``."""
        x, P = self._state.x, self._state.P
        H = model.jacobian(x, u)
        y = m.z - model.h(x, u)
        R = model.R if m.R is None else m.R
        if y.shape != (H.shape[0],):
            raise ValueError(f"measurement has size {y.shape}, model expects ({H.shape[0]},)")

        S = make_spd(H @ P @ H.T + R)
        K = np.linalg.solve(S, H @ P).T          # P H^T S^-1, without forming S^-1
        self._state.x = x + K @ y
        I_KH = np.eye(x.size) - K @ H
        self._state.P = make_spd(I_KH @ P @ I_KH.T + K @ R @ K.T)

        self._last_innovation = y
        self._last_nis = mahalanobis(y, S)
