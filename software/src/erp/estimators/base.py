"""State estimator contract.

One of the four ABCs defining the system contract; changing a signature here
requires sign-off from every module owner (``CLAUDE.md``). Current signatures
are those adopted in ``docs/adr/0001-multi-rate-fusion.md`` decision D1.

This module must never import from :mod:`erp.sensors`, ``firmware`` or ROS 2.
An estimator has to be runnable in a notebook, in CI and against recorded data
with zero hardware attached.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from erp.core.types import Array, GaussianState, Measurement
from erp.models.base import MeasurementModel, ProcessModel

__all__ = ["StateEstimator"]


class StateEstimator(ABC):
    """Recursive Gaussian state estimator.

    Implementations are interchangeable: calling code must not need to know
    whether it holds a KF, an EKF or a UKF.
    """

    def __init__(self, process_model: ProcessModel, initial: GaussianState) -> None:
        """
        Parameters
        ----------
        process_model:
            Model used to propagate the state between measurements.
        initial:
            Initial belief. Copied, so the caller keeps ownership of its arrays.
        """
        self._process_model = process_model
        self._state = initial.copy()

    @property
    def state(self) -> GaussianState:
        """Current belief. Returns the live object; copy before mutating."""
        return self._state

    @property
    def process_model(self) -> ProcessModel:
        """The process model this estimator propagates with."""
        return self._process_model

    @abstractmethod
    def predict(self, u: Array, dt: float) -> None:
        """Advance the belief by ``dt`` seconds under input ``u``.

        ``u`` is constant across the interval. The fusion layer guarantees this
        by splitting every prediction at control-input breakpoints, so
        implementations must never assume a fixed ``dt``.
        """

    @abstractmethod
    def update(self, m: Measurement, model: MeasurementModel, u: Array) -> None:
        """Fold measurement ``m`` into the belief.

        Parameters
        ----------
        m:
            The measurement. ``m.R``, when not ``None``, overrides the model's
            noise covariance for this sample.
        model:
            Measurement model matching ``m``'s sensor.
        u:
            Input in effect at ``m.timestamp``. Required because ``h`` may
            depend on the input directly; see
            :class:`~erp.models.base.MeasurementModel`.
        """
