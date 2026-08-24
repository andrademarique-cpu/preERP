"""Process and measurement model contracts.

Two of the four ABCs that define the system contract. Changing a signature
here requires sign-off from every module owner (see ``CLAUDE.md``); the
current signatures are those adopted in ``docs/adr/0001-multi-rate-fusion.md``
decision D1.

This module must never import from :mod:`erp.sensors`, ``firmware`` or ROS 2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from erp.core.types import Array

__all__ = ["MeasurementModel", "ProcessModel"]


class ProcessModel(ABC):
    """Discretisation of ``dx/dt = f(x, u)`` over an arbitrary interval.

    Every method takes ``dt`` explicitly. Multi-rate fusion segments each
    prediction at measurement arrivals *and* control-input changes, so no
    implementation may assume a fixed step.
    """

    @abstractmethod
    def predict(self, x: Array, u: Array, dt: float) -> Array:
        """Propagate the state mean over ``dt`` seconds.

        ``u`` is constant across the interval: the fusion layer guarantees it
        by splitting at input breakpoints, which makes a zero-order-hold
        discretisation exact rather than approximate.

        Parameters
        ----------
        x:
            State mean, units per the concrete model.
        u:
            Input held over the whole interval.
        dt:
            Interval length in seconds, strictly positive.
        """

    @abstractmethod
    def jacobian(self, x: Array, u: Array, dt: float) -> Array:
        """Return ``d(predict)/dx`` evaluated at ``(x, u)`` over ``dt``.

        Shape ``(n, n)``. For a linear model this is the discrete-time state
        transition matrix and does not depend on ``x`` or ``u``.
        """

    @abstractmethod
    def Q(self, dt: float) -> Array:
        """Return the process-noise covariance accumulated over ``dt``.

        A method rather than a property because the covariance genuinely
        depends on the interval, and under multi-rate operation the interval is
        set by the sensor schedule.

        Implementations must be **schedule invariant**: propagating across
        ``dt1`` then ``dt2`` must give the same covariance as one step of
        ``dt1 + dt2``. A discrete white-noise (DWNA) construction does *not*
        satisfy this -- splitting an interval N ways divides the accumulated
        velocity variance by N -- so specify noise as a continuous-time
        spectral density and discretise with :func:`erp.models.discretize.van_loan`.
        """


class MeasurementModel(ABC):
    """Map state to expected sensor reading.

    ``h`` takes ``u`` as well as ``x``. This is not defensive generality: an
    accelerometer reads ``qacc``, which depends on the torque the servo is
    applying right now, so ``D = dh/du`` is non-zero on accelerometer channels
    and exactly zero on gyros and encoders. Forming an innovation as
    ``z - h(x)`` biases every accelerometer channel -- measured at 4.01 m/s^2,
    roughly 80 sigma, in ``docs/theory/finger_imu_ekf.md`` section 4.
    """

    @abstractmethod
    def h(self, x: Array, u: Array) -> Array:
        """Return the reading expected in state ``x`` under input ``u``.

        Units and reference frame are the sensor's own, e.g. m/s^2 in the
        sensor frame for an accelerometer.
        """

    @abstractmethod
    def jacobian(self, x: Array, u: Array) -> Array:
        """Return ``dh/dx`` at ``(x, u)``, shape ``(m, n)``.

        This is the *total* derivative with respect to the current state,
        including any path through acceleration. Partials that treat position,
        velocity and acceleration as independent are a different quantity and
        are not the ``H`` an EKF needs.
        """

    @property
    @abstractmethod
    def R(self) -> Array:
        """Measurement-noise covariance, units of the reading squared.

        Stays a property: sensor noise does not depend on the prediction
        interval. Per-sample overrides travel on ``Measurement.R``.
        """
