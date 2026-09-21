"""A simulated arm that opens no port, so the whole loop runs unplugged."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from erp.core.types import Array
from erp.robot.base import ArmInterface, JointMap

__all__ = ["DryRunArm"]


class DryRunArm(ArmInterface):
    """Simulated arm. Opens no serial port and moves no hardware.

    Models the firmware as a first-order lag toward the **delayed** setpoint,
    with a speed cap. It does not claim to be the robot: it exists so that
    everything else -- the streaming loop, the log, the plots, the lag
    estimator -- can be exercised and falsified with the arm unplugged.

    **The first-order lag is not cosmetic.** With a pure zero-order hold
    against the delayed setpoint, the transport delay (80 ms) beats against
    the send period (40 ms) and the readback alternates between two setpoints:
    the plot shows TWO PARALLEL BANDS that look like a logging bug and are
    not. A servo with position feedback does not do that.

    Parameters
    ----------
    jmap:
        Converts commanded model radians to the API degrees this arm reports.
    latency_s:
        Transport delay, s: setpoint accepted -> motion begins.
    tau_s:
        First-order lag time constant, s.
    rate_limit_dps:
        Speed cap, deg/s.
    resolution_deg:
        Readback quantisation. 0.01 deg is the real one: the API transmits
        ``int(degrees * 100)`` (``_angle2int``).
    clock:
        Monotonic host clock, seconds, with the same signature and default as
        ``SimSensor``'s. Injectable because the lag is only testable if time
        can be driven, and ``perf_counter`` cannot be. Note this is *not* an
        :class:`~erp.sensors.clock.HostClock`, which converts a device
        timestamp to host time and is a different job entirely.
    """

    def __init__(
        self,
        jmap: JointMap,
        *,
        latency_s: float = 0.08,
        tau_s: float = 0.06,
        rate_limit_dps: float = 120.0,
        resolution_deg: float = 0.01,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if tau_s <= 0.0:
            raise ValueError("tau_s must be > 0; a zero lag is the ZOH this class exists to avoid")
        if resolution_deg <= 0.0:
            raise ValueError("resolution_deg must be > 0")
        self.jmap = jmap
        self.latency_s = float(latency_s)
        self.tau_s = float(tau_s)
        self.rate_limit_dps = float(rate_limit_dps)
        self.resolution_deg = float(resolution_deg)
        self._clock: Callable[[], float] = time.perf_counter if clock is None else clock
        self._queue: deque[tuple[float, Array]] = deque()
        self._setpoint = np.zeros(4)
        self._state = np.zeros(4)  # starts at `home` = [0, 0, 0] deg
        self._t_prev = float(self._clock())
        self.name = (
            f"DryRunArm(latency={latency_s * 1e3:.0f} ms, tau={tau_s * 1e3:.0f} ms, "
            f"vmax={rate_limit_dps:.0f} deg/s)"
        )

    def _advance(self, now: float) -> None:
        """Release any setpoints whose delay has elapsed, then integrate."""
        while self._queue and self._queue[0][0] <= now:
            self._setpoint = self._queue.popleft()[1]
        dt = max(0.0, now - self._t_prev)
        self._t_prev = now
        move = (self._setpoint - self._state) * (1.0 - np.exp(-dt / self.tau_s))
        cap = self.rate_limit_dps * dt
        self._state = self._state + np.clip(move, -cap, cap)

    def send(self, q_rad: npt.ArrayLike, speed: int | None = None) -> None:
        """Queue one setpoint. ``speed`` is accepted and ignored.

        The API's ``speed`` is a firmware scale from 1 to 100, not deg/s, so
        there is nothing meaningful to do with it here; the speed cap is
        :attr:`rate_limit_dps`.
        """
        now = float(self._clock())
        self._advance(now)
        self._queue.append((now + self.latency_s, self.jmap.to_api_deg(q_rad)[0]))

    def read(self) -> Array | None:
        """(4,) API degrees, quantised. Never ``None`` -- this arm always replies."""
        self._advance(float(self._clock()))
        out = np.round(self._state / self.resolution_deg) * self.resolution_deg
        return np.asarray(out, dtype=np.float64)

    def close(self) -> None:
        """No transport to release."""
