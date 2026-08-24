"""Zero-order-hold control-input history, queryable at arbitrary times.

This is the piece that lets prediction run on the *event* timeline rather than
on any single fixed rate. Reference generation, actuator command latching and
sensor sampling all run on different clocks; the estimator only ever needs to
know what ``u`` was in effect over the interval it is integrating across.

See ``docs/adr/0001-multi-rate-fusion.md`` decisions D2 and D4.
"""

from __future__ import annotations

from bisect import bisect_right

import numpy as np

from erp.core.types import Array, ControlInput

__all__ = ["InputHistory"]


class InputHistory:
    """Piecewise-constant input signal with an optional actuation delay.

    Inputs are latched at their timestamps and held until the next one, which
    is what the hardware actually does, so reconstruction here is exact rather
    than interpolated.

    The delay models bus/driver transport: a command latched at ``t`` only
    reaches the plant at ``t + actuation_delay``. It is kept deliberately
    separate from sensor latency, which is a property of a measurement's
    timestamp instead -- conflating the two is the mistake this class exists to
    prevent.
    """

    def __init__(self, *, actuation_delay: float = 0.0) -> None:
        """
        Parameters
        ----------
        actuation_delay:
            Seconds between latching a command and it taking effect at the
            plant. Must be non-negative.
        """
        if actuation_delay < 0.0:
            raise ValueError(f"actuation_delay must be >= 0, got {actuation_delay}")
        self._delay = float(actuation_delay)
        # Stored times are *effective* times (latch + delay), computed once on
        # push rather than by subtracting the delay at query time. Subtracting
        # per query reintroduces float error at exactly the wrong place: with a
        # 20 ms delay, 0.12 - 0.02 evaluates to 0.09999999999999999 and a query
        # at a breakpoint silently returns the previous input. Storing the sum
        # makes breakpoints_in and u_at agree exactly, which is the guarantee
        # FusionEngine relies on when it splits an interval and then asks for
        # the input on each side.
        self._t: list[float] = []
        self._u: list[Array] = []

    @property
    def actuation_delay(self) -> float:
        """Seconds between command latch and effect at the plant."""
        return self._delay

    def __len__(self) -> int:
        return len(self._t)

    def push(self, ci: ControlInput) -> None:
        """Append an input. Timestamps must strictly increase.

        Strictness is deliberate: two inputs sharing a timestamp make the held
        value ambiguous, and silently keeping one of them is exactly the kind
        of quiet timeline corruption that later reads as model error.
        """
        u = np.asarray(ci.u, dtype=np.float64).copy()
        if u.ndim != 1:
            raise ValueError(f"u must be 1-D, got shape {u.shape}")
        if self._u and u.size != self._u[-1].size:
            raise ValueError(f"u has size {u.size}, history holds size {self._u[-1].size}")
        t_effective = float(ci.timestamp) + self._delay
        if self._t and t_effective <= self._t[-1]:
            raise ValueError(
                f"input timestamps must strictly increase: got {ci.timestamp} "
                f"after {self._t[-1] - self._delay}"
            )
        u.flags.writeable = False  # stored arrays are handed out by reference
        self._t.append(t_effective)
        self._u.append(u)

    def u_at(self, t: float) -> Array:
        """Return the input in effect at time ``t``.

        Querying at a time returned by :meth:`breakpoints_in` is guaranteed to
        return the input that breakpoint introduces, with no float slop.

        The returned array is read-only and owned by the history; copy it
        before mutating.

        Raises
        ------
        LookupError
            If ``t`` precedes the first input's effective time. Holding an
            undefined input backwards would fabricate actuation that never
            happened, so this fails loudly instead.
        """
        i = bisect_right(self._t, t) - 1
        if i < 0:
            if not self._t:
                raise LookupError("input history is empty")
            raise LookupError(
                f"no input in effect at t={t}: first takes effect at {self._t[0]}"
            )
        return self._u[i]

    def breakpoints_in(self, t0: float, t1: float) -> list[float]:
        """Return effective times where ``u`` changes, in the interval ``(t0, t1]``.

        Half-open at the low end so that repeatedly advancing an estimator
        never applies the same breakpoint twice.

        These are the points a prediction interval must be split at: ``u`` is
        piecewise constant, so integrating across a change with a single value
        is a modelling error, not a rounding one.
        """
        if t1 < t0:
            raise ValueError(f"t1 ({t1}) must be >= t0 ({t0})")
        return self._t[bisect_right(self._t, t0) : bisect_right(self._t, t1)]

    def prune_before(self, t: float) -> int:
        """Drop inputs no longer reachable by a query at or after ``t``.

        The input in effect at ``t`` is always retained, so :meth:`u_at` stays
        correct for every query at or after ``t``. Bounds memory in a
        long-running realtime session.

        Returns
        -------
        Number of entries dropped.
        """
        i = bisect_right(self._t, t) - 1
        if i <= 0:
            return 0
        del self._t[:i]
        del self._u[:i]
        return i
