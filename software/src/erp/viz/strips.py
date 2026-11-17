"""Fixed-capacity history for a live strip chart. Pure numpy, no backend.

Split from :mod:`erp.viz.live` for the same reason :mod:`erp.viz.geometry` is
split from :mod:`erp.viz.figures`: this half has arithmetic that can be wrong
and is worth testing, and the other half needs a display. An interactive run
lasts as long as someone holds a key down, so an unbounded list is not an
option -- the buffer has to drop the oldest sample rather than the newest, and
a strip that silently stopped updating because it filled up is the failure this
exists to prevent.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from erp.core.types import Array

__all__ = ["ContactTrace", "RingBuffer"]


class RingBuffer:
    """The last ``capacity`` samples of a ``width``-channel signal, in order.

    Overwrites oldest-first once full. :meth:`view` always returns the samples
    oldest to newest, which is what a plot needs and is *not* what the
    underlying storage holds once the buffer has wrapped.
    """

    def __init__(self, capacity: int, width: int) -> None:
        """
        Parameters
        ----------
        capacity:
            Samples retained. At a 30 Hz strip, 1800 is a minute of history.
        width:
            Channels per sample. ``0`` is allowed -- a buffer of bare
            timestamps is useful for marking update instants on a strip.
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        if width < 0:
            raise ValueError(f"width must be >= 0, got {width}")
        self.capacity = int(capacity)
        self.width = int(width)
        self._t = np.zeros(self.capacity, dtype=np.float64)
        self._y = np.zeros((self.capacity, self.width), dtype=np.float64)
        self._i = 0      # next write position
        self._n = 0      # samples held, <= capacity

    def __len__(self) -> int:
        return self._n

    @property
    def full(self) -> bool:
        return self._n == self.capacity

    def append(self, t: float, y: npt.ArrayLike = ()) -> None:
        """Add one sample at time ``t`` (seconds, host base).

        ``y`` must have ``width`` entries. A mismatch raises rather than
        broadcasting: a strip fed the wrong number of channels would plot
        something plausible and wrong.
        """
        y_a = np.asarray(y, dtype=np.float64).reshape(-1)
        if y_a.size != self.width:
            raise ValueError(f"y has {y_a.size} entries, width is {self.width}")
        self._t[self._i] = t
        if self.width:
            self._y[self._i] = y_a
        self._i = (self._i + 1) % self.capacity
        self._n = min(self._n + 1, self.capacity)

    def view(self) -> tuple[Array, Array]:
        """``(t, Y)`` for the samples held, oldest first.

        Returns copies, not views into the storage: the caller hands these
        straight to a plotting backend that may keep the reference, and the
        next :meth:`append` would otherwise rewrite a point already on screen.
        """
        if not self.full:
            return self._t[: self._n].copy(), self._y[: self._n].copy()
        order = np.r_[self._i : self.capacity, 0 : self._i]
        return self._t[order], self._y[order]

    def clear(self) -> None:
        """Forget everything held; capacity and width are unchanged."""
        self._i = 0
        self._n = 0


class ContactTrace:
    """The histories behind the FSR / contact strip, in SI units. Pure numpy.

    Three things are kept apart because they arrive at three different rates
    and mean three different things:

    * the **state**, pushed once per render tick: the plant's fingertip force
      (N, the ``<touch>``), the plant's signed distance ``d(x)`` from the dome
      to the nearest known solid and the estimate's ``d(x_hat)`` (m, negative
      inside). On hardware there is no plant, so ``force`` and ``d_true`` are
      NaN, which a strip draws as a gap rather than as a zero that looks like
      "free";
    * the **measurement**, ~20 Hz, at its sample instant: the force the sensor
      reported, N;
    * the **corrections** the filter actually applied, split by kind, at the
      estimate's ``d`` *before* the correction (m) -- which is the point: an
      ON marker sits where the estimate was when the FSR pulled it to the
      surface, an OFF marker where it was when the FSR pushed it out.

    An OFF reading with the estimate outside the solid is not a correction
    (``EKFFSR`` leaves ``x`` and ``P`` untouched) and must not be pushed here.
    """

    def __init__(self, capacity: int) -> None:
        self.state = RingBuffer(capacity, 3)        # force N, d_true m, d_est m
        self.measured = RingBuffer(capacity, 1)     # force N
        self.on = RingBuffer(capacity, 1)           # d_before m
        self.off = RingBuffer(capacity, 1)          # d_before m

    def push_state(self, t: float, force_n: float, d_true_m: float, d_est_m: float) -> None:
        """Plant force (N), plant ``d`` and estimated ``d`` (m) at ``t`` (s)."""
        self.state.append(t, (force_n, d_true_m, d_est_m))

    def push_measurement(self, t: float, force_n: float) -> None:
        """One FSR sample, N, stamped at its sample instant."""
        self.measured.append(t, (force_n,))

    def push_correction(self, t: float, on: bool, d_before_m: float) -> None:
        """A contact correction ``EKFFSR`` applied at ``t``: ON or OFF-push."""
        (self.on if on else self.off).append(t, (d_before_m,))
