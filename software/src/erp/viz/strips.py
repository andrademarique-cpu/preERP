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

__all__ = ["RingBuffer"]


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
