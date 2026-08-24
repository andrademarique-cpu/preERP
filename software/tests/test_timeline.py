"""Zero-order-hold input history: exact reconstruction, loud failures."""

from __future__ import annotations

import numpy as np
import pytest

from erp.core.timeline import InputHistory
from erp.core.types import ControlInput


def history(times: list[float], *, delay: float = 0.0) -> InputHistory:
    h = InputHistory(actuation_delay=delay)
    for i, t in enumerate(times):
        h.push(ControlInput(u=np.array([float(i)]), timestamp=t))
    return h


def test_holds_value_until_next_input() -> None:
    h = history([0.0, 0.10, 0.20])
    assert h.u_at(0.0) == pytest.approx(0.0)      # at a breakpoint: new value
    assert h.u_at(0.05) == pytest.approx(0.0)     # between: held
    assert h.u_at(0.0999) == pytest.approx(0.0)
    assert h.u_at(0.10) == pytest.approx(1.0)
    assert h.u_at(1e6) == pytest.approx(2.0)      # after the last: held forever


def test_query_before_first_input_raises() -> None:
    """Holding an undefined input backwards would fabricate actuation."""
    h = history([0.5])
    with pytest.raises(LookupError, match="no input in effect"):
        h.u_at(0.4)
    with pytest.raises(LookupError, match="empty"):
        InputHistory().u_at(0.0)


def test_timestamps_must_strictly_increase() -> None:
    h = history([0.0, 0.1])
    for bad in (0.1, 0.05):
        with pytest.raises(ValueError, match="strictly increase"):
            h.push(ControlInput(u=np.array([9.0]), timestamp=bad))


def test_breakpoints_are_half_open_at_the_low_end() -> None:
    """(t0, t1], so repeatedly advancing never applies a breakpoint twice."""
    h = history([0.0, 0.1, 0.2, 0.3])
    assert h.breakpoints_in(0.0, 0.25) == pytest.approx([0.1, 0.2])
    assert h.breakpoints_in(0.1, 0.2) == pytest.approx([0.2])
    assert h.breakpoints_in(0.2, 0.2) == []
    # Walking the timeline in two hops must visit each breakpoint exactly once.
    first, second = h.breakpoints_in(0.0, 0.15), h.breakpoints_in(0.15, 0.3)
    assert sorted(first + second) == pytest.approx([0.1, 0.2, 0.3])


def test_actuation_delay_shifts_effect_not_latch() -> None:
    """A command latched at t only reaches the plant at t + delay."""
    h = history([0.0, 0.1], delay=0.02)
    assert h.u_at(0.11) == pytest.approx(0.0)     # latched at 0.1, not yet effective
    assert h.u_at(0.13) == pytest.approx(1.0)     # now it is
    # 0.02 is a breakpoint too: it is when the *first* command takes effect.
    assert h.breakpoints_in(0.0, 0.2) == pytest.approx([0.02, 0.12])
    with pytest.raises(LookupError):
        h.u_at(0.019)                             # first command not yet in effect


@pytest.mark.parametrize("delay", [0.0, 0.02, 1e-3])
def test_querying_at_a_returned_breakpoint_gets_the_new_input(delay: float) -> None:
    """The round-trip guarantee FusionEngine depends on.

    The engine splits an interval at these breakpoints and then asks for the
    input on each side, so a query at a returned breakpoint must land on the
    input that breakpoint introduces. Applying the delay by subtraction at
    query time breaks exactly this: with a 20 ms delay, 0.12 - 0.02 evaluates
    to 0.09999999999999999 and the query silently returns the previous input.
    """
    h = history([0.0, 0.1, 0.2], delay=delay)
    breaks = h.breakpoints_in(0.0 + delay, 1.0)
    assert len(breaks) == 2
    for bp, expected in zip(breaks, (1.0, 2.0)):
        assert h.u_at(bp) == pytest.approx(expected)


def test_no_input_is_in_effect_before_the_first_takes_hold() -> None:
    """Delay leaves a window with no defined input, and it must not be papered over.

    A FusionEngine advancing from t=0 here hits the LookupError rather than
    integrating against a fabricated zero command.
    """
    h = history([0.0], delay=0.02)
    assert h.breakpoints_in(0.0, 0.05) == pytest.approx([0.02])
    with pytest.raises(LookupError, match="first takes effect at"):
        h.u_at(0.0)


def test_prune_keeps_the_input_still_in_effect() -> None:
    h = history([0.0, 0.1, 0.2, 0.3])
    assert h.prune_before(0.25) == 2
    assert len(h) == 2
    assert h.u_at(0.25) == pytest.approx(2.0)     # still correct after pruning
    assert h.prune_before(0.0) == 0               # nothing droppable


def test_rejects_inconsistent_input_dimension() -> None:
    h = history([0.0])
    with pytest.raises(ValueError, match="history holds size"):
        h.push(ControlInput(u=np.array([1.0, 2.0]), timestamp=0.1))


def test_returned_input_is_not_writable() -> None:
    """Stored arrays are handed out by reference; mutation must not corrupt them."""
    h = history([0.0])
    with pytest.raises(ValueError):
        h.u_at(0.0)[0] = 99.0
