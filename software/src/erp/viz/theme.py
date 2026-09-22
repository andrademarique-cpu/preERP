"""One look for every figure, and one colour per signal.

The colours are the point, not the styling. Three-way plots put plant truth, a
noisy measurement and the filter's estimate on the same axes, and a reader who
has to consult a legend to tell which is which will misread the figure. Truth is
the quiet reference, the measurement is the scatter around it, and the estimate
is the line that matters -- so they get, respectively, a neutral grey, a faded
marker colour and a saturated line colour, and they keep those roles in every
figure in the package.

matplotlib is imported at module scope here and in :mod:`erp.viz.figures`, and
neither is re-exported from ``erp/viz/__init__.py`` -- see that module for why.
"""

from __future__ import annotations

from typing import Any

import matplotlib as mpl

__all__ = ["COLORS", "apply_theme"]

COLORS: dict[str, str] = {
    "truth": "#4a4a4a",      # plant, the reference you are scored against
    "measured": "#d98c3f",   # sensor sample, noise included
    "estimated": "#2f6fb5",  # filter posterior
    "band": "#2f6fb5",       # the estimate's sigma band, same hue as the line
    "target": "#b5432f",     # a target value or an out-of-band marker
}

# Typed `Any`, not `dict[str, Any]`, and that is not laziness. matplotlib ships
# py.typed and types `rcParams` keys as a Literal union of every valid setting,
# which a `dict[str, ...]` cannot satisfy -- but CI installs .[dev] without
# [viz], so there mypy sees no matplotlib at all and `update` takes Any. A
# `type: ignore` would be needed locally and flagged as UNUSED in CI, failing
# the other half of the matrix. `Any` is the one annotation correct in both, and
# nothing flows back out of this dict: it goes into matplotlib's global state
# and is never read by our code.
_RC: Any = {
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.3,
}


def apply_theme() -> None:
    """Set the package's rcParams on the global matplotlib state.

    Global on purpose: a notebook draws a dozen figures across as many cells and
    threading a style object through each call is how half of them end up
    looking different. Call it once per session. It sets only the keys above, so
    anything the caller has already customised elsewhere survives.
    """
    mpl.rcParams.update(_RC)
