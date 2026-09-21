"""Locating the repo and the files inside it.

One root-finder for the whole project. Before ADR-0002 phase P1 there were
four: ``get_project_root`` here, plus a ``find_repo_root`` written separately
in ``mypalletizer260EKF.ipynb`` and ``viewer.ipynb`` and a third spelling as
``resolve_model_path`` in ``finger_imu_practice.ipynb``. They disagreed about
what a root even is -- the notebook versions searched for a *directory named*
``preERP`` first, which stops working the moment the folder is renamed or
cloned under another name, and only fell back to the marker file.

The rule here is the marker file and nothing else: the root is the nearest
ancestor holding ``pyproject.toml``. The search starts from this module rather
than from ``Path.cwd()``, so the answer does not depend on whether a notebook
was started from the repo root or from ``notebooks/``.

**Limitation, stated because it is load-bearing:** this works for an editable
install (``pip install -e .``, which is what every documented path in this repo
uses) and for running from a checkout. Installed non-editably into
``site-packages`` there is no ``pyproject.toml`` above the package and
:func:`repo_root` raises. That is deliberate -- returning a wrong root silently
would send every asset lookup somewhere plausible and empty.
"""

from pathlib import Path

__all__ = ["get_project_root", "repo_root", "resolve_model_path", "resolve_repo_path"]

# Resolved once, at import: the directory holding this module. Used as the
# default starting point for the upward search. A `Path(...)` call in the
# signature's default would be evaluated at import anyway, but ruff flags it
# (B008) because the pattern is a trap for mutable defaults.
_THIS_FILE = Path(__file__).resolve()


def repo_root(starting_path: Path | str | None = None) -> Path:
    """Nearest ancestor directory containing ``pyproject.toml``.

    ``starting_path`` defaults to this module's own location, so the answer
    does not depend on the caller's working directory. Pass ``Path.cwd()``
    only when you specifically mean "the repo the user is standing in".
    """
    start = _THIS_FILE if starting_path is None else Path(starting_path).resolve()
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError(
        f"Could not find the project root: no pyproject.toml at or above {start}. "
        "Install the package editably (pip install -e .) or run from a checkout."
    )


def resolve_repo_path(*parts: str | Path, must_exist: bool = True) -> Path:
    """Absolute path to something under the repo root, e.g. the arm XML.

    This is the general form and the one new code should use::

        resolve_repo_path("mechanical", "mujoco_assets",
                          "MyPalletizer260", "MyPalletizer260.xml")

    ``must_exist=False`` is for outputs that are about to be written.

    Raises
    ------
    FileNotFoundError
        When ``must_exist`` and nothing is there. The message carries the full
        path, because "file not found" against a path assembled from five
        fragments is otherwise unactionable.
    """
    path = repo_root().joinpath(*(Path(p) for p in parts))
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Not found in the repo: {path}")
    return path


def resolve_model_path(filename: str | Path, subfolder: str = "assets") -> Path:
    """Absolute path to a MuJoCo XML under ``notebooks/<subfolder>/``.

    The finger-notebook convenience, kept because
    ``notebooks/finger_imu_toolkit.ipynb`` resolves ``finger_2link.xml``
    through it. It only ever reaches ``notebooks/``, so it cannot find the
    palletizer XML under ``mechanical/`` -- for anything outside the finger
    assets use :func:`resolve_repo_path`.
    """
    return resolve_repo_path("notebooks", subfolder, filename)


# Pre-P1 name, kept so nothing that imported it breaks. `repo_root` is the
# spelling the rest of the package uses.
get_project_root = repo_root
