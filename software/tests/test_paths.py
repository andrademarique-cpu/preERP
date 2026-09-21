"""Repo-root resolution: one root-finder, and it does not depend on the cwd.

Needs neither mujoco nor the LFS assets, so it stays in the fast suite.

Before ADR-0002 phase P1 there were four root-finders (see ``erp.io.paths``),
and the notebook versions searched for a *directory named* ``preERP`` before
falling back to the marker file. These tests pin the two properties that made
consolidating them worth doing: the answer comes from ``pyproject.toml``, and
it is the same wherever the caller happens to be standing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from erp.io.paths import get_project_root, repo_root, resolve_model_path, resolve_repo_path

PALLETIZER_XML = ("mechanical", "mujoco_assets", "MyPalletizer260", "MyPalletizer260.xml")


def test_repo_root_is_the_directory_holding_pyproject() -> None:
    root = repo_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "software" / "src" / "erp").is_dir()


def test_repo_root_does_not_depend_on_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The search starts at the module, not at ``Path.cwd()``.

    The notebook root-finders took ``Path.cwd()``, which is why running one
    from a different directory could resolve somewhere else entirely.
    """
    before = repo_root()
    monkeypatch.chdir(tmp_path)
    assert repo_root() == before


def test_resolve_repo_path_reaches_outside_notebooks() -> None:
    """The whole point of P1: the palletizer XML is under ``mechanical/``."""
    assert resolve_repo_path(*PALLETIZER_XML).is_file()


def test_resolve_repo_path_reports_the_full_path_when_missing() -> None:
    with pytest.raises(FileNotFoundError, match=r"nope\.xml"):
        resolve_repo_path("mechanical", "nope.xml")


def test_resolve_repo_path_allows_paths_that_do_not_exist_yet() -> None:
    """Outputs are resolved before they are written."""
    out = resolve_repo_path("data", "processed", "not-written-yet.npz", must_exist=False)
    assert out.parent.is_dir() and not out.exists()


def test_resolve_model_path_still_finds_the_finger_asset() -> None:
    """``finger_imu_toolkit.ipynb`` resolves its template through this."""
    assert resolve_model_path("finger_2link.xml").is_file()


def test_resolve_model_path_cannot_reach_the_palletizer() -> None:
    """Falsification, and the exact shape of the bug P1 worked around.

    ``resolve_model_path`` only ever looks under ``notebooks/<subfolder>/``, so
    it finds the finger model and not the arm. That is why the palletizer
    notebook carried its own root-finder. The function is kept for its one
    caller rather than widened, because widening it would make a
    ``notebooks/``-relative name silently resolve repo-wide.
    """
    with pytest.raises(FileNotFoundError):
        resolve_model_path("MyPalletizer260.xml")


def test_get_project_root_is_still_importable() -> None:
    """Pre-P1 name, aliased rather than deleted."""
    assert get_project_root() == repo_root()
