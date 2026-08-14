# AGENTS — AI Coding Assistant Guide

Purpose
-------
This short file tells an AI coding agent what it needs to know to be
productive in this repository without re-reading large docs. Prefer
linking to existing documentation rather than copying it.

Immediate reads
---------------
- README.md — project overview, class contracts, and quick start
- CLAUDE.md — contributor & agent conventions
- pyproject.toml — dependency/dev commands and test config
- .github/workflows/ci.yml — CI checks the agent must preserve

Key rules for any automated change
-------------------------------
- The mathematical core does not depend on hardware. `estimators/`
  and `models/` must not import from `sensors/`, `firmware/`, or
  any ROS 2 packages. CI enforces this (import-direction check).
- Four ABCs are the contract: change their public signatures only
  with explicit human approval. Those files are the canonical
  source-of-truth for interface changes.
- Tests target the ABC, not an implementation. If you add a new
  `Sensor`/`Estimator`/`Model`, include a test that exercises the
  appropriate interface level.
- Type hints on all public signatures and docstrings that state
  units and reference frames are required for any merged change.
- Register new estimators or models in `config/` YAML rather than
  instantiating them ad-hoc in code.

Quick developer commands
------------------------
```bash
git lfs install
pip install -e "./software[dev]"
pytest -q
ruff check .
mypy
```

CI and checks (what an agent must preserve)
-----------------------------------------
- `pytest` — unit and NEES/NIS consistency tests
- `mypy` — strict typing; `mypy_path` configured in `pyproject.toml`
- linting (`ruff`) and formatting rules
- import-direction check in CI that forbids core→hardware imports

What to do when unsure
-----------------------
- Stop and ask: any change to an ABC signature, or any cross-boundary
  import, requires human sign-off. Propose a small, focused PR that
  explains the rationale and tests.

Suggested follow-ups (optional)
-------------------------------
- Add a small `skill` that runs CI subset locally (import-direction,
  mypy, pytest) and reports actionable failures.
- Add a `skill` that knows where the four ABCs live and rejects
  candidate changes that touch their signatures without a flag.

Links
-----
- README: ./README.md
- CLAUDE: ./CLAUDE.md
- Project config: ./pyproject.toml
- Tests path: ./software/tests
