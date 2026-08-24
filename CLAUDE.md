# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository. See [`README.md`](./README.md) for full
project context; this file is the condensed set of rules that must hold
for any change to be acceptable.

## Commands

```bash
# Setup (either path)
git lfs install
pip install -e ".[dev]"                 # pip — note the "."; see caveat below
conda env create -f environment.yml     # or conda (same caveat)

# Full local check, mirrors .github/workflows/ci.yml
ruff check software/src software/tests  # lint
mypy software/src                       # strict type check
pytest -q                               # tests, incl. NEES/NIS consistency checks

# Single test
pytest software/tests/test_smoke.py -q
pytest software/tests/test_smoke.py::test_smoke -q

# Import-direction check (what CI enforces; run manually to verify before pushing)
grep -rEl "^\s*(from|import)\s+erp\.(sensors)" software/src/erp/estimators software/src/erp/models
# any match here is a violation — the command should print nothing
```

**Setup caveat — the widely-copied install command is broken.** Several
files still say `pip install -e "./software[dev]"`. That command *cannot
work*: there is no `pyproject.toml` or `setup.py` under `software/`. The
single packaging file is at the repo root and already points at the
nested sources (`packages.find` → `software/src`, `testpaths` →
`software/tests`). So the project root is the repo root; only the
*sources* live under `software/`. Always install from the repo root with
`pip install -e ".[dev]"`.

The stale command is currently in `README.md` (§4 and "Getting started"),
`AGENTS.md`, `environment.yml` (as `- -e ./software`), and — importantly
— the `Install package` step of `.github/workflows/ci.yml`. Treat those
as known-bad until fixed; do not copy the command out of them.

Two other stale commands in `AGENTS.md`: bare `mypy` fails (no target is
configured in `pyproject.toml` — use `mypy software/src`), and
`ruff check .` is not what CI runs. The `## Commands` block above is the
authoritative list; all of it is verified to pass on the current tree.

## What this project is

`erp` is a multi-sensor state estimation stack for robotic hands: KF /
EKF / UKF fusion of accelerometer and contact-force measurements. It is
one of three coupled tracks (software / mechanical / electronics &
firmware) but this repo's code work is almost entirely in `software/`.

**Non-goals — do not add these:** grasp planning, manipulation policy,
learning-based control, or hard real-time guarantees in the Python layer
(<1 kHz timing is firmware's job). If a task drifts toward these, flag it
instead of implementing it.

## The one rule that overrides everything else

> The mathematical core does not know that hardware exists.

`software/src/erp/estimators/` and `software/src/erp/models/` must
**never** import from `software/src/erp/sensors/`, `firmware/`, or any
ROS 2 package (`ros2_ws/`). No exceptions for "just this once" or "to fix
it quickly" — this is checked in CI (import-direction check in
`.github/workflows/ci.yml`) and enforced in review regardless of whether
the code works.

Why this matters concretely: every estimator must be runnable in a
notebook, in CI, and against recorded data with zero hardware attached.
The moment `estimators/` reaches into `sensors/`, that stops being true.

If you need a new measurement backend, add it as a `Sensor` subclass
(real driver, `ReplaySensor`, or `SimulatedSensor` — see below); never
route hardware access through the core.

**Do not treat green CI as proof you honored this rule.** The check is a
single grep for `^\s*(from|import)\s+erp\.(sensors)`, so it only catches
*absolute* imports of `erp.sensors`. It does not catch:

- relative imports — `from ..sensors import IMUSensor` passes CI;
- `firmware/` or ROS 2 imports (`import rclpy`), despite the rule text
  above forbidding both;
- indirect reach-through, e.g. importing a `calibration/` module that
  itself imports `sensors/`.

The rule is broader than its enforcement. Verify by reading the import
block, not by watching the pipeline go green.

## Contract types — treat as frozen unless told otherwise

Four ABCs define the entire system contract:

- `core/types.py` — `Measurement`, `GaussianState`
- `models/base.py` — `ProcessModel`, `MeasurementModel`
- `estimators/base.py` — `StateEstimator`
- `sensors/base.py` — `Sensor`

**Changing a signature on any of these four requires explicit sign-off
from the user, treated as equivalent to "every module owner must
approve."** Do not casually add a parameter or rename a method on an ABC
while implementing something else — implement against the existing
signature, or stop and ask. Concrete subclasses (`KalmanFilter`,
`IMUSensor`, `ConstantAccelModel`, etc.) are normal code and don't carry
this restriction.

## Directory map / where things go

| Path | Contents | Import boundary |
|---|---|---|
| `software/src/erp/core/` | `State`, `Measurement`, base types | no hardware imports |
| `software/src/erp/models/` | process + measurement models | no hardware imports |
| `software/src/erp/estimators/` | KF, EKF, UKF | no hardware imports |
| `software/src/erp/sensors/` | `Sensor` interface, real drivers, `ReplaySensor`, `SimulatedSensor` | may depend on core/models |
| `software/src/erp/calibration/` | calibration procedures | may depend on sensors |
| `software/src/erp/fusion/` | `FusionEngine` — multi-rate sync only | orchestrates estimator + sensors |
| `software/src/erp/io/` | logging, dataset replay | — |
| `software/src/erp/viz/` | plotting/visualization | — |
| `software/tests/` | tests, incl. NEES/NIS consistency checks | — |
| `ros2_ws/src/erp_ros/` | ROS 2 nodes, msgs, launch files **only** — a thin wrapper, not where logic lives | wraps `erp`, never the reverse |
| `config/` | YAML: noise params, geometry, calibration; new estimators/models must be registered here | — |
| `notebooks/` | exploratory work — the finger-IMU EKF (`finger_imu_practice.ipynb`), plus older KF/UKF and MuJoCo practice. Not production code, but see "Current repo state" | may import anything; nothing imports it |
| `mechanical/`, `electronics/`, `firmware/` | hardware tracks — CAD, schematics, MCU firmware | out of scope for Python-layer changes |
| `docs/theory/` `docs/hardware/` `docs/adr/` | filter derivations, assembly/wiring docs, architecture decision records | — |

## Multi-rate fusion rule

In `FusionEngine.step`, prediction **always** advances to the timestamp
of the next measurement being processed — never by a fixed `dt`. Any
change touching `fusion/` must preserve this: fixed-step prediction under
out-of-order sensor arrivals silently corrupts the covariance (the
trajectory still looks plausible; NEES/NIS is what catches it). If you
touch timestamp handling for one sensor, keep it inside `FusionEngine`,
not bolted onto an individual `Sensor` subclass.

## Conventions to follow when writing code here

- **Type hints on all public signatures.** No exceptions.
- **Docstrings state units and reference frames** for any physical
  quantity (e.g. "linear acceleration, m/s², sensor frame"). This is not
  optional boilerplate — silent unit/frame mismatches are the main
  correctness risk in a sensor fusion codebase.
- **New estimators or models must be registered in the relevant
  `config/` YAML**, not just instantiated ad hoc in code.
- **Tests target the ABC, not the implementation.** A new `Sensor` or
  `StateEstimator` should be exercised through its interface (ideally the
  same test works against both a stub and the real implementation) — this
  is what proves the interface is sound, not just the concrete class.
- **Filter correctness is judged by NEES/NIS consistency checks**, not by
  eyeballing a trajectory plot. If you touch an estimator or model, run
  or add the relevant consistency test rather than a qualitative sanity
  plot alone.
- Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`.
- Respect module ownership when suggesting reviewers or making
  cross-cutting changes — see `CODEOWNERS` and README § Team structure.
  A change spanning another owner's directory across the core/hardware
  boundary is exactly the case CI's import-direction check exists to
  catch.

## Current repo state

This repository is at the Sprint-0 scaffolding stage: the directory
structure, package skeleton, and CI shell exist, but the four ABCs
themselves and their stubs (`DummySensor`, `IdentityProcessModel`, etc.)
are **not yet implemented** — that is the next concrete task before any
estimator work can begin.

Concretely, every `__init__.py` under `software/src/erp/` is empty (0
bytes), and the entire test suite is `test_smoke.py` (`assert True`). So
`ruff`/`mypy`/`pytest` all pass, but they are passing over ~no code:
**green CI here is not yet evidence of anything.** None of the classes
named in this file or in `README.md` § 5 exist yet — treat those as the
specification to build against, not as code to import.

The real work to date is in `notebooks/`, which is *not* just stale
pre-existing material. `notebooks/finger_imu_practice.ipynb` (with its
companion report `docs/theory/finger_imu_ekf.md`) is a worked 2-link
finger EKF on MuJoCo, and it is the closest thing the repo has to a
validated design precedent. Read the report before implementing
estimators — § 6 lists findings that are meant to constrain the package,
including: process noise `Q` cannot absorb a deterministic model bias
(augment the state or drop the contaminated sensor instead); covariance
must propagate as a full matrix, since dropping cross-terms barely moves
the error-ellipse *area* while corrupting its *orientation*; a
one-timestep misalignment between state, measurement, and control
masquerades as model error; and `P` needs an eigenvalue floor or NEES
goes negative and the diagnostic fails silently. The older notebooks
(Kalman/UKF linear algebra, MuJoCo practice) are reference material.
Nothing under `notebooks/` is production code or part of the `erp`
package.

`config/` is empty apart from `.gitkeep`, so the "register new estimators
in `config/` YAML" convention has no existing file to follow — you will
be creating the first one.

Note that `docs/theory/` is written in Spanish; match the language of the
document you are editing.
