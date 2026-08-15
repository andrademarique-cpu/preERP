# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository. See [`README.md`](./README.md) for full
project context; this file is the condensed set of rules that must hold
for any change to be acceptable.

## Commands

```bash
# Setup (either path)
git lfs install
pip install -e "./software[dev]"        # pip
conda env create -f environment.yml     # or conda

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

The package lives at `software/`, not repo root (`pyproject.toml` points
`packages.find` at `software/src`, `testpaths` at `software/tests`) — run
the above from the repo root, not from inside `software/`.

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
| `notebooks/` | exploratory work (filter derivations, MuJoCo practice) — not production code | — |
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
structure, package skeleton (`__init__.py` files), and CI shell exist,
but the four ABCs themselves and their stubs (`DummySensor`,
`IdentityProcessModel`, etc.) are **not yet implemented** — that is the
next concrete task before any estimator work can begin. `notebooks/`
holds pre-existing exploratory work (Kalman/UKF linear algebra, MuJoCo
practice) that predates this structure and is reference material, not
part of the `erp` package.
