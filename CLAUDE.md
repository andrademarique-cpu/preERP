# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See [`README.md`](./README.md) for project context and
[`docs/adr/0001-multi-rate-fusion.md`](./docs/adr/0001-multi-rate-fusion.md)
for the design decisions the package is built on. This file is the
condensed set of rules that must hold for any change to be acceptable.

## Commands

Everything below assumes the **`erp` conda env**. The base Anaconda env
does not have the package installed and carries different tool versions.

```bash
# Setup (either path) -- from the REPO ROOT, note the "."; see caveat below
git lfs install
pip install -e ".[dev]"                 # add [app] for scripts/, [viz] for matplotlib
conda env create -f environment.yml     # or conda; installs -e .[dev,viz]

# Full local check, in the order CI runs it -- all three pass on the current tree
ruff check software/src software/tests  # rule set pinned via [tool.ruff.lint] select
mypy software/src                       # strict; clean (23 files)
pytest -q                               # 63 tests, ~7 s, incl. NEES/NIS consistency

# Single test / single case
pytest software/tests/test_consistency.py -q
pytest software/tests/test_fusion.py::test_late_measurements_are_discarded_and_counted -q

# Import-direction check (what CI greps; run manually before pushing)
grep -rEl "^\s*(from|import)\s+erp\.(sensors)" software/src/erp/estimators software/src/erp/models
# any match is a violation -- the command should print nothing

# The live viewer app (needs the [app] extra: mujoco, pyqtgraph, PyQt5, pyserial)
python scripts/finger_viewer.py                    # synthetic input, no hardware
python scripts/finger_viewer.py --pot --simulate   # pot GUI stack, no hardware
python scripts/finger_viewer.py --pot --port COM5  # real potentiometers
```

**Always install from the repo root, with the `.`.** There is no
`pyproject.toml` or `setup.py` under `software/`; the single packaging
file is at the root and already points at the nested sources
(`packages.find` → `software/src`, `testpaths` → `software/tests`). So
the project root is the repo root, and only the *sources* live under
`software/`. The `pip install -e "./software[dev]"` form that used to
appear in `README.md`, `AGENTS.md` and `ci.yml` is not a typo but an
impossibility — it is fixed everywhere now, so if you see it again,
something regressed.

**Dependencies are declared once, in `pyproject.toml`.** `environment.yml`
supplies only the interpreter and pip, then installs `-e .[dev,viz]`.
Do not re-add package lists to `environment.yml`: the two lists drifted
last time and disagreed about what an environment should contain. Extras:
`dev` = pytest/mypy/ruff, `viz` = matplotlib, `app` = mujoco/pyqtgraph/
PyQt5/pyserial for `scripts/` only. `app` is deliberately not installed
by default — the core and its tests need none of it.

**The ruff rule set is pinned by `[tool.ruff.lint] select`, not left to
the default.** This matters more than it sounds: ruff's *default* set
went from 59 rules in 0.12 to 413 in 0.16, so with no `select` the answer
to "does this tree lint clean?" depended on which ruff the developer
happened to have. It now reports identically on both. If lint fails,
fix the code — do not edit `select` to make it pass.

Python 3.10 is the floor (`requires-python`, `[tool.mypy] python_version`,
`[tool.ruff] target-version`, and the lower leg of the CI matrix). The
conda env is 3.11, so it will happily run 3.11-only syntax that CI
rejects; mypy and ruff catch it locally.

## What this project is

`erp` is a multi-sensor state estimation stack for robotic hands: KF /
EKF / UKF fusion of inertial and contact-force measurements. It is one of
three coupled tracks (software / mechanical / electronics & firmware) but
this repo's code work is almost entirely in `software/`.

**Non-goals — do not add these:** grasp planning, manipulation policy,
learning-based control, or hard real-time guarantees in the Python layer
(<1 kHz timing is firmware's job). If a task drifts toward these, flag it
instead of implementing it.

## The one rule that overrides everything else

> The mathematical core does not know that hardware exists.

`software/src/erp/estimators/`, `software/src/erp/models/` and
`software/src/erp/core/` must **never** import from
`software/src/erp/sensors/`, `firmware/`, or any ROS 2 package
(`ros2_ws/`). No exceptions for "just this once" or "to fix it quickly" —
this is checked in CI and enforced in review regardless of whether the
code works.

Why this matters concretely: every estimator must be runnable in a
notebook, in CI, and against recorded data with zero hardware attached.
The moment `estimators/` reaches into `sensors/`, that stops being true.

If you need a new measurement backend, add it as a `Sensor` subclass
(real driver, `ReplaySensor`, `DummySensor`, or an application-local one
like `QueueSensor` in `scripts/finger_viewer.py`); never route hardware
access through the core.

**Do not treat a green pipeline as proof you honored this rule.** The
check is a single grep for `^\s*(from|import)\s+erp\.(sensors)`, so it
only catches *absolute* imports of `erp.sensors`. It does not catch:

- relative imports — `from ..sensors import IMUSensor` passes CI;
- `firmware/` or ROS 2 imports (`import rclpy`), despite the rule text
  above forbidding both;
- indirect reach-through, e.g. importing a `calibration/` module that
  itself imports `sensors/`.

The rule is broader than its enforcement. Verify by reading the import
block. The tree is currently clean under the broader reading too.

## Contract types — frozen

Four ABCs define the entire system contract:

- `core/types.py` — `Measurement`, `ControlInput`, `GaussianState`, `CalibrationResult`
- `models/base.py` — `ProcessModel`, `MeasurementModel`
- `estimators/base.py` — `StateEstimator`
- `sensors/base.py` — `Sensor`

These are now implemented and **frozen**. Changing a signature on any of
the four requires explicit sign-off from the user, treated as equivalent
to "every module owner must approve." Do not casually add a parameter or
rename a method on an ABC while implementing something else — implement
against the existing signature, or stop and ask. Concrete subclasses
(`ExtendedKalmanFilter`, `ReplaySensor`, `LinearTimeInvariantModel`, …)
are normal code and don't carry this restriction.

**`README.md` § 5 still shows the pre-freeze signatures.** It now carries
a banner saying so, but the code block under it was left as the
historical sketch — read the source files, not the README, for the
current contract. ADR-0001 decision D1 amended all four. The differences
that bite:

| Now | README § 5 still says |
|---|---|
| `ProcessModel.Q(dt)` — a method | `@property Q` |
| `MeasurementModel.h(x, u)` / `.jacobian(x, u)` | `h(x)` / `jacobian(x)` |
| `StateEstimator.update(m, model, u)` | `update(m, model)` |
| `Sensor.read() -> Measurement \| None`, plus `drain()` | blocking `read() -> Measurement` |

Each change has a measured justification in ADR-0001 § 2; `Q(dt)` and the
`u` arguments are load-bearing, not generality for its own sake.

## Architecture: everything hangs off the event timeline

The single idea that explains the package is that **there is no tick**.
Reference generation, actuator command latching and each sensor run on
their own clocks, so the filter advances from event to event, where the
event set is the union of measurement timestamps and control-input change
times. Four pieces implement that, and they only make sense together:

1. **`core/timeline.py` — `InputHistory`.** Zero-order-hold input signal
   queryable at arbitrary times, plus an actuation transport delay.
   Stores *effective* times (latch + delay) computed once on push rather
   than subtracting the delay per query — subtracting reintroduces float
   error exactly at breakpoints, so `u_at` would silently return the
   previous input. `breakpoints_in` and `u_at` agreeing exactly is a hard
   requirement of the next piece.

2. **`fusion/engine.py` — `FusionEngine`.** All timestamp handling lives
   here, and nowhere else. `_advance_to` splits every prediction interval
   at input breakpoints so each sub-interval sees one constant `u`, which
   is what makes the ZOH discretisation exact rather than approximate.
   Late measurements (timestamp before filter time) are dropped and
   counted in `.discarded` — assert on that counter in tests, because a
   silent discard path is how a sensor quietly stops contributing while
   every plot still looks correct.

3. **`models/discretize.py` — `van_loan` / `zoh_input`.** Process noise
   is specified as a continuous-time **power spectral density** and
   discretised per segment. The resulting `Q` is *schedule invariant*:
   `predict(dt1)` then `predict(dt2)` equals `predict(dt1 + dt2)` to
   machine precision. A discrete white-noise (DWNA) construction is not,
   so under multi-rate operation adding a faster sensor would shrink the
   effective process noise for reasons unrelated to the physics.

4. **`models/linear.py` — `servoed_finger_model`.** State is
   `[q, v, a]` in **blocks, not interleaved** (angle rad, rate rad/s,
   servo activation rad), matching the notebook so vectors stay
   comparable. Carrying the servo activation `a` as state is not
   optional: without it the accelerometer channels pick up a
   deterministic, time-correlated bias that no `Q` or `R` inflation
   removes (NEES median 344 281 against a target of 4, versus 5.96
   against 6 once modelled).

Two rules follow, and any change touching `fusion/` or a process model
must preserve both:

- **Prediction advances to the next event, never by a fixed `dt`.**
  Fixed-step prediction under out-of-order arrivals silently corrupts the
  covariance — the trajectory still looks plausible, and NEES/NIS is what
  catches it.
- **Timestamp logic stays inside `FusionEngine`.** Converting a device
  clock into the single monotonic host time base is a `Sensor`
  responsibility; it must never appear in `fusion/`. Sensor latency and
  actuation delay are different quantities living in different layers
  (`Measurement.timestamp` is the *sample* instant; `buffer_horizon`
  absorbs reordering; `InputHistory` owns the actuation delay).

### Numerical choices that are load-bearing

- **`core/linalg.make_spd` after every operation on `P`.** Symmetrise and
  floor the eigenvalues. Without it NEES comes out *negative* and the
  consistency diagnostic fails silently — the Jacobian carries entries of
  order 700 and `P` spans ~10 orders of magnitude between its position
  and velocity blocks.
- **Joseph form** for the covariance update in `ExtendedKalmanFilter`,
  which survives round-off that plain `(I - KH) P` does not.
- **`np.linalg.solve`, never an explicit inverse.** Innovation
  covariances reach condition numbers around 1e11 near a stretched
  finger pose.
- **Propagate the full covariance block, never `sqrt(diag(P))`.**
  Dropping the cross-term barely moves the error-ellipse *area* while
  rotating it and inflating the worst-direction sigma — see
  `models/kinematics.tip_covariance` and `viz/ellipse`.

`ExtendedKalmanFilter` handed linear models reduces exactly to the
standard Kalman filter, so there is deliberately no separate
`KalmanFilter` class. The UKF named in README § 5 does not exist yet.

## Directory map / where things go

| Path | Contents | Import boundary |
|---|---|---|
| `software/src/erp/core/` | `types` (the four value types), `timeline` (`InputHistory`), `linalg` (`make_spd`, `nees`, `nis`) | no hardware imports |
| `software/src/erp/models/` | `base` ABCs, `discretize` (van Loan), `linear` (LTI + servoed finger), `measurement`, `kinematics` | no hardware imports |
| `software/src/erp/estimators/` | `base` ABC, `ekf` | no hardware imports |
| `software/src/erp/sensors/` | `Sensor` ABC, `ReplaySensor`, `DummySensor`; real drivers go here | may depend on core/models |
| `software/src/erp/calibration/` | calibration procedures — **empty so far** | may depend on sensors |
| `software/src/erp/fusion/` | `FusionEngine` — multi-rate sync only | orchestrates estimator + sensors |
| `software/src/erp/io/` | logging, dataset replay — **empty so far** | — |
| `software/src/erp/viz/` | ellipse geometry; returns points, imports no plotting backend | — |
| `software/tests/` | 63 tests incl. NEES/NIS consistency and its falsifications | — |
| `scripts/` | **applications, not package code** — `finger_viewer.py` (MuJoCo + PyQt live viewer). Extends `sys.path`, defines its own `Sensor` subclass. Reusable parts belong in `erp` | may import anything |
| `ros2_ws/src/erp_ros/` | ROS 2 nodes, msgs, launch files **only** — a thin wrapper, not where logic lives | wraps `erp`, never the reverse |
| `config/` | `estimation.yaml` — noise params, rates, geometry; new estimators/models must be registered here | — |
| `notebooks/` | exploratory work — the finger-IMU EKF (`finger_imu_practice.ipynb`), plus older KF/UKF and MuJoCo practice, and the potentiometer logger | may import anything; nothing imports it |
| `mechanical/`, `electronics/`, `firmware/` | hardware tracks — CAD, schematics, `potentiometer_logger.ino` | out of scope for Python-layer changes |
| `docs/theory/` `docs/hardware/` `docs/adr/` | filter derivations, assembly/wiring docs, architecture decision records | — |

`software/src/erp/__init__.py` is empty by design — import from the
subpackages (`from erp.models import servoed_finger_model`), which do
re-export their public names.

## Conventions to follow when writing code here

- **Type hints on all public signatures.** No exceptions; mypy runs
  `strict`.
- **Docstrings state units and reference frames** for any physical
  quantity (e.g. "linear acceleration, m/s², sensor frame"). This is not
  optional boilerplate — silent unit/frame mismatches are the main
  correctness risk in a sensor fusion codebase. The existing docstrings
  also record *why* a choice is load-bearing and what the failure looks
  like when it is not honored; match that, not bare descriptions.
- **A PSD is not a per-step sigma.** `psd_alpha` is rad²/s³; the
  notebook's `SIG_ALPHA = 12.0` is a per-step DWNA sigma in rad/s² valid
  only at a fixed 2 ms tick. Converting between them is a re-derivation,
  not a unit change — and the values in `config/estimation.yaml` are
  placeholders pending exactly that work.
- **New estimators or models must be registered in
  `config/estimation.yaml`**, not just instantiated ad hoc in code.
- **Tests target the ABC, not the implementation.** A new `Sensor` or
  `StateEstimator` should be exercised through its interface (ideally the
  same test works against a stub and the real implementation) — this is
  what proves the interface is sound.
- **Filter correctness is judged by NEES/NIS, never by eyeballing a
  trajectory**, and every consistency test must be paired with a
  deliberately-broken variant that **fails** it. See
  `test_consistency.py`: constant-`Q`, DWNA-`Q` and mis-scaled-`Q`
  filters are asserted to fail just as hard as the reference is asserted
  to pass. A consistency test that cannot fail is not evidence of
  anything.
- **Be precise about how a broken variant fails.** The constant-`Q`
  filter lands at ~0.69 of target — conservative, mildly wrong — while
  DWNA lands five orders of magnitude high. Overstating the mild one is
  the easiest way to get the ADR's reasoning quietly disbelieved later.
- Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`.
- Respect module ownership when suggesting reviewers or making
  cross-cutting changes — see `CODEOWNERS` (handles are still
  placeholders) and README § Team structure.
- `docs/theory/` is written in Spanish, `docs/adr/` in English. Match the
  language of the document you are editing.

## Current repo state

The package is implemented through ADR-0001 phase P4 and its test
obligations pass: `core/` (types, timeline, linalg), `models/` (van Loan
discretisation, servoed-finger LTI model, linear measurement models,
kinematics), `estimators/ekf.py`, `sensors/` (`ReplaySensor`,
`DummySensor`), `fusion/engine.py`, `viz/ellipse.py`, and the first
`config/estimation.yaml`. 63 tests pass in ~6 s.

Not yet built: the UKF, real sensor drivers, `calibration/`, `io/`, and
`ros2_ws/`. `test_smoke.py` is still a bare `assert True` from before the
suite existed.

Working tree has uncommitted changes: the packaging/install overhaul
(`pyproject.toml`, `environment.yml`, `.github/workflows/ci.yml`,
`README.md`, `AGENTS.md`, plus the mechanical lint fixes that the new
`select` list surfaced) and `scripts/finger_viewer.py`.

Known open items, from ADR-0001 § 8 — worth reading before tuning
anything:

- `psd_alpha` in `config/estimation.yaml` is a placeholder. Re-deriving
  it from the physics is the blocking item before these filters mean
  anything on real data.
- `actuation_delay_s` is unmeasured on the real chain.
- The servoed-finger model is stiff (`A_c` entries ~5e4). Composition
  holds at ~1e-16 up to ~20 ms gaps but degrades to ~1e-7 by 50 ms, so a
  sensor slower than ~20 Hz would need this revisited.
- `scripts/finger_viewer.py` tracks well but is **overconfident** (±2σ
  coverage ~10 %, not 95 %): `erp` integrates the coupled `(q, v, a)`
  system exactly while MuJoCo evaluates actuator force at a step
  endpoint, and that difference is deterministic, so process noise cannot
  absorb it. Raising `--psd-alpha` to force coverage is whitening a bias
  — the mistake `docs/theory/finger_imu_ekf.md` § 4 documents.

### Reference material

`notebooks/finger_imu_practice.ipynb` and its report
`docs/theory/finger_imu_ekf.md` are the validated design precedent the
package was built from; § 6 of the report lists the findings that
constrain it. `docs/adr/0001-multi-rate-fusion.md` is the authoritative
record of the multi-rate design, including measured numbers for every
decision and § 7's honest post-implementation refinement of its own
argument. The older notebooks (Kalman/UKF linear algebra, MuJoCo
practice) are reference only. Nothing under `notebooks/` or `scripts/` is
part of the `erp` package.
