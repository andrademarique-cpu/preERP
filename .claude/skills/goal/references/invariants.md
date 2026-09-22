# Invariants a phase must not break

Each entry: the property, how to check it, and what the failure looks like.
Most of these are **not** enforced by CI. The ones that are, are enforced
narrowly enough to miss the realistic violation.

## Contents

- [1. Hardware enters through `Sensor`](#1-hardware-enters-through-sensor)
- [2. The golden run does not move](#2-the-golden-run-does-not-move)
- [3. MuJoCo stays behind three modules](#3-mujoco-stays-behind-three-modules)
- [4. One construction point for drivers](#4-one-construction-point-for-drivers-adr-0002--48)
- [5. No wall clock below `runtime/`](#5-no-wall-clock-below-runtime)
- [6. Time discipline](#6-time-discipline)
- [7. Numerical choices that are load-bearing](#7-numerical-choices-that-are-load-bearing)
- [8. Test conventions](#8-test-conventions)
- [9. Language, typing, docstrings](#9-language-typing-docstrings)
- [10. Things that are decided, not open](#10-things-that-are-decided-not-open)

---

## 1. Hardware enters through `Sensor`

`core/`, `models/`, `estimators/` and `sim/` must never import from `sensors/`,
`firmware/`, or any ROS 2 package. Every estimator has to be runnable in a
notebook, in CI, and against recorded data with zero hardware attached; the
moment `estimators/` reaches into `sensors/`, that stops being true.

CI's check, which is also the 3rd step of `ci.yml`:

```bash
grep -rEl "^\s*(from|import)\s+erp\.(sensors)" software/src/erp/estimators software/src/erp/models
# any output is a violation
```

It does not catch, and you must check by reading:

- **relative imports** — `from ..sensors import SerialIMUSensor` passes;
- **`sim/`, `core/`, `io/`, `fusion/`** — not scanned at all;
- **`firmware/` or ROS 2** (`import rclpy`), despite the rule text;
- **reach-through via a package `__init__`** — importing a module that itself
  imports `sensors/`. This is real, not hypothetical: `erp/io/config.py`
  imports `IMUDecoder`, which is why `erp/io/__init__.py` deliberately does not
  re-export it and `test_config.py` asserts the boundary in a subprocess.

The broader sweep:

```bash
grep -rEn "^\s*(from|import)\s+(erp\.sensors|\.\.sensors|rclpy)" \
  software/src/erp/estimators software/src/erp/models \
  software/src/erp/sim software/src/erp/core software/src/erp/io \
  software/src/erp/fusion
```

`calibration/` is the one to watch: the dependency runs `sensors/` →
`calibration/`, one way. Reversing it would put a hardware import one hop from
anything that calibrates.

New measurement backend? Add a `Sensor` subclass — `SerialIMUSensor` (live
device), `ReplaySensor` (log), `SimSensor` (recorded MuJoCo output), or a new
`StreamSensor` subclass for another live link.

## 2. The golden run does not move

`data/processed/golden_ekf_run.npz` is a frozen end-to-end run of the filter
over the committed IMU log. `python scripts/make_golden_run.py --check` re-runs
the pipeline and diffs it at `rtol=1e-12`.

Every phase so far has left it green — the trajectory generator, the blind-model
build, the rest state, the whole EKF, the filter loop, the rest-window
calibration and the config loader have all moved under it without shifting one
digit. That is the only reason any of it can be believed.

```bash
conda run -n EKF python scripts/make_golden_run.py --check
git status --short data/processed/ data/raw/     # both must be empty
```

When it fails, the first question is data or code. Answer it before reading the
estimator:

```bash
git show <last-good-rev>:data/raw/imu_trajectory_raw.csv   # LFS pointer -> oid
# fetch the object from .git/lfs/objects/<ab>/<cd>/<oid> into a shadow root,
# then: run_pipeline(root_dir=<shadow>) and diff against the fixture
```

If that reproduces the fixture, the code is innocent and the log changed.

Do **not** loosen `--rtol`, and do **not** regenerate the fixture to make a test
pass. Regenerating is a deliberate act: it must be committed together with the
log it was generated from, in one commit that says so, and it invalidates every
figure quoted from the fixture in CLAUDE.md and the ADRs.

Two known soft spots, worth stating rather than discovering: `rtol=1e-12` has
only ever been checked on Windows, and CI runs ubuntu on 3.10 and 3.11. A
cross-platform failure is plausible, and the honest fix is loosening to ~1e-9,
not regenerating per platform.

## 3. MuJoCo stays behind three modules

MuJoCo is *the model*, not hardware — deterministic, installs everywhere, needs
no device — which is why it is the deliberate exception to rule 1. Exactly three
modules touch the mujoco API:

```
software/src/erp/sim/mujoco.py       f/F/h/H, the paired forms, make_Q
software/src/erp/sim/plant.py        load_model, blind_variant, sensor_layout
software/src/erp/sensors/mujoco.py   rows_of, make_R
```

```bash
grep -rlE "^\s*(import mujoco|from mujoco)" software/src/erp
# must list exactly those three
```

`erp.sim.mujoco` is the only module allowed to call `mj.*` for the dynamics, and
every function there wraps its result in `np.asarray(..., dtype=np.float64)` or
`int(...)` before returning — mujoco ships no `py.typed`, so without that
wrapping `Any` leaks into the estimator under strict mypy. When adding physics,
add it there and wrap it in a class elsewhere.

`models/` and `core/` are mujoco-free and must stay that way: that is what makes
the EKF-versus-closed-form-Kalman test possible with mujoco absent from
`sys.modules` entirely.

Do not cite ADR-0001 § D5 ("MuJoCo stays behind `SimulatedSensor`") as current
policy — it was overruled by the rewrite and ADR-0001 was not updated. And do
not use this exception to justify another one.

## 4. One construction point for drivers (ADR-0002 § 4.8)

`erp/runtime/session.py` is the only module in the package permitted to name
`SimSensor`, `SerialIMUSensor`, `DryRunArm`, `MyPalletizerArm` or
`ReplaySensor`. Everything downstream takes a `Session` and never sees the mode.
That is what makes the M2 → M3 drop-in claim checkable rather than asserted.

§ 4.8 specifies this grep:

```bash
grep -rlE "SerialIMUSensor|MyPalletizerArm|DryRunArm|SimSensor|ReplaySensor" \
     software/src/erp | grep -vE "erp/(runtime/session|sensors/|robot/)"
```

**As written it fails on prose.** Six modules name a driver class in a docstring
while importing none — `calibration/rest.py`, `core/clock.py`,
`estimators/ekf.py`, `fusion/runner.py`, `io/config.py`, `io/log.py` — plus
their `__pycache__` entries, since the grep does not filter by extension.
Naming the class you are explaining why you do *not* construct is normal and
good. When implementing P7.5, restrict the grep to `--include="*.py"` and to
import statements and call sites. Checked on the current tree, the only real
hits are then `erp/robot/__init__.py` and `erp/sensors/{__init__,sim}.py`, all
already exempt.

## 5. No wall clock below `runtime/`

§ 4.8 specifies this grep, and it has the same prose problem as the one above —
run the filtered form:

Run it from bash — PowerShell mangles the parens.

```bash
grep -rnE '^[^#]*[^.[:alnum:]_`"'"'"']time\.(perf_counter|time|monotonic|sleep)\(' \
  --include="*.py" \
  software/src/erp/core software/src/erp/models software/src/erp/sim \
  software/src/erp/estimators software/src/erp/fusion \
  | grep -v "erp/core/clock.py"
# any output is a violation
```

Verified 2026-09-22: silent on the current tree, and it still catches a plain
`return time.perf_counter()`. The unfiltered version in § 4.8 matches
`core/types.py`'s docstring — which *documents* that timestamps are
`time.perf_counter()` seconds, i.e. states the very convention the rule enforces
— plus every `__pycache__` blob. The character class excludes a preceding
backtick or quote, which is what separates a call from prose about a call. If
you wire this into CI, wire in this form.

The companion driver-name grep for § 4, similarly verified silent:

```bash
grep -rlE '(^|[^A-Za-z_.])(SerialIMUSensor|MyPalletizerArm|DryRunArm|SimSensor|ReplaySensor)\(' \
  --include="*.py" software/src/erp \
  | grep -vE "erp/(runtime/session|sensors/|robot/)"
```

Anything that waits or timestamps takes a `Clock` (`core/clock.py`) or a
`Callable[[], float]`, so a test can drive it with `VirtualClock` — a 3 s
trajectory then runs in under 50 ms and scheduling becomes deterministic. This
is also what guarantees M1 and M2 share a code path: a runner that could read
the clock itself would behave differently under pacing.

Note `core.clock` (what time is it, when should the next thing happen) is a
different job from `sensors.clock` (when was this sample taken). Do not use one
where the other is meant.

## 6. Time discipline

- **One monotonic host time base, `time.perf_counter()`, absolute.** No sensor
  subtracts its own `t0`: two sensors that each rebased to their own start would
  disagree about "now" by however far apart they were started.
- **`Measurement.timestamp` is the sample instant, not the arrival instant.**
  Transport latency is removed by the sensor.
- **Device-clock conversion is a `Sensor` responsibility and must not leak
  upward.** The consumer sees host seconds and nothing else.
- **Advance the filter with `advance_to_safe(tick.t_actual)`, not
  `advance_to(now)`.** A sensor stamps the sample instant and hands it over
  later, so a filter advanced to *now* is ahead of whatever arrives next and
  drops it. Measured: at the `SimSensor`'s 5 ms latency the notebook's dry run
  lost 1 of 61; a phase sweep loses 0, 10, 0, 0 of 40; at 20 ms every phase
  loses a third to a half. The loss is intermittent and phase-dependent — the
  loop still runs and the plots are still drawn, the estimate is merely worse —
  and that is exactly what makes it survive a demo.
- **`buffer_horizon` answers one question:** how far behind real time the
  estimate runs. `0.0` is correct for a replayed log and a single stream. Give
  the runner a `Clock` whenever it is non-zero, or the release watermark is the
  newest timestamp *ingested* and each sample waits for its successor — 50 ms of
  lag instead of 5.

## 7. Numerical choices that are load-bearing

- **`core/linalg.make_spd` after every operation on `P`.** Without it NEES comes
  out *negative* and the consistency diagnostic fails silently.
- **Joseph form** for the covariance update. Measured on two states observed
  through their sum with `sigma_R = 1e-9`: the plain `(I-KH)P` gives a minimum
  eigenvalue of −3.6e-17 — not a covariance — and Joseph gives +5.0e-19. Be
  precise about this: it is one update at an extreme `R`, not a filter visibly
  diverging. The consequence that matters is that `nees_of` then returns a
  negative NEES.
- **`np.linalg.solve`, never an explicit inverse.** Innovation covariances become
  badly conditioned as the arm stretches out.
- **Propagate the full covariance block, never `sqrt(diag(P))`.** Over the golden
  run the effector error ellipsoid's major axis sits a median 24.9° (max 45.0°)
  off the nearest world axis, so per-axis sigmas understate the worst direction
  by a median 9.6% and up to 39.7%.
- **One 12-channel update, not four 3-channel ones** — ~360 µs against ~1.4 ms,
  while parsing a line costs 3–10 µs. That is why one serial line becomes
  exactly one `Measurement`.
- **A PSD is not a per-step sigma.** `make_Q` builds a DWNA `Q` from per-step
  sigmas at a fixed 2 ms tick; it is not schedule-invariant. The `psd_*` values
  in ADR-0001 appendix A are continuous-time PSDs. Converting is a
  re-derivation, not a unit change.
- **The EKF is blind to the control by construction, permanently.** `self.u_blind`
  is a zeros vector allocated once; neither `predict` nor `update` takes a `u`.
  The guarantee is that there is nowhere to put a control. A test asserts it by
  signature. Do not add a `u`.

## 8. Test conventions

- **Target the ABC, not the implementation.** A new `Sensor` goes into
  `test_sensor_contract.py`'s `FACTORIES`, not into bespoke tests. `test_robot.py`
  does the same for `ArmInterface`.
- **No test needs hardware. Most must not need mujoco.** Tests needing mujoco
  **and** the Git-LFS assets carry `@pytest.mark.mujoco`; everything else stays
  in `pytest -m "not mujoco"`. Default to the fast side.
- **Pair every diagnostic with a deliberately-broken variant that fails it.** And
  be precise about how it fails — overstating a mild failure is the easiest way
  to get the reasoning quietly disbelieved later.
- **Do not let a test restate a fact that lives in configuration.** Tests that
  hardcoded the IMU wiring passed under *either* wiring and served the rejected
  one as the worked example, which is how a 2–2 disagreement survived four
  copies. Read the fact out of its single source and assert the contract.
- **Log raw device values, not calibrated ones**
  (`MeasurementLog.save_csv(..., transform=decoder.to_raw)`). A raw log can be
  re-decoded after the layout, axis maps or bias are corrected; a log of `z` has
  the configuration of the day it was recorded baked in.

## 9. Language, typing, docstrings

**English:** `core/`, `io/`, `sensors/`, `robot/`, `fusion/`, `calibration/`,
`trajectory.py`, all tests, `docs/adr/`.
**Spanish:** `sim/`, `estimators/`, `models/`, the notebooks, `docs/theory/`.

Roughly hardware-and-plumbing versus physics-and-filtering. New modules on the
command path follow `sensors/` into English. Match the file you are editing; if
you add a package, record which side it is on in CLAUDE.md's directory map.

Type hints on all public signatures — mypy runs `strict`. Docstrings state units
and reference frames for any physical quantity ("linear acceleration, m/s²,
sensor site frame"). This is not boilerplate: silent unit and frame mismatches
are the main correctness risk here. The existing docstrings also record *why* a
choice is load-bearing and what the failure looks like when it is not honoured —
match that, not bare descriptions.

Ruff's rule set is pinned by `[tool.ruff.lint] select`, not left to the default
(which grew from 59 rules in 0.12 to 413 in 0.16). If lint fails, fix the code —
do not edit `select`.

Python 3.10 is the floor (`requires-python`, mypy `python_version`, ruff
`target-version`, and the lower leg of the CI matrix). The conda env is 3.11, so
it will happily run 3.11-only syntax that CI rejects; mypy and ruff catch it
locally. (Note `pyproject.toml`'s dependency comment claims a 3.11 floor for
MjSpec while `requires-python` says 3.10 — an unresolved contradiction, not a
licence to use 3.11 syntax.)

## 10. Things that are decided, not open

- **Non-goals.** No grasp planning, manipulation policy, learning-based control,
  or hard real-time guarantees in the Python layer (<1 kHz timing is firmware's
  job). If a task drifts toward these, flag it instead of implementing it.
- **No event-timeline machinery, no `InputHistory`.** The deleted `FusionEngine`
  design is not the plan. Do not assume it is without asking.
- **Servo `kv` goes in the joint's `damping`, not the actuator.** With the
  default Euler integrator an actuator `kv` integrates explicitly and at a 2 ms
  timestep the sim rings at 250 Hz on startup — the "double bands" in the sensor
  log. Joint damping is integrated implicitly: same force, no ringing.
- **The `home` keyframe uses `act = 1.5693`, not 1.57**, so the tendon equality
  is satisfied exactly and the solver does not shove the arm on the first step.
- **The trajectory's sample spacing is not `dt`** — `int(period_s/dt)` points over
  `[0, period_s]` inclusive are spaced `period_s/(n-1)`: 2.0013 ms against a
  2.0000 ms physics step, so the commanded profile runs 0.067% slow. Carried
  over deliberately; "fixing" it moves the fixture.
- **A rest-window `R` is 3–8× tighter than nominal** and is computed but unused.
  It is the noise floor *at rest* and says nothing about motion or model error,
  so adopting it would sharpen an already over-confident filter. Do not wire it
  up without re-reading `test_calibration.py`.
