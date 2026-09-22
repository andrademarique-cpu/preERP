# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Read this file before `README.md`. The README, `AGENTS.md`,
`docs/adr/0001-multi-rate-fusion.md` and `docs/theory/finger_imu_ekf.md`
describe the **previous** design (a 2-link finger with a hand-written LTI
model and a `FusionEngine`), which was deleted in `1987f00`/`3ada284`.
See "Documentation that no longer matches the code" at the bottom for
what in them is still true.

[`AGENTS.md`](./AGENTS.md) is a shorter restatement of these rules for
non-Claude agents. Nothing keeps the two in sync, so **when you change a
rule here, check whether `AGENTS.md` now contradicts it** — it currently
does, in several places listed below.

## Commands

**The dev env on this machine is called `EKF`, not `erp`.** Verified
2026-09-21: `conda env list` shows only `base`, `EKF` and `MATH`. `EKF`
is Python 3.11.15 with numpy 2.4.6, mujoco 3.11.0, ruff 0.16.4, pytest
9.1.1, matplotlib, and `erp` installed editable against this checkout —
i.e. it is what `environment.yml` describes, just under a different
name. `conda env create -f environment.yml` has never been run here, and
running it would build a *second* environment rather than fix anything;
`conda activate EKF` is the correct move. `base` is Python **3.14.6**
(not 3.13 as this file used to say) and does not have the package.

Note also that `EKF` **does** have `pyserial` 3.5 and `pymycobot` 4.0.7
installed — the `[app]` extra, which the section below says is absent.
See the correction under "`mujoco` is a core dependency".

```bash
# Setup (either path) -- from the REPO ROOT, note the "."
git lfs install
pip install -e ".[dev]"                 # add [app] for real hardware, [viz] for matplotlib
conda env create -f environment.yml     # installs -e .[dev,viz]; already satisfied by `EKF`

# The checks CI runs, in the order ci.yml runs them. Timings on this machine.
ruff check software/src software/tests  # clean; rule set pinned via [tool.ruff.lint] select
mypy software/src                       # strict; clean (37 files)
grep -rEl ...                           # import-direction, the 3rd step -- BEFORE pytest
pytest -q                               # 277 passed, 1 skipped, ~2.6 s -- green

# The fast loop while working. `mujoco` marks the tests that need mujoco AND
# the Git-LFS model assets -- that shared requirement, not what they assert.
pytest -q -m "not mujoco"               # 227 passed, 1 skipped, 50 deselected, ~1.7 s -- green

# The golden check. Not a separate CI step: `test_golden_run.py` imports
# `run_pipeline` from this script, so `pytest` covers it. Run it directly
# anyway -- it prints the per-key diffs and the summary scalars.
python scripts/make_golden_run.py --check   # must print: OK ... a rtol=1e-12

# Single test file / single case
pytest software/tests/test_sensor_contract.py -q
pytest software/tests/test_clock.py::test_micros_wraparound_does_not_jump_back -q

# Import-direction check (what CI greps; run manually before pushing)
grep -rEl "^\s*(from|import)\s+erp\.(sensors)" software/src/erp/estimators software/src/erp/models
# any match is a violation -- the command should print nothing
```

### If the golden check goes red, suspect the log before the code

**It was red from `677540e` until `60e2d07` (2026-09-22), and the code
was never why.** That episode is worth keeping, because the shape of it
will recur: `677540e` committed a 2026-09-21 *re-recording* of
`data/raw/imu_trajectory_raw.csv` (73 measurements / 1872 steps / 3.60 s
against the fixture's 72 / 1848 / 3.55 s) **without** regenerating
`data/processed/golden_ekf_run.npz`. `pytest -q` failed 12 tests, all in
`test_golden_run.py`, `--check` reported shape mismatches plus large
rtol misses on `imu_bias` and the NIS scalars, and CI on `main` was red
on a **clean tree** — so the advice this section used to give, "check
`git status` on `data/raw/`", pointed at nothing.

The diagnosis that worked, and the one to repeat: `run_pipeline(root_dir=...)`
pointed at a shadow root holding the *previously committed* CSV
(recovered from `.git/lfs/objects/12/0a/120a1573…`, or just
`git show <rev>:<path>` plus the LFS object) reproduced the fixture at
worst rtol **0.000e+00**, every key, on unchanged sources. That isolates
data from code in about a minute and is worth doing before reading a
single line of the estimator.

`60e2d07` restored the log. The options, in order of preference:

1. `git checkout <rev> -- data/raw/imu_trajectory_raw.csv` if the new
   recording was incidental — the fixture goes green immediately. Note
   the `<rev>`: once the recording is committed, a bare
   `git checkout --` restores the *failing* file.
2. Keep the new recording *and* regenerate the fixture
   (`python scripts/make_golden_run.py`, no `--check`) — but that is the
   deliberate act described below, and it must be committed together
   with the CSV in one commit that says so. A regenerated fixture
   defends nothing unless the log it was generated from is the log in
   the tree. Costed: it moves `nis_median_moving` 29.10 → 42.4 and every
   figure quoted from the fixture in this file and in ADR-0002 with it.

Do **not** loosen `--rtol` to paper over this; the failure is in the
shapes, which no tolerance reaches.

That re-recording is also why the notebook changed, and **those changes
are committed**: cell 14 has `SEND_TO_ROBOT = True` and cell 15
`RUN_ROBOT_WITH_IMU = True`, with `ROBOT_PORT = "COM6"` and
`IMU_PORT = "COM7"` hard-coded. Running
`notebooks/mypalletizer260EKF.ipynb` top to bottom **opens both serial
ports and drives the arm** — that is the state of `main`, not a dirty
working tree, so it is what a fresh clone gets. ADR-0002 § 3.1 is still
accurate about this.

**Run the golden check before and after any change to the estimation
path.** `data/processed/golden_ekf_run.npz` is a frozen end-to-end run of
the filter over the committed IMU log, and `--check` re-runs the pipeline
and diffs it at `rtol=1e-12`. Eight phases of refactoring have moved code
under it — the trajectory generator, the blind-model build, the rest
state, the whole EKF, at P5 the filter loop and at P6 the rest-window
calibration — without moving one digit, and that is the only reason any
of it can be believed. If it fails after a refactor, the
refactor changed the numbers; do not regenerate the fixture to make it
pass. Regenerating is a deliberate act that needs saying out loud.

Two known soft spots in it: `rtol=1e-12` has only ever been checked on
Windows, and CI runs ubuntu on 3.10 and 3.11, so a cross-platform failure
is plausible and the honest fix is loosening to ~1e-9, not regenerating
per platform. The fixture's own metadata confirms how narrow the evidence
is — `meta_platform` is `Windows-10-10.0.26200-SP0`, `meta_python`
3.11.15, `meta_numpy` 2.4.6, `meta_mujoco` 3.11.0, i.e. exactly the `EKF`
env on this one machine. And the lag it reports is 400 ms against the
~395 ms quoted below — a 2.5 ms search grid makes those adjacent points,
but the two figures may simply come from different recordings. Three
recordings have now reported **400, 405 and 400 ms**, which settles it:
the spread is recording-to-recording, not a bug.

**`mujoco` is a core dependency, not an extra.** ADR-0002 phase P0 moved
it into `dependencies` and dropped `scipy`, which nothing had imported
since `models/discretize.py` was deleted. `erp.sim`, `erp.sensors.mujoco`
and `erp.estimators` import mujoco at module scope, because MuJoCo *is*
the model — so it was never optional, and while it sat in `[app]` CI
installed `.[dev]` and type-checked a tree missing its central
dependency. The `[app]` extra is now device access only: `pyserial` for
the Teensy, `pymycobot` for the arm, both imported lazily.

**Correction (verified 2026-09-21): both ARE installed in `EKF`** —
`pyserial` 3.5 and `pymycobot` 4.0.7, presumably because the notebook
drives real hardware from this env. So "importing `erp.robot` must not
need `pymycobot`" is *not* being tested locally; it is a formality here
and a real test only in CI, which installs `.[dev]` alone. If you touch
the lazy-import boundary in `robot/mypalletizer.py` or
`sensors/imu_serial.py`, a green local `pytest` proves nothing about it —
CI is the check that matters.

Since ADR-0002 phase P4 the estimator no longer imports mujoco at all:
`estimators/ekf.py` holds a `DiscreteDynamics` (a numpy-only protocol in
`models/base.py`), and `sim/dynamics.MujocoDynamics` is what satisfies it
with physics. `erp.sim` and `erp.sensors.mujoco` still import mujoco at
module scope. The tests split accordingly — most run without mujoco, and
the ones that need it carry the `mujoco` marker.

**Always install from the repo root, with the `.`.** There is no
`pyproject.toml` or `setup.py` under `software/`; the single packaging
file is at the root and already points at the nested sources
(`packages.find` → `software/src`, `testpaths` → `software/tests`). The
`pip install -e "./software[dev]"` form is not a typo but an
impossibility.

**`pyyaml` is also core, since P7**, on the same test: `erp.io.config`
imports it at module scope to turn `config/estimation.yaml` into the
typed dataclasses that are now the single definition of the IMU wiring,
and a config file is not device access. It is pure Python and installs
everywhere, so none of the reasons that keep `pyserial` and `pymycobot`
in `[app]` apply to it.

**Dependencies are declared once, in `pyproject.toml`.**
`environment.yml` supplies only the interpreter and pip, then installs
`-e .[dev,viz]`. Do not re-add package lists to `environment.yml`: the
two lists drifted last time and disagreed about what an environment
should contain.

**The ruff rule set is pinned by `[tool.ruff.lint] select`, not left to
the default** (`E,F,I,UP,B,RUF`). ruff's *default* set went from 59
rules in 0.12 to 413 in 0.16, so with no `select` the answer to "does
this tree lint clean?" depended on which ruff the developer happened to
have. If lint fails, fix the code — do not edit `select` to make it pass.

Python 3.10 is the floor (`requires-python`, `[tool.mypy]
python_version`, `[tool.ruff] target-version`, and the lower leg of the
CI matrix). The conda env is 3.11, so it will happily run 3.11-only
syntax that CI rejects; mypy and ruff catch it locally.

## What this project is

`erp` supports state estimation on an **Elephant Robotics MyPalletizer260**
4-DOF arm: an EKF over MuJoCo dynamics, fed by two MPU-style IMUs
streamed from a Teensy 3.6 over USB serial. It is one of three coupled
tracks (software / mechanical / electronics & firmware); this repo's
code work is almost entirely in `software/` and `notebooks/`.

The earlier target — a 2-link robotic finger with contact-force sensing —
is where the design precedent comes from, but none of its code survives
in the package. The arm is real hardware that exists; the finger was
simulation plus a plan.

**`notebooks/mypalletizer260EKF.ipynb` is the application.** The `erp`
package is the library it pulls from, and the package grew by extracting
pieces of it — ADR-0002 is the plan for that, and phases P0 through P4
are done. What the notebook still owns: the `run_trajectory` loop
(P5's), the EKF driver `run_imu_ekf` (P5's, and currently living in
`scripts/make_golden_run.py`), the configuration values, the plots, and
the narrative. What it no longer owns, despite older text saying so: the
trajectory generator, the blind-model compilation, the arm classes, the
range guard and the root finder.

When deciding where new code goes: if it would work unchanged against a
different arm or a recorded log, it belongs in `erp`; if it knows about
*this* trajectory, *this* API's degree conventions or *this* plot, it
stays in the notebook. Measured constants are the exception — they move
with the code that they annotate, or into ADR-0002. They are not deleted.

**Non-goals — do not add these:** grasp planning, manipulation policy,
learning-based control, or hard real-time guarantees in the Python layer
(<1 kHz timing is firmware's job). If a task drifts toward these, flag
it instead of implementing it.

## The one rule that overrides everything else

> Hardware access enters the stack through `Sensor`, and nowhere else.

`software/src/erp/core/`, `models/`, `estimators/` and `sim/` must
**never** import from `software/src/erp/sensors/`, `firmware/`, or any
ROS 2 package (`ros2_ws/`). CI greps for it, and it is enforced in
review regardless of whether the code works.

Why this matters concretely: every estimator must be runnable in a
notebook, in CI, and against recorded data with zero hardware attached.
The moment `estimators/` reaches into `sensors/`, that stops being true.

If you need a new measurement backend, add it as a `Sensor` subclass —
`SerialIMUSensor` for a live device, `ReplaySensor` for a log,
`SimSensor` for recorded MuJoCo output, or a new `StreamSensor`
subclass for another live link. Never route device access through the
core.

**MuJoCo is the deliberate exception, and it is a real change of
position.** MuJoCo is treated as *the model*, not as hardware: it is
deterministic, installs everywhere, and needs no device attached, so the
property the import rule protects still holds. ADR-0001 § D5 ("MuJoCo
stays behind `SimulatedSensor`, never as a `ProcessModel`") was overruled
by the rewrite and ADR-0001 was not updated. Do not cite D5 as current
policy, and do not use this exception to justify another one.

Since P4 the exception is **narrower than it was**, and worth knowing
precisely: `estimators/ekf.py` no longer imports mujoco. It holds a
`DiscreteDynamics` — a numpy-only protocol — so the filter runs against
`LinearDynamics` with mujoco absent from `sys.modules` entirely, which is
how it gets compared to a closed-form Kalman filter. `models/` and
`core/` are mujoco-free and must stay that way. Exactly three modules
touch the mujoco API today:

```
software/src/erp/sim/mujoco.py       f/F/h/H, the paired forms, make_Q
software/src/erp/sim/plant.py        load_model, blind_variant, sensor_layout
software/src/erp/sensors/mujoco.py   rows_of, make_R
```

**`erp.sim.mujoco` is the only module allowed to call `mj.*` for the
dynamics**, and every function there wraps its result in
`np.asarray(..., dtype=np.float64)` or `int(...)` before returning, so no
`Any` escapes the untyped boundary. When adding physics, add it there and
wrap it in a class elsewhere — that is why `MujocoDynamics` lives in
`sim/dynamics.py` but calls into `sim/mujoco.py` rather than mujoco.

**Do not treat a green pipeline as proof you honored the rule.** The
check is a single grep for `^\s*(from|import)\s+erp\.(sensors)` over
`estimators/` and `models/` only, so it does not catch:

- relative imports — `from ..sensors import SerialIMUSensor` passes CI;
- `erp.sim/`, `erp.core/`, `erp.io/` — not scanned at all;
- `firmware/` or ROS 2 imports (`import rclpy`), despite the rule text;
- indirect reach-through, e.g. importing a `calibration/` module that
  itself imports `sensors/`. **That example stopped being hypothetical at
  P6**: `erp/calibration/` exists, and it imports `erp.core.types` and
  numpy and nothing else in `erp`. The dependency runs
  `sensors/` → `calibration/`, one way. Keep it that way — reversing it
  would put a hardware import one hop from anything that calibrates.

The rule is broader than its enforcement. Verify by reading the import
block. The tree is currently clean under the broader reading.

## Architecture: MuJoCo is the model, and the measurement is a row index

The single idea that explains the package: **nobody writes down `A`,
`B`, `h` or a Jacobian.** MuJoCo holds the dynamics and the sensor
definitions, and the filter asks it. Six pieces implement that, and
they only make sense together.

1. **`sim/mujoco.py` — `f_dyn` / `F_dyn` / `h_dyn` / `H_dyn`.** Each one
   loads the filter state into an `MjData` (`mj_resetData` first, so no
   residual state leaks between calls), runs `mj_step` or `mj_forward`,
   and reads back. `F` and `H` are the `A` and `C` blocks of
   `mjd_transitionFD`. The state vector is `[qpos[:nv], qvel[:nv],
   act[:na]]` — note `nv`, not `nq`: positions live in the **tangent**
   space, which is what makes the finite-difference Jacobian valid.
   `state_dim` is the only correct source for `nx`.

   Since P4 these have **paired forms**, `step_with_jacobian` and
   `observe_with_jacobian`, which return `(x_next, F)` and `(z, H)` from
   one `MjData` load. They are what `MujocoDynamics` calls, and they are
   bit-identical to calling the four separately. **`observe_with_jacobian`
   runs a second `mj_forward` and it is not redundant**:
   `mjd_transitionFD` restores `qpos`/`qvel`/`act` but leaves
   `sensordata` at its last finite-difference perturbation, so reading it
   directly gives a `z` wrong by 4.3e-4 — about 1% of one `sig_acc`,
   small enough to look like a tolerance problem. Do not "simplify" it
   away.

2. **`models/base.py` — `DiscreteDynamics`.** A protocol with `nx`,
   `nz`, `nu`, `dt`, `step(x, u) -> (x_next, F)` and
   `observe(x, u) -> (z, H)`. Pure numpy; this is the seam that keeps an
   `MjModel` out of the filter. Two implementations:
   `sim/dynamics.MujocoDynamics` (physics) and
   `models/linear.LinearDynamics` (constant Jacobians, no mujoco).
   Returning pairs rather than four separate methods guarantees the value
   and its Jacobian come from the *same* linearisation point. It also
   halves the `MjData` loads, but that is worth about 8%, not the 50% the
   count suggests — `mjd_transitionFD` runs `nx+1` internal evaluations
   and is the actual cost.

3. **`estimators/ekf.py` — `EKF`.** Joseph-form update, `np.linalg.solve`
   throughout, `make_spd` after every touch of `P`. It holds a
   `DiscreteDynamics`, **not** an `MjModel`. It is **blind to the control
   by construction, permanently — a project decision, not a transitional
   state**: `self.u_blind` is a zeros vector allocated once, and neither
   `predict` nor `update` takes a `u` parameter. The guarantee is that
   there is nowhere to put a control, not that callers refrain from
   passing one, and a test asserts it by signature. Do not add a `u`.
   Whether blindness is *sound* depends entirely on the model handed to
   it — see the next point.

   `update(z, rows, R=None)` accepts a per-measurement `R` and otherwise
   falls back to `self.R[ix_(rows, rows)]`. Both give the same answer
   today because the sensor's `R` is built as that same block; they
   diverge once `calibrate()` runs on the arm.

4. **The blind model.** `erp.sim.plant.blind_variant` loads the arm XML
   through `MjSpec` and sets every actuator's
   `dyntype` to `mjDYN_INTEGRATOR`, which turns the three position
   servos into activation states (`na` 0 → 3, `nx` 8 → 11). Those
   activations are what the filter estimates in place of the command it
   never receives. Compiling through `MjSpec` rather than editing XML
   text is not cosmetic: the XML has an `<include>` and relative mesh
   paths that `from_xml_string` cannot resolve. The un-edited plant model
   is the one used to *generate* truth, never the one the filter runs on;
   `test_plant.py` asserts the plant *fails* the `na == 3` contract.

5. **`core/types.py` — `Measurement`.** A reading is `(z, timestamp,
   rows, R, source)`, where `rows` are **indices into
   `data.sensordata`**. That is the whole coupling between a physical
   device and the filter: the EKF does `z[rows] - observe(x)[0][rows]`
   and never learns which chip produced it. `rows` and `R` are one shared,
   **read-only** array per sensor, not copied per sample — see
   `shared_rows_R`; a caller mutating `m.R` in place would otherwise
   silently rewrite the noise of every past and future sample, and a
   test asserts the write raises.

6. **`sensors/clock.py` — `ClockSync`.** Device time → host time by a
   sliding-window minimum of `arrival - device_time`, because latency is
   never negative, so the least-delayed sample in the window is the best
   offset estimate. Device ticks are unwrapped first: 32-bit `micros()`
   wraps every ~71.6 min, and a wrap read as a 71-minute jump backwards
   stamps every later sample in the past. `test_clock.py` pairs the
   passing test with a falsification — arrival stamping, even with the
   mean latency subtracted, misses the same 1 ms bound.

Two rules follow, and any change touching sensors or the loop must
preserve both:

- **One monotonic host time base, `time.perf_counter()`, absolute.** No
  sensor subtracts its own `t0`: two sensors that each rebased to their
  own start would disagree about "now" by however far apart they were
  started. `Measurement.timestamp` is the **sample** instant, not the
  arrival instant — transport latency is removed by the sensor.
- **Device-clock conversion is a `Sensor` responsibility and must not
  leak upward.** The consumer sees host seconds and nothing else.

### How the loop actually runs

`run_trajectory` (notebook) ticks at the setpoint rate over a
`core.clock.RateLoop`, and after each `send` calls `drain()` on every
sensor and appends to a `MeasurementLog`. It never blocks on a port —
each live sensor has its own reader thread (`StreamSensor`), so the
setpoint rate is independent of the IMU rate.

**Since P5 that `drain` hook is connected.** Pass `runner=` a
`FilterRunner` and the filter runs *inside* the loop; leave it out and
the loop behaves exactly as before. The same object replays a log
offline via `run()`, which is how the golden run is produced — one
implementation, two entry points, so M1 and M2 cannot drift apart in
their arithmetic. `run_imu_ekf` is gone from both the notebook and
`scripts/make_golden_run.py`.

Three things about that loop that are easy to get wrong:

- **Advance with `advance_to_safe(tick.t_actual)`, not
  `advance_to(now)`.** A sensor stamps the *sample* instant and hands it
  over later, so a filter advanced to now is ahead of whatever arrives
  next and drops it. The loss is intermittent and phase-dependent, not
  total — the dry run loses 1 of 61 at a 5 ms latency, and a phase sweep
  finds cases losing 10 of 40 — which is exactly what makes it survive a
  demo.
- **`buffer_horizon` answers one question, not two:** how far behind real
  time the estimate runs. It governs both when a held measurement is
  released and how far the filter dares step. `0.0` is correct for a
  replayed log and for a single sensor; it earns its keep when a second
  stream with a different latency appears.
- **Give the runner a `Clock` whenever `buffer_horizon > 0`.** Without
  one the release watermark is the newest timestamp *ingested*, so each
  sample waits for its successor — 50 ms of lag instead of 5.

There is still no event-timeline machinery and no `InputHistory`. Do not
assume the deleted `FusionEngine` design is the plan without asking.

### Numerical choices that are load-bearing

- **`core/linalg.make_spd` after every operation on `P`.** Symmetrise
  and floor the eigenvalues. Without it NEES comes out *negative* and
  the consistency diagnostic fails silently.
- **Joseph form** for the covariance update, which survives round-off
  that plain `(I - KH) P` does not.
- **`np.linalg.solve`, never an explicit inverse.** Innovation
  covariances become badly conditioned as the arm stretches out.
- **Propagate the full covariance block, never `sqrt(diag(P))`.** The
  end-effector depends on all three joints at once; dropping the
  cross-terms barely moves the error-ellipse *area* while rotating it
  and inflating the worst-direction sigma. See `site_position_cov` in
  the palletizer notebook.
- **One 12-channel update, not four 3-channel ones.** Measured on this
  model: ~360 µs against ~1.4 ms, while parsing a line costs 3–10 µs.
  That is why one serial line becomes exactly one `Measurement`.

## Hardware facts encoded in the tree

These were measured, not assumed, and re-deriving them is expensive:

- **Wiring and axes.** IMU_0 → `link1`, IMU_1 → `link2`, each rotated
  Rz(∓90°) from chip axes to MuJoCo site axes. Fitted against a MuJoCo
  replay of the same trajectory over all 48 signed permutations and both
  chip assignments, reproduced on two recordings. The layout lives in
  `IMUDecoder`'s `layout`/`axis_maps` and in `config/estimation.yaml`,
  not in code. **Note the two disagree today**: `conftest.py` and the
  config assign the chips to opposite links. The config carries the
  fitted residuals (gyro rms 0.06–0.10 rad/s vs 0.34 for the swapped
  wiring), so trust the config and treat the test fixture as arbitrary
  test data.
- **The real arm lags the MuJoCo replay by ~395 ms** (arm transport and
  queueing, not the IMUs). Log for a tail after the last setpoint or the
  motion gets cut.
- **Servo `kv` goes in the joint's `damping`, not the actuator.** With
  the default Euler integrator an actuator `kv` integrates explicitly
  and at a 2 ms timestep the sim rings at 250 Hz (Nyquist) on startup —
  the "double bands" in the sensor log. Joint damping is integrated
  implicitly: same `-kv*qdot` force, no ringing, no integrator change.
- **The `home` keyframe uses `act = 1.5693`, not 1.57**, so the tendon
  equality is satisfied exactly and the solver does not shove the arm on
  the first step.
- **Initialise from the servoed equilibrium, not zeros.**
  `erp.sim.plant.warmup_to_rest` warms up 100 steps under the command.
  Measured `link2_acc_x` at rest: **9.810** from the warmed equilibrium,
  **4.144** from the `home` keyframe with `act = 0` and no warmup,
  against ~9.65 on the real IMU. That 5.7 m/s² offset on a channel whose
  sigma is 0.05 is a **114-sigma** bias — no `Q` or `R` tuning repairs
  it. Careful with the counterfactual: zeroing the *whole* state instead
  gives 345 m/s², because that also breaks the tendon equality. It is a
  different and much louder failure, and quoting it would overstate this
  one.

### Numbers measured during the migration

These were produced by the phase work and are cheaper to read than to
re-derive. Each has a test that would fail if it stopped being true.

- **Absolute deadlines vs cumulative sleeps**, 76 ticks at 25 Hz on this
  machine: a cumulative `sleep(1/rate)` loop finishes **+65.4 ms** late
  and its error grows monotonically; absolute deadlines finish **+0.0
  ms** with a worst per-tick error of 0.7 ms. `sleep(2 ms)` measures 2.49
  ms here (viewer.ipynb recorded 2.86 ms under different load). This is
  why `RateLoop` exists and why it never sleeps a period.
- **Joseph vs `(I-KH)P`**, on two states observed through their sum with
  a posterior spanning 8 orders of magnitude and `sigma_R = 1e-9`: the
  plain form gives a minimum eigenvalue of **−3.6e-17** — not a
  covariance — and Joseph gives **+5.0e-19**. Precisely: that is one
  update at an extreme `R`, *not* a filter visibly diverging. The
  consequence that matters is that `nees_of` on such a `P` returns a
  negative NEES, so the diagnostic reports something impossible rather
  than raising.
- **The effector error ellipsoid is not axis-aligned.** Over the golden
  run its major axis sits a median **24.9°** (max 45.0°) off the nearest
  world axis, so per-axis sigmas understate the worst direction by a
  median 9.6% and up to **39.7%**. The fixture currently stores only
  `sqrt(diag(C))`; the full 3×3 `C_ef` is computed and discarded. Keep
  it when `propagate_to_site` lands at P8.
- **Pairing the MuJoCo calls saves ~8%, not 50%.** 143.5 µs → 131.5 µs
  per predict+update. Loads halve, but `mjd_transitionFD` runs `nx+1`
  internal evaluations and is the real cost.
- **The trajectory's sample spacing is not `dt`.** `int(period_s/dt)`
  points over `[0, period_s]` inclusive are spaced `period_s/(n-1)`:
  2.0013 ms against a 2.0000 ms physics step, so the commanded profile
  runs 0.067% slow. Carried over deliberately — the golden run was
  generated with it, so "fixing" it moves the fixture.
- **Advancing the filter to `now` loses samples, intermittently.** At the
  `SimSensor`'s 5 ms latency the notebook's own dry run lost **1 of 61**;
  a sweep over four phases loses **0, 10, 0, 0 of 40**, and at 20 ms
  every phase loses between a third and a half. The loss is
  phase-dependent because 25 Hz commands and 20 Hz samples are not
  integer multiples, so the phase drifts through a run. It is not a
  filter that stops working — the loop runs and the plots are drawn, the
  estimate is merely worse — and *that* is what makes it survive a demo.
  `advance_to_safe` loses none, at every phase, with and without a clock.
- **Those drop counts are ULP-fragile above 5 ms, so only the 5 ms row is
  pinned.** Whether a sample is lost turns on whether a command tick
  falls inside the latency window after its timestamp, so a sample within
  one ULP of that boundary flips with the arithmetic used to *build the
  fixture*: writing the sample instants `i / 20.0` instead of `i * 0.05`
  moves the 20 ms counts by three. `test_fusion_runner.py` asserts the
  monotonicity (more latency never loses fewer) instead, because that is
  a property of the runner rather than of the test.
- **The rest calibration, packaged (P6).** The accelerometer channels of
  the cell's formula and `calibration_from_samples` were **already
  identical** (`0.000e+00`); the gyro channels differ by exactly
  `h_rest[gyro]`, max **2.352e-4 rad/s** — 4.7% of `sig_gyro`. That is
  residual velocity `warmup_to_rest` leaves, i.e. a *model* artifact
  folded into a *sensor* bias. Expecting 0 is the better physics and was
  **not** adopted, because it moves the frozen run; zero the gyro
  entries of `expected_rest` whenever someone decides to.
- **A rest-window `R` is 3–8× tighter than nominal** (σ ratios 0.13–0.39
  on the committed log). It is the noise floor *at rest* and says
  nothing about motion or model error, so adopting it would sharpen an
  already over-confident filter. Computed by `rest_bias`, used nowhere.
- **The offline rest window holds 8 samples** (0.4 s at ~20 Hz) against
  a live `min_samples` default of 20 — and a rejection returns
  `valid=False` *together with a zero bias*, which looks exactly like
  success to any caller that skips the flag. `rest_bias` defaults to 3.
- **The online dry run** (P5's new notebook cell, `DryRunArm` +
  `SimSensor`, no hardware): 1500 predicts and 61 updates over 3 s, NIS
  median **9.6** against a target of 12, nothing discarded. Unlike every
  other number here this one has **no test pinning it** — it lives in a
  notebook cell — and it is a plumbing check, not evidence about the
  estimator: the simulated IMU draws noise from the same `R` the filter
  is given and the plant *is* the model. The figure that matters is the
  one on real data, NIS median 29.

## Directory map / where things go

| Path | Contents | Import boundary |
|---|---|---|
| `software/src/erp/core/` | `types` (`Measurement`, `CalibrationResult`, `Array`), `linalg` (`make_spd`, `nees_of`), `clock` (`Clock`, `WallClock`, `VirtualClock`, `RateLoop`) | numpy only; no hardware, **no mujoco** |
| `software/src/erp/models/` | `base` (`DiscreteDynamics` protocol), `linear` (`LinearDynamics`, `affine_predict`), `finger_config` (legacy finger dataclasses, dead) | numpy only; **no mujoco** — that is what makes the no-mujoco EKF test possible |
| `software/src/erp/sim/` | `mujoco` (`f_dyn`/`F_dyn`/`h_dyn`/`H_dyn`, the paired forms, `linearize`, `make_Q`, `state_dim`), `plant` (`load_model`, `blind_variant`, `warmup_to_rest`, `sensor_layout`), `dynamics` (`MujocoDynamics`) | mujoco yes, `erp.sensors` never |
| `software/src/erp/estimators/` | `ekf` — `EKF` over a `DiscreteDynamics`; no `MjModel`, no mujoco import | builds on `models/` + `core/` |
| `software/src/erp/robot/` | `base` (`ArmInterface`, `JointMap`), `dry_run` (`DryRunArm`), `mypalletizer` (`MyPalletizerArm`) — **command sink, never a measurement source** | may depend on `core/`; nothing in the estimation stack may import it |
| `software/src/erp/trajectory.py` | `sine_sweep`, `resample` — the setpoint profile | numpy only |
| `software/src/erp/sensors/` | `base` (`Sensor` ABC), `stream` (threaded base), `imu_serial` (`IMUDecoder`, `SerialIMUSensor`), `replay`, `sim`, `clock` (`ClockSync` — device→host, *not* `core.clock`), `mujoco` (`rows_of`, `make_R`). `calibration_from_samples` is **re-exported, not defined here** — it moved to `calibration/` at P6 | may depend on `core/` and `calibration/` |
| `software/src/erp/io/` | `log` (`MeasurementLog`), `paths` (`repo_root`, `resolve_repo_path`, `resolve_model_path`), `config` (P7 — `load_config`, `build_decoder`, the frozen config dataclasses, `ConfigError`). **English** | `core/` only for `log`/`paths`. `config` imports `sensors.IMUDecoder`, so it is deliberately **not** re-exported from `io/__init__.py` — `import erp.io` must stay core-only, because `fusion/` may import `erp.io` and may never reach `sensors/`. A subprocess test pins it |
| `software/src/erp/fusion/` | `runner` — `FilterRunner` (timestamps → filter steps), `History`, `Estimator` protocol. **English**, beside `core/`/`io/`, not beside `estimators/`: it decides *when* the filter steps, never what it computes. Landed at P5; the old `FusionEngine` it replaces was deleted and shares no design with it | builds on `core/` + `models/`; **never** `sensors/`, `robot/` or a wall clock |
| `software/src/erp/calibration/` | `rest` — `calibration_from_samples` (moved here from `sensors/imu_serial.py` at P6, still re-exported there), `rest_bias` (the offline counterpart of `SerialIMUSensor.calibrate`). **English** | `core/` + numpy **only**; `sensors/` imports this, never the reverse |
| `software/src/erp/viz/` | **empty** (P8). `__init__.py` carries a one-line docstring ("Plotting and visualisation helpers. Geometry only, no backend imports.") and nothing else — honor that "no backend imports" when P8 lands | — |
| `software/tests/` | 278 tests. `conftest` (fake serial port + IMU layout), `test_sensor_contract` (one suite over all three `Sensor`s), `test_imu_serial`, `test_clock`, `test_replay_log`, `test_paths`, `test_plant`, `test_golden_run`, `test_robot`, `test_trajectory`, `test_core_clock`, `test_dynamics`, `test_ekf_linear`, `test_fusion_runner` (47, no mujoco — carries a verbatim copy of the old `run_imu_ekf` as a bit-identity oracle), `test_calibration` (11, 9 of them fast — same oracle trick against notebook cell 19's block), `test_config` (24, 21 of them fast — every malformed-config case paired with the good file it mutates) | — |
| `scripts/` | `make_golden_run.py` — generates and re-checks the golden fixture. A **holding pen**: `run_imu_ekf` left at P5, so what remains is `estimate_lag` and `site_position_cov`, both P8's. When they go, M1 closes | not linted by CI |
| `data/processed/` | `golden_ekf_run.npz` — the frozen run. Git-LFS | — |
| `notebooks/` | `mypalletizer260EKF.ipynb` — **the driver application**; `viewer.ipynb` (MuJoCo viewer + `SimLog`); `finger_imu_toolkit.ipynb` (finger EKF rebuilt on `erp`); `finger_imu_practice.ipynb`, `Palletizer.ipynb` (reference) | may import anything |
| `mechanical/mujoco_assets/` | `MyPalletizer260/MyPalletizer260.xml` + STL meshes, `axis_xyz.xml` — the model everything runs on | — |
| `config/` | `estimation.yaml` — the fitted IMU layout, axis maps, rates. **Read by `erp.io.config` since P7**, and now the definition rather than documentation; the notebook still mirrors the values by hand. The finger sections moved to ADR-0001 appendix A | — |
| `data/raw/` | three short IMU logs (raw device columns, calibrated columns, and a layout-named variant) + `palletizer_traj.npz` | — |
| `ros2_ws/src/erp_ros/`, `docs/hardware/`, `electronics/` | **empty** (`.gitkeep` only) | — |
| `firmware/` | `potentiometer_logger/potentiometer_logger.ino` only — the Teensy IMU firmware is not in this repo | out of scope for Python changes |
| `docs/theory/`, `docs/adr/` | derivations and decision records — both describe the finger stack | — |

`software/src/erp/__init__.py` is empty by design — import from the
subpackages. The `__all__` lists in `core`, `models`, `estimators`, `io`,
`sensors` and `robot` are the accurate inventory of what is public.
(`erp/models/__init__.py` used to re-export nothing, with its import
block commented out "pending migration"; P4 gave it real contents. Let
that be the warning against adding a type nothing reads yet.)

`io/paths` was reconciled in P1. `repo_root()` searches upward for
`pyproject.toml` **from the module**, not from `Path.cwd()` and not by
looking for a directory *named* `preERP` (which broke when the clone was
renamed), so the answer no longer depends on where the kernel started.
`resolve_repo_path` reaches anywhere in the repo; `resolve_model_path`
still means `notebooks/<subfolder>/` on purpose, for its one caller.
There were three copies of the root finder in three notebooks; there is
one now. `get_project_root` is kept as an alias.

## Conventions to follow when writing code here

- **Type hints on all public signatures.** mypy runs `strict`.
- **Docstrings state units and reference frames** for any physical
  quantity ("linear acceleration, m/s², sensor site frame"). This is not
  optional boilerplate — silent unit/frame mismatches are the main
  correctness risk here. The existing docstrings in `core/`, `io/` and
  `sensors/` also record *why* a choice is load-bearing and what the
  failure looks like when it is not honored; match that, not bare
  descriptions.
- **Language follows the module, not your preference.** English:
  `core/`, `io/`, `sensors/`, `robot/`, `fusion/`, `calibration/`,
  `trajectory.py`, all tests, and `docs/adr/`. Spanish: `sim/`,
  `estimators/`, `models/`, the notebooks
  and `docs/theory/`. The split is roughly hardware-and-plumbing versus
  physics-and-filtering; new modules on the command path (`robot/`,
  `trajectory.py`) follow `sensors/` into English. Match the file you are
  editing, and if you add a package, say in this list which side it is on.
- **Tests target the ABC, not the implementation.**
  `test_sensor_contract.py` runs one parametrised suite against
  `ReplaySensor`, `SimSensor` and `SerialIMUSensor` (over a fake port) —
  a new `Sensor` should be added to `FACTORIES` there, not given its own
  bespoke tests. `test_robot.py` does the same for `ArmInterface` over
  `DryRunArm` and `MyPalletizerArm`.
- **Tests must not need hardware. Most must not need mujoco.** This
  replaces an earlier flat "no mujoco" rule, which P0.5 broke on purpose:
  M1's acceptance test *is* the model, so it cannot be written without
  it. Tests needing mujoco **and** the Git-LFS assets carry
  `@pytest.mark.mujoco`; everything else stays in `pytest -m "not
  mujoco"`, which runs in ~0.9 s. Default to the fast side — 150 of 196
  tests are there, including the whole EKF-versus-Kalman comparison.
- **Inject the clock; do not read one.** Anything that waits or
  timestamps takes a `Clock` (`core/clock.py`) or a
  `Callable[[], float]`, so it can be driven by `VirtualClock` in a test.
  A 3 s trajectory then runs in under 50 ms and scheduling becomes
  deterministic. Note `core.clock` (what time is it, when should the next
  thing happen) is a different job from `sensors.clock` (when was this
  sample taken) — do not use one where the other is meant.
- **Pair a diagnostic test with a deliberately-broken variant that
  fails it.** `test_clock.py` is the model: the sync passes a 1 ms
  bound and arrival stamping is asserted to miss it. A test that cannot
  fail is not evidence of anything. The same applies to NEES/NIS work —
  filter correctness is judged by those, never by eyeballing a
  trajectory.
- **Be precise about how a broken variant fails.** Overstating a mild
  failure is the easiest way to get the reasoning quietly disbelieved
  later.
- **A PSD is not a per-step sigma.** `make_Q` in `sim/mujoco.py` builds
  a **DWNA** `Q` from per-step sigmas at a fixed 2 ms tick — it is not
  schedule-invariant, so it is only valid at the model's timestep. The
  `psd_*` values in `config/estimation.yaml` are continuous-time PSDs in
  rad²/s³ and rad²/s from the old design; converting between the two is
  a re-derivation, not a unit change.
- **Log raw device values, not calibrated ones**, via
  `MeasurementLog.save_csv(..., transform=decoder.to_raw)`. A raw log can
  be re-decoded after the layout, axis maps or bias are corrected; a log
  of `z` has the configuration of the day it was recorded baked in.
  `test_replay_log.py` demonstrates exactly this repair.
- Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`.
  Recent history does not follow this ("Refactor", "NA", "working on
  it") — follow the convention anyway.
- Respect module ownership when suggesting reviewers — see `CODEOWNERS`
  (handles are still placeholders; its `fusion/` entry points at a
  package that is still empty, and it has no entry for `robot/`,
  `trajectory.py` or `scripts/`).

## Current repo state

`git log --oneline -5` and `git status` are the authority; this section
goes stale on its own. **The ADR-0002 work is now committed** — as of
2026-09-22 the tip is P7 (`erp.io.config`), on top of `60e2d07` (the
golden-log restore) and `677540e` (`added calibration package`). Neither
`587162f` nor `677540e` follows the conventional-commit rule this file
states; follow the convention anyway.

The 2026-09-21 hardware run (`data/raw/imu_trajectory_raw.csv`,
`data/raw/palletizer_traj.npz`, `notebooks/mypalletizer260EKF.ipynb`)
was committed in `677540e`, and the CSV part of it was reverted in
`60e2d07` because it had been committed without regenerating the golden
fixture — see the top of this file. The notebook and `palletizer_traj.npz`
were left as they are.

Working: the `Sensor` stack (`SerialIMUSensor` + `IMUDecoder` over a
Teensy, `ReplaySensor` with CSV loaders and paced release, `SimSensor`
over recorded MuJoCo output, `ClockSync`), `MeasurementLog`, the MuJoCo
`f/F/h/H` helpers and their paired forms, `erp.sim.plant`,
`erp.robot`, `erp.trajectory`, `erp.core.clock`, the
`DiscreteDynamics` seam, the blind EKF, `erp.fusion.FilterRunner`,
`erp.calibration`, `erp.io.config`, and the palletizer notebook end
to end (trajectory → real arm + MuJoCo replay → IMU log → EKF →
end-effector covariance). 278 tests collected; **277 pass, 1 skipped**,
`ruff` and `mypy --strict` (37 files) are clean, `pytest -m "not
mujoco"` is green (227 passed, 1 skipped, 50 deselected), and
`make_golden_run.py --check` reproduces the fixture at `rtol=1e-12`.

**ADR-0002 phases P0, P0.5, P1, P2, P3, P3.5, P4, P5, P6 and P7 are done.** Read
`docs/adr/0002-notebook-to-package.md` before starting any of the rest:
each finished phase carries a *What landed* record, including where the
plan turned out to be wrong and was changed on purpose (P3 was reshaped
before it was built; P4 corrected a performance claim in § 4.4).

Two things to preserve from the finished phases:

- **`erp.sim.mujoco` is the only module that may call the mujoco API for
  the dynamics, and every function there wraps its result in
  `np.asarray(..., dtype=np.float64)` or `int(...)` before returning**,
  so no `Any` leaks into the estimator under strict mypy.
- **Every phase so far has left `make_golden_run.py --check` green at
  `rtol=1e-12`**, including P4, which rewrote the EKF. That is the
  standard: a refactor that moves the numbers is not a refactor. Still
  true as of 2026-09-21 — re-verified by running `run_pipeline` against
  the committed log on the current sources.

**M1 is still not achieved, and P5 did not close it** — these are
different claims and it is easy to bank the wrong one. M1 needs *the
package*, not a script, to reproduce the golden run, plus a
`ConsistencyReport`. P5 took `run_imu_ekf` into `erp.fusion`, so what is
left in `scripts/make_golden_run.py` is `estimate_lag` and
`site_position_cov` — both P8's. **M1 now closes at P8 alone.**

Not built: `viz/`, `analysis/`, `runtime/`, `ros2_ws/`, and any UKF.
Online fusion exists
(`erp.fusion.FilterRunner`) but has only been run against `DryRunArm` +
`SimSensor`, never against the Teensy.

Known open items:

- ~~`run_trajectory` still has its own absolute-deadline loop.~~
  **Closed at P5.** It runs on `RateLoop` now and the achieved-rate and
  jitter arithmetic went with it. The policy question P3.5 left open was
  decided in favour of `RateLoop`'s nominal period (`t0 + i / rate_hz`)
  over `resample`'s endpoint-preserving axis: ADR-0002 § 4.7 makes the
  command period a *scheduling parameter*, and pacing off the
  trajectory's axis couples the scheduler to the trajectory. `resample`
  still supplies the setpoint values, endpoint included, so only the
  *timing* changes, and only when `span * rate_hz` is fractional — at the
  notebook's 3 s and 25 Hz the two are bit-identical.
- ~~**A fourth copy of the IMU wiring** lives in `sensors/imu_serial.py`.~~
  **Closed at P7.** The split was 2–2 and is now 4–0 on **IMU_0 → link1,
  IMU_1 → link2**, the assignment the fit chose and the one the golden
  fixture was generated from. `config/estimation.yaml` is the definition,
  loaded by `erp.io.config`; `conftest.py` and the `IMUDecoder` docstring
  were corrected to match it, and `test_config.py` asserts that
  `scripts/make_golden_run.py`'s dict still agrees, so editing the YAML
  alone now fails. The swapped form survives in exactly one place,
  `test_golden_run.SWAPPED_LAYOUT`, as the falsification.

  Worth knowing why it went undetected: the tests read the assignment
  back out of `conftest.LAYOUT`, so they passed under either wiring while
  serving the rejected one as the worked example. The corrected tests
  derive block order from `LAYOUT` and assert the decoder's contract,
  which holds whichever way the arm is wired.
- ~~The `Measurement` → filter path is only half connected.~~ **Closed
  at P5**: `FilterRunner.ingest` takes a `Measurement` and passes `m.R`
  into `EKF.update`. One piece is deliberately left — `MeasurementLog`
  stores `(t, Z, rows)` and drops `R`, so a run driven from a logged
  session rebuilds the measurements with the decoder's `R`. Teaching the
  log to carry `R` is P7's, with the config loader.

- `sig_acc` / `sig_gyro` in the config are nominal, not measured;
  `SerialIMUSensor.calibrate()` exists to replace them with a real `R`
  and has not been run on the arm. **P6 measured what would happen if it
  were**: an `R` from the static rest window is 3–8× tighter than
  nominal, because it is the noise floor at rest and says nothing about
  motion or model error. Adopting it would make an already
  over-confident filter worse, so `rest_bias` returns it and nothing
  uses it. Do not wire it up without re-reading that test.
- The firmware sends no timestamp, so `SerialIMUSensor` defaults to
  `ArrivalClock(0.0)` — bias and jitter both present. `ClockSync` is
  written, tested, and unused until the Teensy sends `micros()`.
  Switching the firmware to the `csv` line format at the same time
  halves the bytes and drops the regex.
- IMU rate is ~20 Hz (50 ± 1 ms) against a 2 ms physics step, so there
  are ~25 predicts between updates. That ratio, not the filter, is the
  current accuracy limit.
- ~~The chip→link assignment disagrees between `config/estimation.yaml`
  and `software/tests/conftest.py`.~~ **Closed at P7** (see above).
- ~~`config/estimation.yaml` still carries the whole finger section.~~
  **Closed at P7.** `joints`, `finger_ekf`, `finger_ekf_blind`,
  `imu_proximal`, `imu_distal`, `gyro_proximal` and `encoders` moved to
  **ADR-0001 appendix A verbatim** rather than being deleted — their
  comments are the only record of why those numbers were what they were,
  and `psd_act`'s note ("at 1.0e-6 the filter is 959× overconfident") is
  a result, not a setting. Do not copy a value out of that appendix into
  the arm's config: they are continuous-time PSDs and `make_Q` takes
  per-step DWNA sigmas.
- `inputs.rate_hz` in the config was `200` and is now `25` — a wrong
  value corrected at P7, not a finger leftover stripped. No arm run ever
  used 200; the notebook streams at 25 (`STREAM_RATE_HZ`) and the golden
  run was generated at 25. Before P7 nothing read the key.

## Documentation that no longer matches the code

Assume these are historical unless you have checked against the source:

- **`README.md` § 4 (repository structure)** lists directories that do
  not exist and omits `sim/`. **§ 5 (class hierarchy)** describes four
  ABCs — `ProcessModel`, `MeasurementModel`, `StateEstimator`, `Sensor`,
  plus `GaussianState`/`ControlInput` — of which **only `Sensor` and a
  reshaped `Measurement` survive**; the rest were deleted in `1987f00`.
  The "Concrete implementations" table is almost entirely fictional. § 6
  describes the `FusionEngine` that no longer exists.
- **`AGENTS.md`** repeats the "four ABCs are frozen" rule and points at
  `models/base.py` and `estimators/base.py`, which are gone. Its install
  and check commands are still correct (modulo the `[app]` caveat above).
- **`docs/adr/0001-multi-rate-fusion.md`** is the authoritative record of
  the *finger* design. Its reasoning on schedule-invariant `Q` (§ 2.1),
  actuator lag as state (§ 2.3) and late measurements (§ D6) is still
  worth reading before making the same mistakes; its decisions D1 and D5
  have been overruled by the rewrite, and § 8's open items refer to
  deleted code.
- **`docs/theory/finger_imu_ekf.md`** and
  `notebooks/finger_imu_practice.ipynb` are the validated precedent the
  whole approach came from — matched vs mismatched models, why the
  activation must be a state, why `R` inflation does not fix a bias.
  Read § 6 before tuning anything. They describe the finger, not the arm.
- `docs/adr/0002-notebook-to-package.md` is the **current** plan, and
  now also the record of what has been built against it —
  P0 → P5 are marked done with per-phase notes. It is the first thing to
  read before touching the estimation path. Its remaining content: the
  pipeline as it exists, what blocks packaging it, the target module
  layout and a phased roadmap. It frames the work as three capability
  milestones — **M1** simulated estimation (achieved *in the notebook*;
  not yet in the package), **M2** real-time simulated operation with
  decoupled command and sensor clocks, **M3** the same code on physical
  hardware (the objective). Each phase is tagged with the milestone it
  serves.

  **Two phases remain, and neither blocks the other.** **P7.5**
  (`erp.runtime.session`: `Mode`, `Session`, `build_session`, the two CI
  greps of § 4.8, M2 fault injection — plus `build_sensor()`, moved out
  of P7 because only that module may name a driver class, and
  `MeasurementLog` carrying `R`, deferred twice because it changes the
  CSV format the golden run reads) is the last phase before hardware.
  **P8 is what closes M1** — the `ConsistencyReport` plus
  `propagate_to_site`, which empties `scripts/make_golden_run.py`. P8
  depends only on P5, so it is startable now; § 5.1 puts it outside the
  phase groups on purpose.

  Unlike the documents above it, this one is **not** historical, and it
  is kept current as phases land rather than rewritten — a finished phase
  keeps its original text, struck through where the plan was changed, so
  the disagreement stays readable.

If you update one of these documents, say plainly in the text which
parts are historical rather than silently rewriting history — the
measured numbers in them are the main reason they are still worth
keeping.
