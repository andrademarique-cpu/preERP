# ADR-0001 — Multi-rate fusion: contract changes and engine design

- **Status:** Proposed. Decision D1 touches all four ABCs and therefore
  requires sign-off from every module owner (`CODEOWNERS`, CLAUDE.md
  § "Contract types").
- **Date:** 2026-08-23
- **Supersedes:** nothing. Constrains the Sprint-0 ABC freeze.

---

## 1. Context

The system has **three independent clocks**, not one:

| Clock | Typical rate | Notes |
|---|---|---|
| Reference / setpoint (`u`) | 50–200 Hz | piecewise constant (ZOH) between updates |
| Actuator command latch | 500 Hz–1 kHz | plus internal lag `tau = 4 ms` and possible bus/driver transport delay |
| Sensors | IMU ~1 kHz, contact force ~100 Hz | per-sensor rate *and* per-sensor latency |

All prior validated work — `notebooks/finger_imu_practice.ipynb` and its
report [`docs/theory/finger_imu_ekf.md`](../theory/finger_imu_ekf.md) —
runs at a **single fixed 2 ms tick** with every sensor sampled
simultaneously. That assumption is load-bearing in three places, and each
one breaks under multi-rate operation.

CLAUDE.md already states the governing rule for `FusionEngine.step`:
prediction advances to the timestamp of the next measurement, never by a
fixed `dt`. This ADR extends that rule to inputs and records what the
extension costs.

---

## 2. Evidence

### 2.1 The process-noise construction is schedule-dependent

`make_Q` (notebook cell 21) is a DWNA model: one random angular
acceleration drawn per step and held constant, `G = [dt²/2 I; dt I; 0]`.
This is correct at a fixed tick and wrong under variable step. Splitting
one interval into N sub-steps divides the accumulated velocity variance
by N:

```
interval = 20 ms, split N ways
   N |   DWNA vel-var |   CWNA vel-var
   1 |   4.000000e-04 |   2.000000e-02
   2 |   2.000000e-04 |   2.000000e-02
   5 |   8.000000e-05 |   2.000000e-02
  10 |   4.000000e-05 |   2.000000e-02
```

Continuous-time (van Loan) discretization composes exactly, DWNA does not:

```
predict(3 ms) then predict(11 ms)  vs  predict(14 ms)
  CWNA  max|difference| = 1.36e-20
  DWNA  max|difference| = 6.60e-05
```

Under multi-rate the segmentation is dictated by the **sensor schedule**.
With DWNA, adding a faster sensor silently shrinks the process noise for
reasons that have nothing to do with the physics — the filter grows
overconfident because of a wiring change. This is the failure class
§ 6 of the EKF report is about: invisible on a trajectory plot, caught
only by NEES.

### 2.2 `h` genuinely depends on `u`

Established in the report § 3.5 and cell 18: accelerometers read `qacc`,
and `qacc` depends on the servo torque, so `D = ∂h/∂u` is non-zero on
accelerometer channels and exactly zero on gyros and position sensors.
Using the wrong Jacobian gives **9.22 against a true value of 716.35**
for row `acc_i1_y`. An innovation formed as `z - h(x)` is biased on every
accelerometer channel.

### 2.3 Actuator lag must be a state, not a noise term

Report § 4, matched vs mismatched, same data, same trajectory:

| | matched (6 states) | mismatched (4 states) |
|---|---|---|
| `\|acc_i2_y\|` bias | 0.0000 m/s² | 4.0101 m/s² (~80σ) |
| NEES median | 5.96 (target 6) | 344 281 (target 4) |
| ±2σ coverage, q1 | 78.9 % | 0.1 % |

The bias is deterministic and time-correlated, so `Q` can widen `P` but
never remove it, and inflating `R` flattens NIS while leaving NEES
untouched. The actuator's activation state must be carried in the filter
state.

---

## 3. Decisions

### D1 — Amend the four ABCs before the Sprint-0 freeze

These are Sprint-0 *inputs*, not changes to a frozen contract: none of
the ABCs are implemented yet (CLAUDE.md § "Current repo state"). Adopting
them now is cheap; after the freeze each costs unanimous owner approval.

```python
# models/base.py
class ProcessModel(ABC):
    def predict(self, x: NDArray, u: NDArray, dt: float) -> NDArray: ...
    def jacobian(self, x: NDArray, u: NDArray, dt: float) -> NDArray: ...
    def Q(self, dt: float) -> NDArray: ...          # WAS: @property Q

class MeasurementModel(ABC):
    def h(self, x: NDArray, u: NDArray) -> NDArray: ...          # WAS: h(self, x)
    def jacobian(self, x: NDArray, u: NDArray) -> NDArray: ...   # WAS: jacobian(self, x)
    @property
    def R(self) -> NDArray: ...                     # unchanged

# estimators/base.py
class StateEstimator(ABC):
    def predict(self, u: NDArray, dt: float) -> None: ...
    def update(self, m: Measurement, model: MeasurementModel,
               u: NDArray) -> None: ...             # WAS: update(self, m, model)

# sensors/base.py
class Sensor(ABC):
    def read(self) -> Measurement | None: ...       # WAS: -> Measurement (blocking)
    def drain(self) -> list[Measurement]: ...       # NEW: all samples since last call
    def calibrate(self) -> CalibrationResult: ...
    @property
    def measurement_model(self) -> MeasurementModel: ...
```

Rationale per change:

- **`Q(dt)`** — a `@property` cannot express § 2.1. Required.
- **`h(x, u)`, `jacobian(x, u)`, `update(..., u)`** — required by § 2.2.
- **`read() -> Measurement | None` plus `drain()`** — a blocking
  single-sample pull cannot serve a 1 kHz IMU and a 100 Hz force sensor
  in one loop without aliasing or blocking. `drain()` returns
  timestamp-ordered samples accumulated since the previous call.
- **`R` stays a property** — sensor noise does not depend on `dt`, and
  `Measurement.R` already carries the per-sample override.

`ProcessModel.predict(x, u, dt)` already takes `dt`, so it needs no change.

### D2 — Prediction advances event-to-event, not measurement-to-measurement

The CLAUDE.md rule generalizes:

> Prediction advances to the next **event**, where the event set is the
> union of measurement timestamps and control-input change times.

`u` is ZOH, so integrating across a `u` discontinuity with a single value
is a modelling error. At 100 Hz force sensing against a 500 Hz actuator,
every force interval contains roughly five input changes.

```python
def _advance_to(self, t_target: float) -> None:
    t = self._t
    for t_next in [*self.inputs.breakpoints_in(t, t_target), t_target]:
        if t_next > t:
            self.estimator.predict(self.inputs.u_at(t), t_next - t)
            t = t_next
    self._t = t
```

### D3 — Process noise is specified as a continuous-time PSD

Models expose a continuous spectral density and discretize per segment
via van Loan, so that `Q` depends on elapsed time and not on the
measurement schedule:

```
M    = [[-A_c, G q Gᵀ], [0, A_cᵀ]] · dt
E    = expm(M)
F_d  = E[1,1]ᵀ
Q_d  = F_d · E[0,1]
```

**`SIG_ALPHA = 12.0` must be re-derived, not copied.** Its DWNA reading
is "angular acceleration held constant over one 2 ms step" (rad/s²); the
continuous form is a spectral density in rad²/s³.

### D4 — Actuator lag enters as augmented state; the two delays are separated

State becomes `[q, v, a]` per § 2.3. The `filterexact` activation is
exactly integrable at any step, so the augmented block stays exact under
variable `dt`:

```
a(t + dt) = u + (a(t) - u) · exp(-dt / tau)
```

Two delays, handled in different layers — conflating them is the mistake
this decision exists to prevent:

| Delay | Meaning | Where it lives |
|---|---|---|
| Actuation transport `d_act` | command latched at `t` takes effect at `t + d_act` | `InputHistory` query offset: `u_eff(t) = u_at(t - d_act)` |
| Sensor latency | sample taken at `t`, reaches the host later | `Measurement.timestamp` is the **sample instant**; `buffer_horizon` absorbs reordering |

`notebooks/assets/finger_2link.xml` already stubs the actuation-delay
hooks (`nsample` / `delay` / `interp="zoh"`, line 108).

All timestamps must share one monotonic time base. Device-to-host clock
conversion is a `Sensor` adapter responsibility and must never appear in
`fusion/`.

### D5 — MuJoCo stays behind `SimulatedSensor`, never as a `ProcessModel`

`mj_step` is locked to `model.opt.timestep` and cannot take an arbitrary
`dt` without mutating the model — which would also change integrator
behaviour mid-run. The production process models (`ConstantAccelModel`,
`FingerKinematicModel`, `ContactDynamicsModel`) are analytic, so exact
variable-`dt` discretization is available in closed form. Keeping MuJoCo
as the truth generator rather than the filter's process model is what
makes D2 and D3 clean.

### D6 — Late measurements are discarded and counted

A measurement whose timestamp precedes the filter's current time is
dropped, and a per-sensor counter is incremented. `buffer_horizon` is
sized to make this rare. The counter is **asserted in tests**: a silent
discard path is how a sensor quietly stops contributing while every plot
still looks correct.

Rollback-replay and Larsen retrodiction are deliberately deferred. The
first target is offline replay (§ 5), where a globally sorted dataset
makes out-of-order arrival impossible by construction.

---

## 4. Consequences

**Accepted costs**

- All four ABCs change before freeze; every owner must approve.
- `SIG_ALPHA` and any tuning derived from it must be re-derived.
- `Q(dt)` is computed per segment. At 1 kHz with several input
  breakpoints per interval this is a real cost; a matrix-exponential
  cache keyed on `dt` is the expected mitigation if profiling demands it.
- `EKF.update` currently recomputes `H` (2·nx finite differences) on
  every call. Under multi-rate that runs at the fastest sensor's rate.
  Relinearization cadence becomes a tuning parameter.

**Preserved**

- The import direction is untouched. `InputHistory` is pure time/data
  logic in `core/`, so `fusion/` may use it while `estimators/` and
  `models/` receive `u` as an argument and import nothing new.

**Rejected alternatives**

- *Fuse gyros only* (report § 4, option 2). Makes `D = ∂h/∂u` exactly
  zero and would let `h(x)` stand, but abandons accelerometer fusion,
  which is the project's stated purpose.
- *Absorb the mismatches in `FusionEngine`.* Leaves accelerometer fusion
  inconsistent in a way no `Q` or `R` tuning repairs (§ 2.3).
- *Keep DWNA and re-tune per deployment.* Makes the noise model a
  function of the wiring harness.

---

## 5. Implementation phases

| Phase | Deliverable |
|---|---|
| P0 | This ADR, signed off by all owners |
| P1 | `core/timeline.py` — `ControlInput`, `InputHistory` (ZOH + delay), event merge |
| P2 | Process model with van Loan `Q(dt)` and the augmented actuator state |
| P3 | `FusionEngine` — event-segmented prediction, per-sensor partial update, D6 counters |
| P4 | First `config/` YAML: per-sensor rates and latencies, input rate, `tau`, `d_act`, PSD |
| P5 | Multi-rate NEES/NIS harness with known injected process noise |

First target is **offline replay** over a globally sorted dataset, so the
multi-rate mathematics is validated before latency handling is added.
This also keeps the zero-hardware-attached principle intact throughout.

`config/` is currently empty apart from `.gitkeep`, so P4 creates the
first file and sets the precedent for the "register new estimators in
`config/` YAML" convention.

---

## 6. Test obligations

Filter correctness is judged by consistency checks, not plots. Following
the falsification pattern the EKF notebook already uses for the
`A[2,0] = 0` row:

1. **Composition** — `predict(dt1)` then `predict(dt2)` equals
   `predict(dt1 + dt2)` for mean *and* covariance to machine precision.
   Sharpest test available here, and the direct analogue of the report's
   J-invertibility NEES check (measured 1.0e-09).
2. **Schedule invariance** — identical data delivered on different
   measurement schedules yields the same posterior at common timestamps.
3. **Falsification** — a fixed-`dt` engine variant and a DWNA-`Q` variant
   must **fail** NEES. If they pass, the consistency test is too weak to
   be worth keeping.
4. **Discard counter** — asserted explicitly per D6.
5. **Multi-rate NEES/NIS** — heterogeneous rates with an explicitly
   injected torque disturbance, so `Q` is known rather than tuned
   (report § 3.6). Monte Carlo over independent seeds for a real ANEES;
   the report notes single-run median is only a robust proxy.

Per CLAUDE.md, these target the ABCs, so the same tests must run against
a stub and against the real implementation.

---

## 7. Measured after implementation

P1-P4 are implemented and the ADR-0001 test obligations pass. Numbers from
`software/tests/test_consistency.py`, pooled median NEES over 20 Monte Carlo
runs on a two-sensor schedule (97 Hz encoders, 253 Hz gyros, 200 Hz inputs;
162 segments spanning 0.043-3.953 ms, a 91x spread). Target is the
chi-squared median for nx = 6, which is 5.348.

| Filter | median NEES | ratio to target |
|---|---|---|
| Reference (van Loan `Q(dt)`) | 5.286 | 0.99 |
| Constant `Q` (what a `Q` property forces) | 3.698 | 0.69 |
| DWNA `Q`, tuned at the nominal rate | 314 375 | 58 800 |

Stable across five disjoint 20-run blocks: the reference stays within
0.967-1.016 of target, the constant-`Q` variant within 0.675-0.692.

**This refines D1, and not entirely in its favour.** The DWNA failure is as
severe as § 2.1 predicted, five orders of magnitude, matching the
mismatched-model result in the EKF report. But a *constant* `Q` is only
mildly wrong here: biased low, meaning conservative, by about 30 %. So the
honest argument for `Q(dt)` is that a dt-less `Q` cannot be correct, only
conservatively wrong by a schedule-dependent margin -- not that it blows up.
Overstating that would be the easiest way to get this ADR's reasoning quietly
disbelieved later.

Two further findings from implementation:

- **Applying the actuation delay by subtraction at query time is a bug.**
  With a 20 ms delay, `0.12 - 0.02` evaluates to `0.09999999999999999`, so a
  query exactly at a breakpoint silently returns the *previous* input.
  `InputHistory` therefore stores effective times (latch + delay) computed
  once on push, which makes `breakpoints_in` and `u_at` agree exactly. That
  round-trip is a hard requirement, since `FusionEngine` splits an interval at
  those breakpoints and then asks for the input on each side.
- **The servoed-finger model is stiff**, carrying `A_c` entries of order
  `kp/inertia = 5e4`. Composition holds at machine precision (~1e-16) for
  intervals up to ~20 ms, but by 50 ms scaling-and-squaring inside `expm` has
  degraded it to ~1e-7. Prediction gaps are bounded by the slowest sensor, so
  this is comfortable at the rates in `config/estimation.yaml` -- but a sensor
  slower than roughly 20 Hz would need this revisited.

## 8. Open items

- Value of `d_act` for the real bus/driver chain — currently unmeasured.
- Relinearization cadence for `H` under multi-rate (P3 profiling).
- `psd_alpha` in `config/estimation.yaml` is a placeholder. Re-deriving it
  from the physics, rather than converting the notebook's `SIG_ALPHA = 12.0`,
  is the blocking item before these filters mean anything on real data.
- Discretisation accuracy degrades for prediction gaps beyond ~20 ms (§ 7).
  Revisit if any sensor slower than ~20 Hz joins the stack.
- Whether contact force sensing needs its own process-model branch
  (`ContactDynamicsModel`) at contact transitions, where the report notes
  proprioception is weakest. Out of scope here; likely ADR-0002.
