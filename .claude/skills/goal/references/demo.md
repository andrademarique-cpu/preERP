# The three-way demo

What it shows: the arm moving in a live MuJoCo viewer, beside live plots in
which three traces of the *same* quantity are overlaid — what the plant actually
did, what a noisy virtual sensor reported, and what the EKF estimated from that
report alone.

This is P8's acceptance artifact (`erp.viz.three_way` is named for it in
ADR-0002 § 4.1) and the thing to build when the user asks to *watch* the
estimator.

## The three signals, precisely

Getting these confused produces a demo that looks convincing and means nothing.

| Trace | Source | Notes |
|---|---|---|
| **Truth** | the **un-edited plant** model — `erp.sim.plant.load_model`, stepped under the commanded trajectory | Not `blind_variant`. The plant model is what *generates* truth and is never what the filter runs on; `test_plant.py` asserts the plant fails the filter's `na == 3` contract. |
| **Measured** | `SimSensor` over that recorded output, which adds white noise drawn from `R` and releases samples at the IMU's ~20 Hz with latency | The channel values are in MuJoCo sensordata layout, addressed by `rows`. |
| **Estimated** | the EKF's state, pushed back through the observation model: `h(x̂)[rows]` for a sensor channel, or `propagate_to_site` for the effector | The filter runs on `blind_variant` (`na` 0 → 3, `nx` 8 → 11) and never sees the command. |

Plot at least one accelerometer channel, one gyro channel, and the effector
position with its ±σ band. The effector is where the estimate earns its keep,
because it is the quantity nobody measures directly.

## Say what the demo does and does not show

**A `SimSensor` draws its noise from the same `R` the filter is given, and the
plant is the model.** So the filter is being handed exactly the world it
assumes. Measured on the existing dry run: NIS median **9.6** against a target
of 12, nothing discarded, over 1500 predicts and 61 updates. On real data the
same filter gives NIS median **29**.

That gap is the honest content of the demo. It is a **plumbing check** — that
the timestamps, the row indices, the axis maps and the loop all line up — not
evidence that the estimator is good. Put that sentence in the demo's own output
or docstring, not only in the commit message. A demo that implies more than it
shows is the failure mode here, and this project's documents are careful about
it everywhere else.

If you want the demo to be evidence, give it a way to fail: a toggle that
mis-wires the layout, or inflates the true noise above `R`, or starts the filter
from zeros. The estimate should visibly degrade. Pair the good run with one of
those, the same way `test_clock.py` pairs its bound with arrival stamping.

## Traps specific to this model

- **Initialise from the servoed equilibrium, not zeros.**
  `erp.sim.plant.warmup_to_rest` warms up 100 steps under the command. Measured
  `link2_acc_x` at rest: **9.810** from the warmed equilibrium, **4.144** from
  the `home` keyframe with `act = 0` and no warmup, against ~9.65 on the real
  IMU. On a channel whose sigma is 0.05 that 5.7 m/s² offset is a **114-sigma**
  bias, and no `Q` or `R` tuning repairs it. (Careful with the counterfactual:
  zeroing the *whole* state gives 345 m/s², because that also breaks the tendon
  equality — a different and much louder failure. Quoting it would overstate
  this one.)
- **Advance with `advance_to_safe(tick.t_actual)`.** See `invariants.md` § 6.
  With `advance_to(now)` the demo still runs and still draws, it just silently
  drops samples — 1 of 61 at 5 ms latency, up to half at 20 ms.
- **Never hardcode sensordata slices.** Use `erp.sim.plant.sensor_layout(model)`
  or `erp.sensors.mujoco.rows_of(model, *layout)`. Adding a sensor above the
  others in the XML leaves every hardcoded `[:, 0:3]` returning numbers from the
  wrong channel — the exact failure `sensor_layout` exists to prevent, and one
  the notebook already hit once.
- **Take the layout from `config/estimation.yaml`**, via
  `erp.io.config.load_config` and `build_decoder`. It is the single definition
  since P7; a fifth copy in the demo would undo that.
- **Plot the full covariance block, not `sqrt(diag(P))`.** Over the golden run
  the effector ellipsoid's major axis sits a median 24.9° off the nearest world
  axis, so per-axis sigmas understate the worst direction by up to 39.7%. If the
  demo draws an ellipse, draw the real one.

## Where it lives

- **`erp/viz/`** — the figure functions: `theme`, `three_way`, `sensor_compare`,
  `effector_band`. Pure geometry plus matplotlib, gated behind the `[viz]`
  extra. The package's current docstring says "geometry only, no backend
  imports"; ADR-0002 § 4.2 explicitly supersedes that, because the old rule
  produced an empty package while the real plotting accumulated in notebook
  cells. Update the docstring when you land it.
- **`erp/analysis/`** — `estimate_lag`, `ConsistencyReport`, `propagate_to_site`.
  Read-only consumers of a finished run. Moving `estimate_lag` and
  `site_position_cov` here is what finally empties
  `scripts/make_golden_run.py` and closes M1.
- **A script or notebook cell** — the live loop itself. It needs a display and
  it blocks, so it is never a test and never runs in CI.

Test the geometry (does `three_way` return the right shapes, does the ellipse
match the covariance) in the fast suite. Do not try to test the viewer.

## Live viewer mechanics

`notebooks/viewer.ipynb` already has the pattern to follow —
`mujoco.viewer.launch_passive` plus a `SimLog` that grows preallocated arrays
for `qpos`, `qvel`, `ctrl` and `sensordata`. Reuse it rather than inventing a
second one; if it needs to move into the package to be reusable, that is a
relocation and follows § 5.3 (move unchanged, then clean up).

Three practical notes:

- **Pace the loop with `RateLoop`**, not a cumulative `sleep(1/rate)`. Measured
  over 76 ticks at 25 Hz: cumulative sleeps finish **+65.4 ms** late with
  monotonically growing error; absolute deadlines finish **+0.0 ms** with a
  worst per-tick error of 0.7 ms. `sleep(2 ms)` actually measures ~2.49 ms here.
- **Do not block the viewer on the plots.** Update the figure every N ticks with
  `plt.pause` or a blitted artist; redrawing a full matplotlib figure at 25 Hz
  will not keep up and the arm will stutter in a way that looks like a physics
  problem.
- **Real time is not the physics step.** The model runs at a 2 ms tick and the
  IMU at ~20 Hz, so there are ~25 predicts between updates — that ratio, not the
  filter, is the current accuracy limit. Make the plot show update instants
  distinctly (markers, not just a line) so the viewer can see it.

## Definition of done

- Runs headless and writes the figures; `--live` additionally opens the viewer.
- Takes its wiring from `config/estimation.yaml`, its model from
  `erp.sim.plant`, and its timing from `core.clock` — no hardcoded copies.
- Names in its own output what it is and is not evidence for.
- Ships with a degraded-run toggle that visibly fails.
- `erp/viz/` and `erp/analysis/` geometry covered by fast tests; the live loop
  covered by nothing, deliberately, and said so.
