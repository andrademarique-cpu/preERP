# The interactive estimator demo — mechanics

Everything in this file was checked against **this machine** on 2026-09-22:
mujoco 3.11.0, pyqtgraph 0.14.0, PyQt5, Python 3.11.15 in the `erp` env (there
is no `EKF` env; the name was wrong here and in CLAUDE.md). Where a number
appears it was measured, not recalled.

## The three signals

Confusing these produces a demo that looks convincing and means nothing.

| Trace | Source | Note |
|---|---|---|
| **Truth** | the **un-edited plant** (`erp.sim.plant.load_model`), stepped under the keyboard's `ctrl` | Never `blind_variant`. `test_plant.py` asserts the plant *fails* the filter's `na == 3` contract. |
| **Measured** | a live virtual IMU over that plant: `N(0, R)` noise, ~20 Hz, with latency | MuJoCo sensordata layout, addressed by `rows`. |
| **Estimated** | the blind EKF on `blind_variant`, pushed back through `h(x̂)[rows]`, or `propagate_to_site` for the effector | Never sees `ctrl`. |

**Say what it does not show.** The virtual IMU draws its noise from the same `R`
the filter is given, and the plant *is* the model — the filter is handed exactly
the world it assumes. On the three-way demo that lands at NIS median **9.59**;
the same filter on the real arm gives **29**. This is a plumbing check
(timestamps, row indices, axis maps, loop), not evidence the estimator is good.
Print that sentence in the demo's own output, not only in the commit message.

## The model, as it actually is

```
PLANT   nq 4  nv 4  nu 3  na 0   nsensordata 15   dt 0.002   ngeom 27
BLIND   nq 4  nv 4  nu 3  na 3   nx = 4+4+3 = 11             ngeom 27
joints      rot, link1, link2, act
actuators   rot_servo, link1_servo, link2_servo
sensors     link1_acc, link2_acc, link1_gyro, link2_gyro, efector_pos
```

**"All joints" is three, not four.** `act` is tendon-coupled with range `[0, 0]`
and is not commandable — it follows. Say so in the on-screen help rather than
letting the user hunt for a key that does nothing.

**The ranges are asymmetric**, so a naive ±step from zero leaves the range
immediately on two of three:

| Actuator | ctrlrange (rad) |
|---|---|
| `rot_servo` | −2.79 … +2.79 |
| `link1_servo` | 0 … 1.57 |
| `link2_servo` | 0 … 1.04 |

Clamp every jog to `model.actuator_ctrlrange`. Read it from the model — do not
copy the table above into code.

## The scene, and why the floor is inside a body

`mechanical/mujoco_assets/scene.xml` holds the sky, the checkered ground and the
offscreen framebuffer size; the arm XML pulls it in with one `<include/>` as its
last child. All visual — stepped 1848 times (the golden run's length) against a
model with the scenery deleted, `max |sensordata diff|` is **0.000e+00**, and
`make_golden_run.py --check` still reproduces at `rtol=1e-12`.

**The floor lives in its own `<body name="scenery">`, and that is load-bearing.**
MuJoCo numbers geoms by body and the world is always body 0, so a geom hung
straight off `<worldbody>` takes id 0 and shifts every geom of the arm up by one
*wherever it is written in the file* — which moves the mesh ids the ghost looks
up. In its own static body, created after the arm, it collects the last id
instead: meshes stay at `[0, 1, 2, 10, 18]` with `geom_dataid [0, 1, 2, 3, 4]`,
and `floor` is geom 26. `test_plant.py` asserts both, and the falsification was
run: lifting the geom into `<worldbody>` gives meshes `[1, 2, 3, 11, 19]`.

Checker squares are `1 / texrepeat` metres — **measured**, by rendering from a
known height and counting. The obvious reading, `1 / (2 * texrepeat)` because the
builtin checker is a 2×2 pattern, is off by a factor of two.

**`mujoco.Renderer` works headlessly here**, which is how the look was checked
without opening a window. Two gotchas, both hit: the offscreen framebuffer
defaults to 640x480 and the renderer refuses anything larger (`scene.xml` now
sets 1280x720), and a second `Renderer` in the same process renders black unless
the first is `.close()`d.

## Verified API surface

```python
mujoco.viewer.launch_passive(
    model, data, *,
    key_callback: Callable[[int], None] | None = None,
    show_left_ui: bool = True,
    show_right_ui: bool = True,
) -> Handle
```

`Handle` has: `sync`, `is_running`, `close`, `lock`, `cam`, `opt`, `perturb`,
`user_scn`, `viewport`, `set_texts`, `set_figures`, `set_images`.

- **It does not block.** It returns a handle; you call `.sync()` yourself. That
  is what makes a Qt-owned main loop possible.
- **`key_callback` receives a GLFW keycode as an `int`.** For letters that is
  `ord('W')` etc. Arrows are 262–265 (right, left, down, up). Letter keys are
  the simpler, more portable choice.
- **Keys only reach the focused window.** The viewer window and the plot window
  are separate. Either accept keys in both (a Qt `keyPressEvent` alongside
  `key_callback`) or say in the help text that the 3D window must have focus.
- `user_scn` is an `MjvScene` you populate yourself: set `ngeom`, fill
  `geoms[i]`, and it is composited over the rendered scene.
- On macOS `launch_passive` requires `mjpython`. This repo runs on Windows, so
  it is a note, not a blocker.

## The ghost overlay, and the trap in it

Run `mj_forward` on a second `MjData` holding x̂, then copy each geom's world
pose into `user_scn`:

```python
MESH = int(mj.mjtGeom.mjGEOM_MESH)
scn = viewer.user_scn
scn.ngeom = 0
for i in range(mb.ngeom):
    if scn.ngeom >= scn.maxgeom:
        break
    g = scn.geoms[scn.ngeom]
    mj.mjv_initGeom(
        g, int(mb.geom_type[i]), mb.geom_size[i],
        db.geom_xpos[i], db.geom_xmat[i].reshape(9), GHOST_RGBA,
    )
    if int(mb.geom_type[i]) == MESH:
        g.dataid = 2 * int(mb.geom_dataid[i])  # <-- NOT set by initGeom, and NOT geom_dataid
    g.objtype = int(mj.mjtObj.mjOBJ_UNKNOWN)  # not pickable
    scn.ngeom += 1
```

There are **two** traps in that one field, and the second was found the hard way
— it shipped in `efc8df7` and was what "the ghost is oriented wrong" turned out
to be.

**1. `mjv_initGeom` leaves `dataid = -1` on every geom**, verified. It sets the
fields it is passed and defaults the rest, and `dataid` is not one of them.

This model's 27 geoms are **5 meshes, 12 spheres, 9 cylinders and the ground
plane**, and the meshes at indices `[0, 1, 2, 10, 18]` are *the visible arm*.
Forget the `dataid` line and the spheres and cylinders still draw, so the ghost
half-appears — a scattering of decorations where the arm should be.

**2. The value is `2 * geom_dataid`, not `geom_dataid`.** The render context
holds **two entries per mesh**, and `mjv_updateScene` — MuJoCo building the same
scene for itself — writes the doubled index. Measured: `geom_dataid` over those
five geoms is `[0, 1, 2, 3, 4]`; the renderer's own scene gives them
`[0, 2, 4, 6, 8]`.

Pass the undoubled value and geom `k` draws mesh `k // 2`: base_link's hull on
`rot`, rot's mesh on `link1`, rot's hull on `link2`, link1's mesh on `act`. Every
mesh carries its own compiled frame (`mesh_quat` is ~120° off identity on four of
the five), so a wrong mesh at a right frame **reads as a rotation bug**. That is
what makes it expensive: you go looking at `geom_xmat`, at `mj_forward`, at the
filter's state — and `mat` is fine. Verified: MuJoCo's own scene geoms match
`geom_xmat` exactly, so `mjv_initGeom` copies the orientation correctly.

The geom construction is pure array work and **runs headless**, so test it —
but not against a hand-written expectation. The original test asserted
`dataid == geom_dataid`, which proved the field had been *set* and never that it
had been set to what the renderer reads, so it pinned the bug for two commits.
**Use `mjv_updateScene` as the oracle**: build a real scene over the plant at the
same `qpos`, index it by `objid` (it emits 29 geoms against the model's 26, so
order is not identity), and assert type, pos, mat *and* dataid all agree.
`test_viz_live.py` does this now.

**Draw only what can move.** `ghost_geom_ids` keeps geoms whose body is not
welded to the world (`body_weldid != 0`) — 25 of 27 here. The two it drops are
the ground plane from `scene.xml`, which would otherwise be painted as a
translucent plane exactly coplanar with the real floor (z-fighting and a
blue-tinted ground), and `base_link`, which is bolted down and so can never show
error. **Scene slot is therefore not model geom id**: anything mapping back has
to go through `ghost_geom_ids`.

Alpha ~0.35 reads as a ghost without hiding the truth. Keep `maxgeom` headroom —
25 drawn geoms against a 200 default is not close.

## Timing

| Quantity | Value |
|---|---|
| Physics step | 2 ms (500 Hz) |
| IMU | ~20 Hz, 5 ms latency |
| Render / tick | ~60 Hz (8 physics steps per tick) |
| Paired EKF predict+update | ~131.5 µs |
| Filter cost per wall-clock second | 500 × 131.5 µs ≈ **66 ms, ~7% duty** |

Real time is comfortable. The headroom goes to rendering and plotting, so
decimate the strips (~30 Hz) rather than pushing a point per physics step.

- **Pace with `core.clock.RateLoop`**, or a Qt timer on absolute deadlines —
  never a cumulative `sleep(1/rate)`. Measured over 76 ticks at 25 Hz: the
  cumulative loop finishes **+65.4 ms** late with monotonically growing error;
  absolute deadlines finish **+0.0 ms**, worst per-tick 0.7 ms. `sleep(2 ms)`
  measures 2.49 ms here.
- **Advance with `advance_to_safe(t)`, not `advance_to(now)`.** A sensor stamps
  the *sample* instant and hands it over later, so a filter advanced to now is
  ahead of the next arrival and drops it. At 5 ms latency that is 1 of 61; at
  20 ms, up to half. It is intermittent and phase-dependent, so the demo still
  runs and still draws — which is exactly what lets it survive unnoticed.
- **~25 predicts per update.** That ratio, not the filter, is the accuracy
  limit. Mark update instants on the strips (markers, not a line) so it is
  visible.

## Why `SimSensor` cannot be reused

`SimSensor` is a `ReplaySensor` over a precomputed `(t, sensordata)` array; its
noise is drawn **once, up front**, and `rate_hz`/`latency_s` decimate and pace
an existing log. Teleoperation has no log — truth is produced as keys are
pressed.

So: a new `Sensor` subclass that samples a live `MjData`. It belongs in
`erp/sensors/` (hardware-shaped input enters through `Sensor`, always), it takes
its `rows`/`R` the same way, and it joins `FACTORIES` in
`test_sensor_contract.py` — the one parametrised suite that runs against every
`Sensor`, rather than bespoke tests.

Keep `SimSensor`'s honesty: expose the clean reading alongside the noisy one, so
the "measured vs truth" strip is free.

## Traps carried over from the three-way demo

- **Initialise from the servoed equilibrium**, `erp.sim.plant.warmup_to_rest`,
  not the keyframe. `link2_acc_x` at rest: **9.810** warmed, **4.144** from the
  `home` keyframe with `act = 0`, against ~9.65 on the real IMU — a 113-σ bias
  in `h(x₀)` on a channel whose sigma is 0.05. It does **not** survive the run
  (the tendon equality recovers in ~2 steps), so do not claim it wrecks the
  filter. Zeroing the *whole* state instead gives 345 m/s², a different and much
  louder failure.
- **Never hardcode sensordata slices.** Use `erp.sim.plant.sensor_layout` or
  `erp.sensors.mujoco.rows_of`. A sensor added above the others in the XML
  leaves every hardcoded `[:, 0:3]` reading the wrong channel — the notebook hit
  this once already.
- **Take the wiring from `config/estimation.yaml`** via
  `erp.io.config.load_config` / `build_decoder`. It has been the single
  definition since P7; a fifth copy in the demo would undo that. IMU_0 → link1,
  IMU_1 → link2.
- **Plot the full covariance block, not `sqrt(diag(P))`.** The effector
  ellipsoid's major axis sits a median 24.9° (max 45.0°) off the nearest world
  axis, so per-axis sigmas understate the worst direction by a median 9.6% and
  up to 39.7%. `propagate_to_site` returns the full 3×3.
- **One 12-channel update, not four 3-channel ones** — ~360 µs against ~1.4 ms.
  One sample becomes exactly one `Measurement`.

## A keymap that fits the hardware

Three actuators, so three jog pairs plus the controls the demo needs. Print it
on screen; do not make the user read the source.

```
 Q / A    rot_servo    +/-          P       toggle ghost
 E / D    link1_servo  +/-          O       re-home (warmup_to_rest)
 U / J    link2_servo  +/-          X       stop
                                    ESC     quit
```

**Every letter A–Z is already a viewer shortcut**, so there is no key the viewer
ignores — `mjVISSTRING` and `mjRNDSTRING` between them assign one to all 26. What
you get to choose is *which* collision, and the two classes are not equivalent:

| Class | Examples | Reachable from Python? |
|---|---|---|
| `mjvOption` flags | `Q` camera, `A` auto-connect, `E` equality, `D` static body, `U` actuator, `J` joint, `O` perturb object, `P` contact split, `X` texture | **Yes** — `viewer.opt`, so re-assert it every tick and the toggle lasts under one frame |
| render flags | `W` wireframe, `S` shadow, `R` reflection, `G` fog, `K` skybox, `L` additive | **No** — they live on the internal `mjvScene`, and the passive `Handle` exposes only `cam`, `opt`, `perturb`, `user_scn`, `m`, `d` |

The keymap above uses only the first class. The one it replaced used `W`/`S` for
link1 and `G`/`R` for ghost/re-home — all four in the second — so jogging a joint
turned on wireframe and killed the shadows, which reads as the ghost rendering
badly rather than as the viewer working correctly. Snapshot `viewer.opt.flags`
and `viewer.opt.geomgroup` after `launch_passive` and rewrite them at the top of
each tick under `viewer.lock()`; pass `show_left_ui=False, show_right_ui=False`
as well, but do not rely on hiding the panels alone — whether that also stops the
dispatch was never verified.

**`key_callback` fires on press only, never on release.** So "jog while held" is
not implementable: there is no release event to stop on. Either one press is one
step (0.01 rad at 0.6 rad/s and 60 Hz — 150 presses to cross a range), or the key
latches and an explicit stop key clears it. The demo latches. A rate of roughly
0.5 rad/s is controllable; faster than ~2 rad/s and the servos saturate, which
looks like a filter problem and is not.
