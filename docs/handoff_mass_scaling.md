# Handoff — mass scaling for the MyPalletizer 260 MuJoCo model

**Purpose of this document:** hand a second reviewer (human or AI) enough context to
critically evaluate a proposed change to a MuJoCo model, without access to the repo.
Every number below was measured against the compiled model, not estimated. The
questions I actually want answered are in §7.

---

## 1. Objective

A MuJoCo model of an Elephant Robotics **MyPalletizer 260** arm, generated from
Fusion 360 CAD, reports a total mass of **3.4573 kg**. The real arm weighs
**960 g** (figure supplied by the owner, not independently verified here).

The goal is to bring the model to 960 g **without breaking the physics that was
derived from the wrong masses** — specifically the inertia tensors and the
per-joint servo gains, which were sized from the model's own gravity torques and
joint-space inertia.

Naively editing `mass=` attributes would leave inertias and gains inconsistent.
The proposal in §5 is intended to be self-consistent; I want it challenged.

---

## 2. Toolchain and file layout

```
Fusion 360 CAD
  -> Fusion2Mujoco v1.0 (github.com/jgillick/Fusion2Mujoco)
    -> MJCF: preERP/mechanical/mujoco_assets/MyPalletizer260/freshfile_mujoco.xml
      -> consumed by: preERP/notebooks/viewer.ipynb (interactive MuJoCo viewer +
         a jitter-diagnostic tool)
```

The exporter emits visual STL meshes only, no collision primitives, and no
actuators or sensors. Everything in §3 beyond geometry was added by hand across
previous sessions.

MuJoCo Python bindings, `dt = 0.002 s`.

---

## 3. Current model state (all measured)

**Structure:** 6 bodies (incl. `world`), 4 hinge joints, 5 mesh geoms, 4
actuators, 3 sensors.

```
world -> base_link -> rot -> link1 -> link2 -> act
```

`base_link` has **no joint**, so MuJoCo welds it to the world (`body_weldid = 0`).
Its mass is in the model but no joint carries it and gravity never moves it.

| body | mass (kg) | moves? |
|---|---|---|
| base_link | 1.0931 | no — welded to world |
| rot | 0.6514 | yes |
| link1 | 0.6544 | yes |
| link2 | 0.8817 | yes |
| act | 0.1767 | yes |
| **total** | **3.4573** | |
| **moving only** | **2.3642** | |

**Hand-added on top of the raw export:**

- `<geom contype="0" conaffinity="0" />` in `<default>` — collisions off. Without
  it the convex hulls of the *visual* meshes collide: 13 contacts at rest with up
  to 88 mm interpenetration, and `rot`/`link1` are pinned (the parent-child
  collision filter does not apply when the parent is the world body, which
  `base_link` is).
- Per-joint position servos (see table below), `ctrlrange = ±3.14 rad`.
- Sites `link1_site`, `link2_site`, `efector`; sensors `link1_acc`, `link2_acc`
  (accelerometers), `efector_pos` (`framepos`).
- `rot` joint axis/anchor cleaned from float noise to `axis="0 1 0" pos="0 0 0"`.

**Servo gains currently in the file:**

| joint | kp | kv | forcerange (N·m) |
|---|---|---|---|
| rot | 1.62 | 0.477 | ±1.0 |
| link1 | 63.35 | 3.342 | ±7.0 |
| link2 | 29.75 | 0.955 | ±2.0 |
| act | 24.58 | 0.143 | ±0.3 |

These were derived in an earlier session from the only hardware datum available,
`v_max = 120 deg/s = 2.094 rad/s`, plus gravity torque sampled over a 13⁴ pose
sweep:

```
forcerange ≈ 3 × worst gravity torque for that joint
kv         = forcerange / v_max        (motor torque-speed line)
kp         = kv² / (4·M_max)           (ζ = 1 at the least-damped pose)
```

The stability criterion being targeted is `kv·dt/M < 2` — the explicit-integration
limit that applies when the actuator saturates, because at saturation
`d(force)/d(velocity) = 0`, `implicitfast` loses its implicit term, and the
integrator degenerates to explicit.

**Measured `kv·dt/M`, and its pose dependence:**

| pose | rot | link1 | link2 | act |
|---|---|---|---|---|
| `q = 0` | 0.40 | 0.44 | 0.25 | **1.37** |
| `[0, 0.4, −0.7, 0.3]` | 0.12 | 0.37 | 0.25 | **1.37** |
| `[0, 1.2, −2.0, 0]` | 0.03 | 0.19 | 0.25 | **1.37** |

`act` is pose-independent (it is the distal-most body; its own inertia does not
change with configuration) and sits closest to the limit.

**Integrator — a known inconsistency.** The XML does **not** contain an
`<option integrator="..."/>` element, so it compiles with MuJoCo's default
**Euler**. The gain derivation assumes `implicitfast`. The notebook patches
`model.opt.integrator` at runtime, so the notebook is fine; any other consumer of
the XML is not. Measured at `ctrl = 0`, after 8 s settle, over the next 2 s:

| integrator | `act` mean | `act` peak-to-peak | max \|qvel\| |
|---|---|---|---|
| Euler (what the XML compiles to) | −0.405° | **0.365°** | **2.883 rad/s** |
| implicitfast (what the gains assume) | −0.008° | 0.000° | 0.000 rad/s |

Under Euler the arm never settles — `act` sits in a permanent limit cycle.

---

## 4. Why the model mass is wrong

Fusion assigned a solid-material density to CAD bodies that are, in reality,
thin-walled shells with servo motors inside. The geometry is right; the density
is not. Model/real = 3.4573 / 0.960 = **3.60×** too heavy.

---

## 5. The proposal

### 5.1 Scale density uniformly

```
s = 0.960 / 3.4573 = 0.277676
```

| quantity | scales by | reason |
|---|---|---|
| `mass` | s | m = ρV |
| `fullinertia` (all 6 components) | s | I = ∫ρr²dV, geometry unchanged |
| per-body CoM (`ipos`) | **1** | a uniform density change does not move a centroid |
| gravity torque τ(q) | s | linear in mass |
| joint-space inertia M(q) | s | linear in mass |

Verified: every body scales by exactly 0.277676 in both mass and inertia,
`body_ipos` delta = 0, τ ratio = 0.277676 on every joint, M(q) ratio = 0.277676
on every joint.

### 5.2 Scale the servo gains by the same `s`

Pushing `s` through the gain recipe:

```
forcerange ∝ τ_max        ∝ s
kv = forcerange / v_max   ∝ s
kp = kv² / (4·M_max) = s²/s ∝ s
```

So all three scale by `s`, and every dimensionless quantity is then **invariant**:
`kv·dt/M`, `ω_n = √(kp/M)`, and droop `= τ/kp`. The arm moves identically; only
the forces shrink.

Measured, identical step command `[0.5, 0.3, −0.4, 0.2]`, 3 s:

| | max \|Δqpos\| vs unscaled | `act` kv·dt/M |
|---|---|---|
| mass **and** gains scaled | **1.9e-16 rad** | 1.37 (unchanged) |
| mass scaled, gains untouched | 0.118 rad (6.78°) | **4.94** — past the limit of 2 |

Scaling mass alone would drive `act` back into chattering, worse than the bug
already fixed.

### 5.3 Bake it into the XML, do not call `mj_setTotalmass` at runtime

`mj_setTotalmass()` correctly rescales `body_mass` and `body_inertia` but leaves
`body_subtreemass` **stale**, which corrupts `subtree_com`:

```
subtree_com of the arm   before: [-0.02370, 0.00024, 0.05227]
                          after: [-0.00658, 0.00007, 0.01451]   <- scaled by s
body_subtreemass after scaling: [3.4573 3.4573 2.3642 1.7128 1.0584 0.1767]  (unchanged)
```

The dynamics survive this (`subtree_com` is only a reference origin for spatial
velocity/acceleration and the results are invariant to it), and the step-response
check in §5.2 confirms that. But any CoM readout or `subtreecom` sensor is
silently wrong by a factor of `s`. Editing the XML makes the compiler recompute
all derived constants.

### 5.4 The numbers

```
mass=   base_link 1.0930923393163645 -> 0.3035255800681111
        rot       0.6513790826917025 -> 0.18087238086572255
        link1     0.6543664159166849 -> 0.1817018918021331
        link2     0.8817118285851836 -> 0.24483027151357356
        act       0.17671625983548697 -> 0.049069875750459596

fullinertia:  all 6 components of every body × 0.277676066

joint      kp -> kp'        kv -> kv'         forcerange -> new
rot      1.62 -> 0.4498    0.477 -> 0.13245    1.0 -> 0.2777
link1   63.35 -> 17.5908   3.342 -> 0.92799    7.0 -> 1.9437
link2   29.75 -> 8.2609    0.955 -> 0.26518    2.0 -> 0.5554
act     24.58 -> 6.8253    0.143 -> 0.03971    0.3 -> 0.0833
```

---

## 6. Concerns I already hold

Listed strongest first. I would rather have these confirmed or demolished than
agreed with politely.

### 6.1 Uniform scaling probably *underestimates* inertia (my main worry)

For a fixed outer shape, a hollow shell has a larger radius of gyration than a
solid body of the same mass. Uniform density scaling gives
`I = m_real · k²_solid`, but the truth is `I = m_real · k²_shell` with
`k²_shell > k²_solid`. Classic bounds: thin spherical shell vs solid sphere is
2/3 R² vs 2/5 R² (1.67×); thin tube vs solid cylinder about its axis is 2×.

So the proposal likely under-models inertia — plausibly by 1.5–2× on the local
term. How much that matters depends on how much of M(q) is local inertia versus
the parallel-axis term `m·d²` (which depends only on mass and CoM position, both
of which uniform scaling handles correctly). Measured, by zeroing `body_inertia`
and re-computing M(q):

| pose | rot | link1 | link2 | act |
|---|---|---|---|---|
| `q = 0` | **98.2%** | 33.4% | 31.3% | 39.2% |
| `[0, 0.4, −0.7, 0.3]` | 35.3% | 28.0% | 31.5% | 39.2% |

(share of M(q) coming from local body inertia)

So roughly a third of joint inertia is exposed to the shape error for most
joints — and nearly all of it for `rot` in the folded pose, where the
parallel-axis term nearly vanishes because the mass is stacked on the rotation
axis. **Is uniform density scaling still the right call, or should the inertias
be scaled by a different factor than the masses?** The latter breaks the clean
invariance argument in §5.2, which is why I did not propose it.

### 6.2 The mass *distribution* is preserved, and it is probably also wrong

Uniform scaling keeps the CAD's link-to-link mass ratios. A real arm concentrates
mass in joint motors and leaves link shells nearly empty, so the true per-link
masses are almost certainly not a flat 27.8% of the CAD values. This proposal
fixes the total and the gross gravity loading; it does not fix the ratios. Per-link
scaling factors would be strictly better if per-link masses can be obtained, but
then the gain recipe must be **re-run**, not uniformly scaled.

### 6.3 Does 960 g include the welded base?

- Including base: `s = 0.277676`
- Moving links only: `s = 0.406062`

I assumed the former (whole-robot spec). This has not been confirmed and changes
every number in §5.4.

### 6.4 `act` has thin margin and it is scale-invariant

`kv·dt/M = 1.37` against a limit of 2, in every pose. Scaling does not help or
hurt it — but it means the model is one geometry or timestep change away from
chattering again. Should `act` be re-tuned for margin while the file is being
touched anyway?

### 6.5 `forcerange = 3 × worst gravity torque` has no hardware backing

It is a heuristic. After scaling, the values become 0.28 / 1.94 / 0.56 / 0.083
N·m. Nobody has checked those against the real servos' stall torque. If the real
servos are stronger, the model will saturate — and hit the explicit-integration
regime — far sooner than the hardware does, which would make the jitter
diagnostic misleading in the direction of false alarms.

### 6.6 `v_max = 120 deg/s` is the only hardware datum in the whole derivation

Both `kv` and (through it) `kp` trace back to this single number. If it is wrong
or refers to something else (max joint speed vs max tool speed), the entire gain
set moves.

### 6.7 The gains were never re-derived on the current geometry

A newer CAD export moved link offsets by ~1–4 mm. The owner decided to keep the
gains as-is, which is defensible (masses and inertias were unchanged by that
export), but the measured figures recorded in the XML's own comment came from the
older geometry. At least one has drifted: the comment claims `kv·dt/M = 0.37` for
`link2`; it currently measures 0.25.

### 6.8 No validation against hardware exists at any point

Every number in this document is model-internal. Nothing has been compared
against a real measurement of the real arm — not gravity torque, not settling
behaviour, not the accelerometer channels. The 960 g figure is the first real
datum to enter the model, which is exactly why getting the propagation right
matters.

---

## 7. What I want from the review

1. **§6.1 is the crux.** Is uniform density scaling defensible given the
   hollow-shell argument, or does the ~30% (98% for `rot`) local-inertia exposure
   make it unacceptable? If unacceptable, what is the better estimator given that
   only the *total* mass is known?
2. Is the invariance argument in §5.2 actually complete? I claim `kv·dt/M`, `ω_n`
   and droop are all invariant under a single global `s`. Is there a dimensionless
   quantity I have missed that is *not* invariant — contact stiffness, solver
   tolerances, `armature`, anything scale-dependent I have not considered?
   (The model currently has no `armature` and no contacts.)
3. Is `mj_setTotalmass` leaving `body_subtreemass` stale a real MuJoCo behaviour I
   have correctly diagnosed, or am I misreading an API that expects a recompile
   step I failed to call?
4. Is there a reason to prefer scaling `density` in `<geom>`/`<compiler>` and
   letting MuJoCo recompute inertia from the meshes, instead of hand-editing
   `mass` and `fullinertia`? That path would use mesh volume rather than the
   CAD-supplied tensors — I did not take it because the exporter's `fullinertia`
   values encode the real part geometry (ribs, bosses) that the visual hull does
   not, but I am not certain that is the right trade.
5. Anything in §3 that looks wrong independent of the scaling question.
