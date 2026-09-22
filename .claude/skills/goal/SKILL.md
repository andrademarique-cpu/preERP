---
name: goal
description: Build and run the interactive estimator demo for this repo — a live MuJoCo scene you fly the MyPalletizer260 around in from the keyboard, with the blind EKF's estimate drawn over it as a translucent ghost, beside real-time pyqtgraph strips of the noisy IMU measurements, the estimated state and the true state. Use this whenever the user asks to see, watch, drive, jog, teleoperate or "play with" the arm, to see the filter against truth, to see the estimate diverge, or for "the demo". Also use it for any change to that demo's plotting, keyboard control, ghost overlay or live sensor. Prefer it over ad-hoc implementation, because the golden-run, import-direction and mujoco-boundary guarantees this repo depends on are silent when broken.
---

# /goal — the interactive estimator demo

One window you drive the arm in, and a set of live strips that show whether the
filter is keeping up.

The thesis, and the reason this is worth building at all: **the EKF is blind to
the control by construction.** It never sees your keypress. It estimates the
servo activations from two noisy IMUs alone. So when you jog a joint and the
ghost lags, then catches up, you are watching the filter *infer* a command it
was never given — which is the one claim this project makes that a plot of a
sine sweep cannot show.

## What "done" looks like

Four things, in one process:

1. **A MuJoCo scene of the plant, driven from the keyboard** — all three
   commandable joints, jogged live.
2. **The EKF's estimate of the arm, drawn in the same scene as a translucent
   ghost.** Truth solid, estimate ghosted, error visible as separation.
3. **Live strips of the noisy IMU measurements** against what the plant actually
   read.
4. **Live strips of estimated state vs true state**, with the update instants
   marked, because there are ~25 predicts between them and that ratio is the
   current accuracy limit.

Plus the two things this repo requires of anything that claims to be evidence: a
**toggle that makes it visibly fail**, and a printed sentence saying what the
demo does *not* show.

## Arguments

| Invocation | Does |
|---|---|
| `/goal` | Build whatever is missing, then run it |
| `/goal live` | Just run it — no rebuild |
| `/goal degrade <mode>` | Run the falsification (`swap`, `overconfident`) |
| `/goal check` | The five verification commands, nothing else |

## 1. Orient

Read before touching anything:

1. `references/teleop.md` — the architecture, the verified API mechanics, and
   the traps. **Everything in it was checked against this machine**, including
   two that look like nothing and cost an afternoon each.
2. `references/invariants.md` — what a change here must not break.
3. `scripts/demo_three_way.py` — the closest precedent, P8's acceptance
   artifact. The new work is its interactive sibling and shares its framing,
   its constants and its honesty about what a simulated sensor proves.
4. `CLAUDE.md` — conventions, and "Current repo state".

**Confirm the tree is green before you start** (§ 5). You cannot tell what you
broke from a baseline that was already broken, and a red golden check usually
means the data moved rather than the code.

## 2. The architecture — decided, do not re-litigate

These were settled with the user and by testing the APIs on this machine. If one
turns out to be wrong, say so and why; do not quietly pick a different one.

```
   keyboard ──► ctrl setpoints ──► PLANT model (un-edited)  ──► truth
                                        │                        │
                                        │  live sampling         │  solid
                                        ▼  @20 Hz + N(0,R)       │  render
                                   LiveSimSensor ──► Measurement │
                                        │                        │
                                        ▼                        ▼
                                   FilterRunner ──► blind EKF ──► GHOST render
                                   (advance_to_safe)  (never sees ctrl)
```

- **One window, ghost overlay.** The estimate is drawn into `viewer.user_scn` as
  translucent geoms. Not two windows: MuJoCo supports one passive viewer per
  process, and a second process needs state IPC that drifts.
- **Qt owns the main loop.** `launch_passive` does not block — it returns a
  handle whose `.sync()` you call yourself. A `QTimer` at ~60 Hz steps physics,
  syncs the viewer and updates the curves. Two GUI event loops in one thread is
  the failure this avoids.
- **The keyboard jogs `ctrl`, never `qpos`.** Writing `qpos` directly breaks the
  tendon equality, and the state that produces reads **345 m/s²** on
  `link2_acc_x` against ~9.8 — you would be demoing a broken model.
- **The entry point is a script**, `scripts/demo_teleop.py`, beside
  `demo_three_way.py`, with `argparse` and `__main__`. **Spanish**, like its one
  sibling in `scripts/`.
- **pyqtgraph, not matplotlib.** Already installed here (0.14.0, with PyQt5).
  matplotlib cannot hold 30 Hz on a growing strip; `pyproject.toml` still
  carries the comment about pyqtgraph having been a dependency before.

## 3. What has to be built

Nothing below exists yet. Check before writing — some of it may have landed
since this was written.

| Piece | Where | Language | Why there |
|---|---|---|---|
| Live virtual IMU | `erp/sensors/live.py` | English | Hardware-shaped input enters through `Sensor`, always |
| Live strip widgets | `erp/viz/live.py` | English | Beside `figures.py`; pyqtgraph is a second backend, not a replacement |
| Ghost scene geoms | `erp/viz/ghost.py` | English | See the boundary note below |
| The loop + keymap | `scripts/demo_teleop.py` | **Spanish** | Matches `scripts/`' only sibling |

**`SimSensor` cannot do this job, and that is the largest single piece of work.**
It is a `ReplaySensor` over a *precomputed* array whose noise is drawn once up
front. Teleoperation has no precomputed array — the truth is generated as you
press keys. You need a new `Sensor` subclass that samples a live `MjData` at
~20 Hz, adds `N(0, R)` noise and applies latency. Per this repo's test
convention it goes into `FACTORIES` in `test_sensor_contract.py` and is covered
by the one parametrised contract suite, not by bespoke tests.

**The mujoco-boundary note, which needs a decision recorded.** `CLAUDE.md` lists
*exactly three* modules that may touch the mujoco API and `erp/viz/ghost.py`
would be a fourth. The rule's stated purpose is that no `Any` leaks into the
estimator under strict mypy, and render-only code cannot do that — but the list
is explicit, so **update it in `CLAUDE.md` with the reason** rather than letting
the tree quietly contradict the file. Two constraints come with it:

- Wrap every returned mujoco value in `np.asarray(..., dtype=np.float64)` or
  `int(...)`, the same discipline `erp/sim/mujoco.py` keeps.
- **Withhold the mujoco- and Qt-importing modules from `erp/viz/__init__.py`**,
  which today re-exports only `geometry` so `import erp.viz` stays numpy-only
  and CI can import it without `[viz]`. Same shape as `erp/io/__init__.py`
  withholding `config`.

**Dependencies.** Add a `[live]` extra (`pyqtgraph`, `PyQt5`) — not into
`[viz]`, which is matplotlib. Add a mypy override for `pyqtgraph`, which ships
no stubs, alongside the existing ones. Declare it in `pyproject.toml` only;
`environment.yml` installs `-e .[dev,viz]` and does not carry package lists.

## 4. Make it able to fail

A demo that only ever looks good cannot distinguish a stack that works from one
that works by accident. Carry the `--degrade` toggles from
`scripts/demo_three_way.py`, where their real effects are already measured:

| Mode | Effect | Measured |
|---|---|---|
| `swap` | link2's reading into link1's slot, `rows` unchanged | NIS 74 725 |
| `overconfident` | declared `R` 100× tight, noise still honest | NIS 65 530 |
| `zeros` | start from the keyframe, not the servoed equilibrium | **NIS unchanged** |

**Be precise about `zeros`.** It is a 113-σ bias in `h(x₀)` that does *not*
survive the run: the tendon equality pulls the state back in ~2 physics steps,
before the first measurement arrives at 5 ms. Saying it "wrecks the filter"
would be the overstatement this repo's documents avoid everywhere else. The two
that genuinely degrade are `swap` and `overconfident`.

In the interactive demo the falsification is better than a number: under `swap`
the **ghost visibly detaches** from the truth and stays detached. That is the
shot worth having.

## 5. Verify

```bash
conda run -n EKF ruff check software/src software/tests
conda run -n EKF mypy software/src
grep -rEl "^\s*(from|import)\s+erp\.(sensors)" software/src/erp/estimators software/src/erp/models
conda run -n EKF python -m pytest -q
conda run -n EKF python scripts/make_golden_run.py --check
```

The environment is **`EKF`**, 3.11, with the package installed editable. There
is no `erp` env on this machine. The grep must print nothing. The golden check
must print `OK: reproduce golden_ekf_run.npz a rtol=1e-12`.

Then, and this is not a CI step:

```bash
git status --short data/processed/ data/raw/
```

**Must be empty.** A green golden check against a fixture you regenerated proves
nothing.

This work is additive and should not move a digit of the golden run. **If it
does, that is a real finding, not a tolerance problem** — you touched the
estimation path without meaning to. Do not regenerate the fixture.

`scripts/` is not linted by CI, so run `ruff` over the new script by hand before
committing; `demo_three_way.py` had `B905` violations that had never been
checked for exactly this reason.

**The live loop gets no tests, deliberately** — it needs a display and it
blocks. What does get tested: the new `Sensor` through the contract suite, the
strip decimation arithmetic, and the ghost geom construction (which is pure
array work and runs headless — it is how the `dataid` trap in
`references/teleop.md` was found). Say in the script's docstring that the loop
is untested and why.

## 6. Record it

- **Docstrings state units and reference frames** for every physical quantity,
  and record *why* a choice is load-bearing and what the failure looks like.
  Match the density of `core/`, `io/` and `sensors/`.
- **`CLAUDE.md`**: the directory map (new modules, which language), the
  three-modules mujoco list, the `[live]` extra, and the test count.
- **Measured numbers are never deleted.** Anything you measure building this —
  frame budget, dropped samples, the NIS you see while jogging — goes in a
  docstring or the ADR, not in a commit message that scrolls away.
- **Commit** as `feat:`, ending with:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

## 7. Stop and ask

- **Never run `notebooks/mypalletizer260EKF.ipynb`.** `SEND_TO_ROBOT = True` and
  `RUN_ROBOT_WITH_IMU = True` are committed on `main` with `COM6`/`COM7`
  hard-coded. Running it top to bottom opens both serial ports and drives real
  hardware. Read it, edit it, do not execute it.
- The golden run moves and you believe it should.
- The work drifts toward a stated non-goal — grasp planning, manipulation
  policy, learning-based control, or hard real-time guarantees in Python.
- Driving the **real** arm from the keyboard comes up. That is a different and
  much more dangerous demo: this one is simulation only, and nothing here has
  ever run against the Teensy.
