---
name: goal
description: Drive the ADR-0002 notebook-to-package migration forward in this repo — pick the next unblocked phase, build it under the project's invariants, record the decisions it forced, verify nothing moved, commit and push to main, then repeat until no phase remains. Also builds the three-way demo (plant truth vs noisy virtual sensors vs EKF estimate, live MuJoCo viewer plus live plots). Use this whenever the user asks to advance, continue, finish or work on a phase, a milestone, M1/M2/M3, P7.5, P8, "the migration", "the roadmap", or the ADR — and whenever they ask to see the estimator against truth, to "watch" the arm, or for the demo. Prefer it over ad-hoc implementation for any change to the estimation path, because the golden-run and import-direction guarantees are easy to break and expensive to notice.
---

# /goal — finish the migration, one phase at a time

`docs/adr/0002-notebook-to-package.md` is the plan and the record. This skill
runs its remaining phases to completion: each one built, annotated, verified,
committed and pushed before the next begins.

The reason it is a skill rather than a habit: this codebase's guarantees are
*silent* when broken. A filter with a wrongly-wired sensor still runs. An
estimator that reaches into `sensors/` still passes CI's grep if the import is
relative. A refactor that moves the numbers still produces plots. Every check
below exists because something once looked fine and was not.

## Arguments

| Invocation | Does |
|---|---|
| `/goal` | Every remaining phase, in dependency order, until none is left |
| `/goal P8` | That phase alone |
| `/goal demo` | The three-way demo artifact alone (see `references/demo.md`) |

## 1. Orient — from the ADR, not from memory

Read, in this order, before touching anything:

1. `docs/adr/0002-notebook-to-package.md` § 5.1 (the phase graph — which phases
   are `DONE`, what each remaining one depends on), § 5.2 (the phase entries,
   including every finished phase's *What landed* record), and § 6 (the test
   obligations table, which is the authority on whether a phase is done).
2. `CLAUDE.md` — the conventions, and the "Current repo state" section.
3. `git log --oneline -8` and `git status`.

**§ 6's table and § 5.1's graph are the authority on what is done.** `CLAUDE.md`
summarises the roadmap and goes stale on its own — it has been wrong about which
phase is next before. When the two disagree, believe the ADR and fix CLAUDE.md
as part of the phase.

Pick the next phase whose dependencies in § 5.1 are all `DONE`. If several are
unblocked, prefer the one the ADR calls out as closing a milestone. Say which
one you picked and why before starting.

**Before any code:** confirm the tree is green. Run the five checks in § 4. If
they are red *before* you start, stop and say so — you cannot tell what you
broke from a baseline that was already broken, and a red golden check usually
means the data moved, not the code (`CLAUDE.md`, "If the golden check goes red").

## 2. Build it

Read `references/invariants.md` now. It is the list of properties a change here
must not break, each with the command that checks it and a note on what the
failure looks like. Most are not enforced by CI.

The four that catch people most often:

- **Hardware enters through `Sensor`, nowhere else.** CI greps only
  `estimators/` and `models/` for *absolute* `erp.sensors` imports. The rule is
  broader than its enforcement: relative imports, `sim/`, `core/`, `io/`,
  `fusion/`, and reach-through via a package `__init__` all pass CI and all
  violate it. Verify by reading the import block.
- **The golden run does not move.** `rtol=1e-12` over the frozen end-to-end run.
  A refactor that changes a digit is not a refactor. Regenerating the fixture to
  make a test pass is never the fix — see § 4.
- **Language follows the module.** English: `core/`, `io/`, `sensors/`,
  `robot/`, `fusion/`, `calibration/`, `trajectory.py`, all tests, `docs/adr/`.
  Spanish: `sim/`, `estimators/`, `models/`, the notebooks, `docs/theory/`.
  Match the file you are editing. If you add a package, say in CLAUDE.md's
  directory map which side it is on.
- **Pair every diagnostic with a variant that fails it.** `test_clock.py` is the
  model: the sync passes a 1 ms bound *and* arrival stamping is asserted to miss
  it. A test that cannot fail is not evidence. Be precise about how the broken
  variant fails — overstating a mild failure is the fastest way to get the
  reasoning disbelieved later.

Follow § 5.3's migration discipline when moving code out of the notebook: move
it **unchanged** first, including its Spanish docstring and its measured
constants; import it in the notebook and delete the cell; diff against the
golden fixture; only then clean it up. Do not refactor and relocate in the same
commit.

**Never execute `notebooks/mypalletizer260EKF.ipynb`.** `SEND_TO_ROBOT = True`
and `RUN_ROBOT_WITH_IMU = True` are committed on `main` with `COM6`/`COM7`
hard-coded, so running it top to bottom opens both serial ports and drives real
hardware. Read it, edit its cells, but do not run it.

## 3. Annotate the decisions — this is half the phase

A phase that lands working code and no record is half done. The value that
accumulates here is the reasoning, because the numbers are expensive to
re-derive and the wrong turns are invisible once the code looks tidy.

For each phase, write:

**In ADR-0002 § 5.2's entry for the phase:**

- Mark it `DONE`, or `DONE, reshaped` if the plan changed. **Keep the original
  text and strike it through** (`~~like this~~) rather than rewriting it. The
  disagreement between what was planned and what was built is the most useful
  thing in the document — P3 and P7 both read that way on purpose.
- A *What landed* paragraph: what exists now, module by module.
- Where the plan turned out to be wrong, say so plainly and say why the change
  was made. If you rejected an alternative, name it and give the reason.
- What you deliberately did **not** do, and what defers it. A reader needs to
  tell "not yet" from "decided against".

**In ADR-0002 § 6's table:** the phase's row, with the falsification the phase
actually supplied — not the one that was planned, if they differ.

**In the code:** docstrings state units and reference frames for every physical
quantity, and record *why* a choice is load-bearing and what the failure looks
like when it is not honoured. Match the density of `core/`, `io/` and
`sensors/`; bare descriptions are below this codebase's bar.

**Measured numbers are never deleted.** If a constant moves, its measurement and
its justification move with it, into the new home or into the ADR. If you strip
something from a config or a notebook, relocate the prose — ADR-0001 appendix A
is the precedent for finger-era values.

**In CLAUDE.md:** correct whatever the phase made false — test counts, the
directory map, the "Known open items" list, the remaining-phases summary. Strike
through closed items rather than deleting them, so the history stays readable.

**In the commit message:** the same record, compressed. What landed, what was
reshaped and why, what was deliberately deferred, and the check results.

## 4. Verify — all five, in this order

CI runs the first four, in exactly this order (`.github/workflows/ci.yml`). The
import-direction grep runs **before** pytest, not after.

```bash
conda run -n erp ruff check software/src software/tests
conda run -n erp mypy software/src
grep -rEl "^\s*(from|import)\s+erp\.(sensors)" software/src/erp/estimators software/src/erp/models
conda run -n erp python -m pytest -q
conda run -n erp python scripts/make_golden_run.py --check
```

The environment is **`erp`** — 3.11, with the package installed editable against
this checkout. (`base` is 3.14 and does not have it. Older notes in CLAUDE.md
name an `EKF` env; it does not exist on this machine.) The grep must print
nothing. The golden check must print `OK: reproduce golden_ekf_run.npz a
rtol=1e-12`.

Then one more, which is not a CI step and matters most:

```bash
git status --short data/processed/ data/raw/
```

**Both must be empty.** A green golden check against a fixture you regenerated
proves nothing. If the check fails, the question is always "did the code move
the numbers, or did the data change underneath?" — answer it by running
`run_pipeline(root_dir=...)` against a shadow root holding the previously
committed log before you touch the estimator. `CLAUDE.md`'s golden-check section
has the recipe.

Also re-check the broader import rule and the driver-name rule by reading, since
no CI step covers them — `references/invariants.md` has the commands.

## 5. Commit and push

Only when all five checks are green and `data/` is clean.

Conventional commits — `feat:`, `fix:`, `docs:`, `refactor:`, `test:`. Recent
history does not follow this; follow it anyway. End the message with:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

Then `git push` to `main`. One phase, one commit, one push.

**If any check is red, stop.** Do not commit, do not push, do not start the next
phase. Report what failed, with the output. A phase that lands red breaks the
baseline every later phase is measured against.

## 6. Loop

Report what landed in two or three sentences, then go back to § 1 and pick the
next unblocked phase. Stop when no phase remains before M3, or when a phase
needs a decision that is not yours to make.

**Stop and ask** — do not guess — when:

- A phase requires overruling a decision recorded in an ADR. Reshaping a phase
  is normal and is recorded; contradicting § 4.8's construction guarantee or the
  import direction is not.
- The golden run moves and you believe it *should*. Regenerating the fixture is
  a deliberate act that needs saying out loud, and it invalidates every figure
  quoted from it.
- The work drifts toward a stated non-goal: grasp planning, manipulation policy,
  learning-based control, or hard real-time guarantees in the Python layer.
  Flag it rather than implementing it.
- A phase would need hardware attached to verify.

## The demo

P8's acceptance artifact, and the thing to build when the user asks to *watch*
the estimator: the arm moving in a live MuJoCo viewer beside live plots of plant
truth, the noisy virtual sensor, and the EKF's estimate of the same channel.

Read `references/demo.md` before building it. It has the three-signal
definition, the traps specific to this model (which model produces truth, why
the filter must start from the servoed equilibrium, why `advance_to_safe`), and
the honest framing — a `SimSensor` draws its noise from the same `R` the filter
is given and the plant *is* the model, so a good-looking demo is a plumbing
check, not evidence about the estimator.
