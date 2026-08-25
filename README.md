# ERP — Enhanced Robotic Perception

Multi-sensor state estimation stack for robotic hands, fusing inertial
measurements with contact force sensing through Kalman filter variants
(KF / EKF / UKF).

## 1. Purpose

The project spans three coupled tracks:

| Track | Deliverable |
|---|---|
| Software | Installable Python package `erp` implementing the estimation stack |
| Mechanical | Sensorized finger/hand assembly hosting the sensor suite |
| Electronics & firmware | Acquisition chain with deterministic timestamping |

The intended outcome is a system that estimates finger pose, velocity, and
contact state more reliably than either sensing modality alone —
particularly during contact transitions, where proprioception is weakest.

## 2. Objectives

**Primary**

- Implement KF, EKF, and UKF against a common `StateEstimator` interface,
  so estimators are interchangeable without touching calling code.
- Fuse accelerometer and contact force measurements running at different
  rates and latencies into a single consistent state estimate.
- Validate estimator consistency quantitatively (NEES / NIS) on both
  simulated and recorded data.

**Secondary**

- Design and manufacture a sensorized hand assembly that physically hosts
  the sensor suite.
- Provide a ROS 2 integration layer that wraps — never replaces — the
  core package.
- Produce reproducible datasets and calibration procedures under version
  control.

**Explicit non-goals**

- Grasp planning, manipulation policy, or learning-based control. ERP
  produces state estimates; downstream consumers are out of scope.
- Real-time guarantees below ~1 kHz. Firmware handles hard timing; the
  Python layer does not.

## 3. Architectural principle

> The mathematical core does not know that hardware exists.

`estimators/` and `models/` must never import from `sensors/`,
`firmware/`, or ROS 2. This is not stylistic — it is what allows every
filter to run in a notebook, in CI, and on recorded data with no hardware
attached.

Consequences that follow directly:

- Estimators are testable without sensors.
- The measurement backend (real driver, replayed dataset, simulator) is
  swappable via the `Sensor` adapter.
- ROS 2 is a thin wrapper over `erp`, not a dependency of it.

Any pull request that violates the import direction is rejected
regardless of whether it works. This is enforced by review and by the
import-direction check in CI.

## 4. Repository structure

```
.
├── README.md
├── CLAUDE.md
├── pyproject.toml            # the only packaging file; install from HERE
├── environment.yml           # conda path; delegates deps to pyproject.toml
├── .gitattributes            # Git LFS: CAD, STEP, STL, datasets
├── CODEOWNERS
├── .github/workflows/ci.yml
│
├── docs/
│   ├── theory/                # filter derivations, process/measurement models
│   ├── hardware/               # assembly, wiring, calibration procedures
│   └── adr/                    # architecture decision records
│
├── mechanical/
│   ├── cad/                    # native source files
│   ├── step/                   # neutral exchange format
│   ├── stl/                    # printable geometry
│   ├── drawings/                # dimensioned PDFs with GD&T
│   └── bom/bom.csv
│
├── electronics/
│   ├── schematics/
│   ├── pcb/
│   └── datasheets/
│
├── firmware/                   # MCU sampling + timestamping
│
├── software/
│   ├── src/erp/
│   │   ├── core/                # State, Measurement, base types
│   │   ├── models/              # process + measurement models
│   │   ├── estimators/          # KF, EKF, UKF
│   │   ├── sensors/             # Sensor interface + drivers + replay
│   │   ├── calibration/
│   │   ├── fusion/              # multi-rate synchronization
│   │   ├── io/                  # logging, dataset replay
│   │   └── viz/
│   └── tests/
│
├── ros2_ws/src/erp_ros/        # nodes, msgs, launch files only
├── config/                      # YAML: noise params, geometry, calibration
├── data/                        # raw/ processed/ (LFS or DVC)
├── notebooks/                   # exploratory work (Kalman/UKF derivations, MuJoCo practice)
└── scripts/
```

## 5. Class hierarchy

Four abstract base classes define the entire contract. Everything else
derives from them.

> **The signatures below are the pre-freeze sketch and are out of date.**
> [ADR-0001](./docs/adr/0001-multi-rate-fusion.md) D1 amended all four
> before the freeze: `Q` became `Q(dt)`, `h`/`jacobian` take `u`,
> `update` takes `u`, and `Sensor` gained `drain()` with a non-blocking
> `read() -> Measurement | None`. Read the source files for the current
> contract; each change has a measured justification in ADR-0001 § 2.

```python
# core/types.py
@dataclass(frozen=True)
class Measurement:
    z: np.ndarray
    timestamp: float
    frame_id: str
    R: np.ndarray | None = None      # per-sample noise override

@dataclass
class GaussianState:
    x: np.ndarray
    P: np.ndarray

# models/base.py
class ProcessModel(ABC):
    """Discretization of dx/dt = f(x, u)."""
    def predict(self, x, u, dt) -> np.ndarray: ...
    def jacobian(self, x, u, dt) -> np.ndarray: ...   # EKF only
    @property
    def Q(self) -> np.ndarray: ...

class MeasurementModel(ABC):
    def h(self, x) -> np.ndarray: ...
    def jacobian(self, x) -> np.ndarray: ...
    @property
    def R(self) -> np.ndarray: ...

# estimators/base.py
class StateEstimator(ABC):
    def __init__(self, process_model: ProcessModel, initial: GaussianState): ...
    def predict(self, u, dt) -> None: ...
    def update(self, m: Measurement, model: MeasurementModel) -> None: ...

# sensors/base.py
class Sensor(ABC):
    def read(self) -> Measurement: ...
    def calibrate(self) -> CalibrationResult: ...
    @property
    def measurement_model(self) -> MeasurementModel: ...
```

### Concrete implementations

| Base class | Derived classes |
|---|---|
| `ProcessModel` | `ConstantAccelModel`, `FingerKinematicModel`, `ContactDynamicsModel` |
| `MeasurementModel` | `AccelerometerModel`, `ContactForceModel`, `JointEncoderModel` |
| `StateEstimator` | `KalmanFilter`, `ExtendedKalmanFilter`, `UnscentedKalmanFilter` |
| `Sensor` | `MembraneForceSensor`, `IMUSensor`, `EncoderSensor`, `ReplaySensor`, `SimulatedSensor` |

### Design patterns in use

- **Strategy** — sigma point generation is injected into the UKF rather
  than hardcoded:
  `UnscentedKalmanFilter(model, sigma_points=MerweScaledSigmaPoints(alpha, beta, kappa))`.
  Parametrizations can be compared without modifying the filter.
- **Adapter** — `ReplaySensor` reads recorded data and exposes the
  identical `Sensor` interface. Offline validation therefore exercises
  the same code path as the robot.

## 6. Multi-rate fusion

The accelerometer runs near 1 kHz; contact force sensing runs roughly an
order of magnitude slower, with different latency. This asymmetry is
isolated in one class:

```python
class FusionEngine:
    def __init__(self, estimator: StateEstimator,
                 sensors: dict[str, Sensor],
                 buffer_horizon: float = 0.05): ...

    def step(self, t_now: float) -> GaussianState:
        # 1. sort buffered measurements by timestamp
        # 2. predict forward to each measurement timestamp
        # 3. update using that sensor's measurement model
```

**Rule:** `predict` advances to the measurement timestamp, never by a
fixed `dt`. Fixed-step prediction with out-of-order arrivals corrupts the
covariance, and the failure is silent — the estimate looks plausible
while consistency metrics degrade.

## 7. Team structure and workflow

### Module ownership

Ownership is by directory, not by task. This is what keeps merge
conflicts near zero.

| Owner | Directories |
|---|---|
| A | `software/src/erp/estimators/`, `models/` |
| B | `software/src/erp/sensors/`, `firmware/`, `calibration/` |
| C | `mechanical/`, `electronics/` |
| D | `software/src/erp/fusion/`, `io/`, `viz/`, CI |

Encoded in `CODEOWNERS` so GitHub assigns reviewers automatically.

### Sprint 0 — the contract

Week 1, whole team, before any implementation work begins. Merge to `main`:

- The four ABCs: signatures, docstrings stating units and reference
  frames, `raise NotImplementedError` bodies.
- A trivial stub per ABC (`DummySensor`, `IdentityProcessModel`).

The stubs are what unblock parallel work: the UKF author does not wait
for force sensors, and the driver author does not wait for the filter.

**Hard rule:** changing a merged ABC signature requires a PR approved by
every owner. This is the only genuine source of cross-team breakage.

### Git conventions

- `main` is protected — no direct pushes (Settings → Branches).
- Short-lived branches: `feat/ukf-sigma-points`, `fix/imu-bias`. Maximum
  lifetime ~5 days.
- Every change lands via PR: ≥1 approval, CI green.
- Daily `git pull --rebase origin main`.
- Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`.
- Git LFS configured before the first commit for `.SLDPRT`, `.STEP`,
  `.stl`, and datasets. Retrofitting LFS is painful.
- Hardware revisions tagged independently (`hw-v1.2`) from software
  (`v0.3.0`), with a compatibility table in this README.

### Definition of Done

A PR merges only when all of the following hold:

- [ ] Tests pass, including filter consistency checks (NEES/NIS on
      simulated data)
- [ ] Type hints on all public signatures
- [ ] Docstrings state units and reference frames
- [ ] No import from another owner's module across the core/hardware
      boundary
- [ ] New estimators or models registered in the relevant `config/` YAML

### Onboarding sessions

Three 90-minute sessions before Sprint 1. Skill level should not be
assumed — one force-push to `main` costs more than these sessions save.

1. **Applied OOP** — ABC vs. concrete inheritance, composition over
   inheritance, and why `estimators/` imports nothing from `sensors/`.
2. **Git for teams** — rebase vs. merge, conflict resolution, and
   recovery (`git reflog`).
3. **Testing as contract** — write tests against the ABC, not the
   implementation. A test that passes with both the stub and the real
   driver proves the interface is sound.

### Hardware / software compatibility

| Hardware tag | Compatible software versions | Notes |
|---|---|---|
| `hw-v1.0` | — | not yet cut |

## 8. Known risks

| Risk | Mitigation |
|---|---|
| Interface churn after Sprint 0 stalls everyone | Unanimous approval required for ABC changes; stubs absorb early uncertainty |
| Core module imports hardware "just to fix something quickly" | Enforced in review; CI import-direction check |
| Mechanical and software revisions drift out of sync | Independent tags plus a README compatibility table |
| Filter tuned to look good rather than be consistent | NEES/NIS in CI, not visual inspection of trajectory plots |
| Multi-rate timestamp handling done ad hoc per sensor | All of it confined to `FusionEngine` |

## 9. Immediate next steps

- [x] Initialize the repository with `.gitattributes` and LFS first.
- [x] Run Sprint 0 to freeze the four ABCs plus stubs. Signatures were
      amended first by [ADR-0001](./docs/adr/0001-multi-rate-fusion.md)
      D1; they are frozen as of that ADR.
- [x] Stand up CI: pytest, type checking, import-direction check. The
      install step was broken from the start (`-e ./software`), so the
      job failed before reaching any check — fixed, and now runs on a
      3.10/3.11 matrix.
- [x] Establish the simulated ground-truth scenario used to validate
      every estimator — `software/tests/test_consistency.py`, multi-rate
      NEES/NIS with falsification variants that must fail.
- [ ] Assign module ownership and commit real handles to `CODEOWNERS`
      (currently placeholders).
- [ ] Re-derive `psd_alpha` from the physics. The value in
      `config/estimation.yaml` is a placeholder, and it blocks these
      filters meaning anything on real data.
- [ ] Measure the actuation transport delay `d_act` on the real chain.
- [ ] Implement the UKF and the first real sensor drivers;
      `calibration/`, `io/` and `ros2_ws/` are still empty.

## Getting started

**Install from the repository root.** `pyproject.toml` lives here and
already points setuptools at `software/src`; there is no build file under
`software/`, so the `-e ./software` form that appeared in earlier
revisions of this file could never work.

```bash
git lfs install

# pip
pip install -e ".[dev]"          # core + pytest/mypy/ruff
conda activate erp               # or: conda env create -f environment.yml && conda activate erp

pytest -q                        # 63 tests, ~7 s
```

### Extras

Dependencies are declared once, in `pyproject.toml`. `environment.yml`
only supplies the interpreter and pip, then installs this package — so
there is no second list to keep in sync.

| Extra | Pulls in | You need it for |
|---|---|---|
| *(none)* | numpy, scipy | importing `erp`, running estimators |
| `dev` | pytest, mypy, ruff | the checks below |
| `viz` | matplotlib | plotting in notebooks |
| `app` | mujoco, pyqtgraph, PyQt5, pyserial | `scripts/finger_viewer.py` only |

`conda env create` installs `[dev,viz]`. The `app` stack is deliberately
left out — the estimation core and its tests need none of it, and that is
the point of § 3. Add it only if you intend to run the viewer:

```bash
pip install -e ".[app]"
python scripts/finger_viewer.py            # synthetic input, no hardware
python scripts/finger_viewer.py --pot --port COM5   # real potentiometers
```

### Checks

These are exactly what CI runs, in order:

```bash
ruff check software/src software/tests
mypy software/src
pytest -q
```

Python 3.10 is the supported floor — what `requires-python`, mypy and
ruff all target, and what CI tests alongside 3.11. The conda environment
is 3.11, so it will happily run 3.11-only syntax that CI then rejects;
mypy catches this locally.

See [`CLAUDE.md`](./CLAUDE.md) for the conventions an AI coding assistant
(or a new contributor) should follow when working in this repository, and
[`docs/adr/0001-multi-rate-fusion.md`](./docs/adr/0001-multi-rate-fusion.md)
for the design decisions the package is built on.
