"""FusionEngine: event-driven prediction, ordering, and late-arrival accounting."""

from __future__ import annotations

import numpy as np
import pytest

from erp.core.timeline import InputHistory
from erp.core.types import CalibrationResult, ControlInput, GaussianState, Measurement
from erp.estimators.ekf import ExtendedKalmanFilter
from erp.fusion.engine import FusionEngine
from erp.models.base import MeasurementModel
from erp.models.linear import JointParams, servoed_finger_model
from erp.models.measurement import joint_block_model
from erp.sensors.base import Sensor
from erp.sensors.replay import DummySensor, ReplaySensor

JOINTS = [
    JointParams(kp=2.0, kv=0.025, damping=0.003, inertia=4e-5, tau=0.004, psd_alpha=1e-2),
    JointParams(kp=0.8, kv=0.010, damping=0.0028, inertia=2e-5, tau=0.004, psd_alpha=1e-2),
]
NX = 6


def make_filter() -> ExtendedKalmanFilter:
    initial = GaussianState(np.zeros(NX), np.diag([1e-4] * 2 + [1e-2] * 2 + [1e-4] * 2))
    return ExtendedKalmanFilter(servoed_finger_model(JOINTS), initial)


def make_inputs(pairs: list[tuple[float, list[float]]]) -> InputHistory:
    h = InputHistory()
    for t, u in pairs:
        h.push(ControlInput(u=np.array(u, dtype=float), timestamp=t))
    return h


ENCODER = joint_block_model(2, "angle", sigma=1.7e-3)


class LateSensor(Sensor):
    """Stub that withholds its sample until after the filter has moved past it.

    Reproduces live transport behaviour that replay cannot: a measurement whose
    timestamp is already in the filter's past by the time it arrives.
    """

    def __init__(self, m: Measurement, model: MeasurementModel, release_on_call: int) -> None:
        self._m: Measurement | None = m
        self._model = model
        self._calls = 0
        self._release_on = release_on_call

    def read(self) -> Measurement | None:
        self._calls += 1
        if self._calls >= self._release_on and self._m is not None:
            m, self._m = self._m, None
            return m
        return None

    def drain(self) -> list[Measurement]:
        m = self.read()
        return [] if m is None else [m]

    def calibrate(self) -> CalibrationResult:
        return CalibrationResult(np.zeros(2), np.ones(2), np.zeros((2, 2)), True, "stub")

    @property
    def measurement_model(self) -> MeasurementModel:
        return self._model


def test_prediction_splits_at_input_breakpoints() -> None:
    """The engine must integrate each constant-u segment separately.

    Compared against the same filter driven by hand, and against the wrong
    answer you get by holding the stale input across the whole interval.
    """
    z = np.array([0.01, -0.02])
    meas = Measurement(z=z, timestamp=0.010, frame_id="joint")
    inputs = make_inputs([(0.0, [0.0, 0.0]), (0.005, [1.0, 0.5])])

    engine = FusionEngine(
        make_filter(), {"enc": ReplaySensor([meas], ENCODER)}, inputs, buffer_horizon=0.0
    )
    engine.step(0.010)

    by_hand = make_filter()
    by_hand.predict(np.array([0.0, 0.0]), 0.005)
    by_hand.predict(np.array([1.0, 0.5]), 0.005)
    by_hand.update(meas, ENCODER, np.array([1.0, 0.5]))

    assert engine.time == pytest.approx(0.010)
    assert np.allclose(engine.state.x, by_hand.state.x, rtol=1e-10, atol=1e-14)
    assert np.allclose(engine.state.P, by_hand.state.P, rtol=1e-9, atol=1e-20)

    unsplit = make_filter()
    unsplit.predict(np.array([0.0, 0.0]), 0.010)
    unsplit.update(meas, ENCODER, np.array([1.0, 0.5]))
    assert not np.allclose(engine.state.x, unsplit.state.x, rtol=1e-6, atol=1e-9)


def test_processes_multi_rate_measurements_in_timestamp_order() -> None:
    """Two sensors on unrelated rates interleave correctly."""
    fast = [Measurement(np.zeros(2), t, "joint") for t in np.arange(0.001, 0.05, 1 / 300)]
    slow = [Measurement(np.zeros(2), t, "joint") for t in np.arange(0.002, 0.05, 1 / 70)]
    inputs = make_inputs([(0.0, [0.0, 0.0]), (0.02, [0.5, 0.5])])

    engine = FusionEngine(
        make_filter(),
        {"fast": ReplaySensor(fast, ENCODER), "slow": ReplaySensor(slow, ENCODER)},
        inputs,
        buffer_horizon=0.0,
    )
    engine.step(0.05)

    assert engine.time == pytest.approx(max(fast[-1].timestamp, slow[-1].timestamp))
    assert engine.discarded == {}
    assert np.all(np.isfinite(engine.state.x))


def test_buffer_horizon_defers_measurements_until_releasable() -> None:
    meas = [Measurement(np.zeros(2), t, "joint") for t in (0.01, 0.02, 0.03)]
    engine = FusionEngine(
        make_filter(),
        {"enc": ReplaySensor(meas, ENCODER)},
        make_inputs([(0.0, [0.0, 0.0])]),
        buffer_horizon=0.015,
    )
    engine.step(0.020)                      # releases only timestamps <= 0.005
    assert engine.time == pytest.approx(0.0)
    assert engine.pending == 3

    engine.step(0.040)                      # releases <= 0.025
    assert engine.time == pytest.approx(0.02)
    assert engine.pending == 1


def test_late_measurements_are_discarded_and_counted() -> None:
    """A silent discard path is how a sensor quietly stops contributing."""
    on_time = [Measurement(np.zeros(2), t, "joint") for t in (0.010, 0.020)]
    stale = Measurement(np.zeros(2), 0.005, "joint")

    engine = FusionEngine(
        make_filter(),
        {"enc": ReplaySensor(on_time, ENCODER), "flaky": LateSensor(stale, ENCODER, 2)},
        make_inputs([(0.0, [0.0, 0.0])]),
        buffer_horizon=0.0,
    )
    engine.step(0.020)
    assert engine.discarded == {}
    assert engine.time == pytest.approx(0.020)

    engine.step(0.030)                      # stale sample surfaces, already in the past
    assert engine.discarded == {"flaky": 1}
    assert engine.time == pytest.approx(0.020)


def test_engine_never_advances_backwards() -> None:
    engine = FusionEngine(
        make_filter(), {}, make_inputs([(0.0, [0.0, 0.0])]), buffer_horizon=0.0, t0=0.05
    )
    with pytest.raises(ValueError, match="cannot advance backwards"):
        engine._advance_to(0.01)


def test_belief_at_current_time_returns_an_unaliased_copy_of_the_belief() -> None:
    """Extrapolating to `time` itself is a no-op, but must not hand out internals."""
    engine = FusionEngine(
        make_filter(), {}, make_inputs([(0.0, [0.0, 0.0])]), buffer_horizon=0.0
    )
    belief = engine.belief_at(engine.time)

    assert np.array_equal(belief.x, engine.state.x)
    assert np.array_equal(belief.P, engine.state.P)

    belief.x[0] = 999.0
    assert engine.state.x[0] != 999.0


def test_belief_at_matches_a_manual_predict() -> None:
    """With no input change in the interval, extrapolation is one predict."""
    engine = FusionEngine(
        make_filter(), {}, make_inputs([(0.0, [0.3, -0.2])]), buffer_horizon=0.0
    )
    belief = engine.belief_at(0.010)

    by_hand = make_filter()
    by_hand.predict(np.array([0.3, -0.2]), 0.010)

    assert np.allclose(belief.x, by_hand.state.x, rtol=1e-12, atol=1e-16)
    assert np.allclose(belief.P, by_hand.state.P, rtol=1e-12, atol=1e-20)


def test_belief_at_splits_at_input_breakpoints() -> None:
    """Falsification: holding a stale u across the whole gap gives a different answer.

    This is the test that fails if `_segments` ever stops splitting -- the same
    guarantee `test_prediction_splits_at_input_breakpoints` makes for the
    mutating path, which now shares that implementation.
    """
    inputs = make_inputs([(0.0, [0.0, 0.0]), (0.005, [1.0, 0.5])])
    engine = FusionEngine(make_filter(), {}, inputs, buffer_horizon=0.0)
    belief = engine.belief_at(0.010)

    by_hand = make_filter()
    by_hand.predict(np.array([0.0, 0.0]), 0.005)
    by_hand.predict(np.array([1.0, 0.5]), 0.005)
    assert np.allclose(belief.x, by_hand.state.x, rtol=1e-12, atol=1e-16)

    unsplit = make_filter()
    unsplit.predict(np.array([0.0, 0.0]), 0.010)
    assert not np.allclose(belief.x, unsplit.state.x, rtol=1e-6, atol=1e-9)


def test_belief_at_does_not_advance_the_filter() -> None:
    """Extrapolation is a read: scoring the filter must not perturb it."""
    meas = [Measurement(np.zeros(2), t, "joint") for t in (0.010, 0.020)]
    engine = FusionEngine(
        make_filter(),
        {"enc": ReplaySensor(meas, ENCODER)},
        make_inputs([(0.0, [0.0, 0.0])]),
        buffer_horizon=0.0,
    )
    engine.step(0.020)
    t_before, x_before, P_before = engine.time, engine.state.x.copy(), engine.state.P.copy()

    engine.belief_at(0.035)

    assert engine.time == t_before
    assert np.array_equal(engine.state.x, x_before)
    assert np.array_equal(engine.state.P, P_before)


def test_belief_at_rejects_times_before_the_filter() -> None:
    engine = FusionEngine(
        make_filter(), {}, make_inputs([(0.0, [0.0, 0.0])]), buffer_horizon=0.0, t0=0.05
    )
    with pytest.raises(ValueError, match="cannot extrapolate backwards"):
        engine.belief_at(0.01)


@pytest.mark.parametrize("build", ["replay", "dummy"])
def test_sensor_contract_holds_for_every_implementation(build: str) -> None:
    """The same interface-level test must pass against a stub and a real source.

    That is what proves the Sensor ABC is sound, rather than proving one
    concrete class happens to work.
    """
    sensor: Sensor
    if build == "replay":
        sensor = ReplaySensor([Measurement(np.zeros(2), 0.01, "joint")], ENCODER)
    else:
        sensor = DummySensor(ENCODER, np.zeros(2), period=0.01)

    drained = sensor.drain()
    assert len(drained) == 1
    assert isinstance(drained[0], Measurement)
    assert drained[0].z.shape == (2,)
    assert sensor.measurement_model is ENCODER
    assert sensor.calibrate().valid

    engine = FusionEngine(
        make_filter(), {"s": sensor}, make_inputs([(0.0, [0.0, 0.0])]), buffer_horizon=0.0
    )
    engine.step(1.0)
    assert np.all(np.isfinite(engine.state.x))
