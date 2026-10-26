"""ADR-0002 P2: the arm contract, the joint map, and the guard that gates a send.

Two things are under test and they fail differently. The `ArmInterface`
contract is plumbing: a broken arm shows up immediately. `JointMap.validate`
is a *guard*, and a guard that cannot fire is worth nothing -- so every
positive case here is paired with the trajectory it must reject, lifted from
the notebook's negative-check cell.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from erp.io.paths import resolve_repo_path
from erp.robot import ArmInterface, CoupledLimit, DryRunArm, JointMap, MyPalletizerArm
from erp.trajectory import sine_sweep

# ---------------------------------------------------------------- fixtures

XML = "mechanical/mujoco_assets/MyPalletizer260/MyPalletizer260.xml"

# The fitted palletizer map. Lives here rather than in the package because
# `erp.io.config` (P7) is what turns config/estimation.yaml into one of these;
# hardcoding it in `erp.robot` now would just be a third copy to keep in step.
API_LIMITS_DEG = {
    1: (-162.0, 162.0),  # rot   / base
    2: (-2.0, 90.0),  # link1 / shoulder
    3: (-92.0, 60.0),  # link2 / elbow
    4: (-180.0, 180.0),  # end effector, not modelled
}


def make_jmap(signs: tuple[float, ...] = (1.0, 1.0, 1.0), **kw: Any) -> JointMap:
    """The real map by default; `signs=` builds the miswired variants."""
    return JointMap(
        names=kw.pop("names", ("rot", "link1", "link2")),
        api_ids=kw.pop("api_ids", (1, 2, 3)),
        signs=np.array(signs, dtype=float),
        offsets_deg=kw.pop("offsets_deg", np.zeros(3)),
        api_limits_deg=kw.pop("api_limits_deg", API_LIMITS_DEG),
        **kw,
    )


def sine_trajectory(
    amps_deg: tuple[float, float, float], period_s: float, dt: float = 0.002
) -> tuple[np.ndarray, np.ndarray]:
    """The notebook's trajectory shape: starts at 0, peaks at 2*amp.

    Was a fourth hand-written copy of the sine formula until P3; now the
    package generator, so a change to the trajectory reaches these tests.
    """
    t, q, _ = sine_sweep(amps_deg, period_s, dt)
    return t, q


@pytest.fixture(scope="module")
def jmap() -> JointMap:
    return make_jmap()


@pytest.fixture(scope="module")
def model() -> Any:
    mj_plant = pytest.importorskip("erp.sim.plant")
    m, _ = mj_plant.load_model(resolve_repo_path(XML))
    return m


class FakeClock:
    """A host clock the test drives. `DryRunArm`'s lag is only testable so."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = float(t)

    def __call__(self) -> float:
        return self.t


_UNSET = object()


class _FakeMyCobotBase:
    """Stands in for pymycobot's MyPalletizer260, which is not installed here."""

    def __init__(self, *, reply: Any = _UNSET) -> None:
        self.sent: list[tuple[str, list[float], int]] = []
        self.reply = [1.0, 2.0, 3.0, 4.0] if reply is _UNSET else reply

    def send_radians(self, rad: list[float], speed: int) -> None:
        self.sent.append(("radians", rad, speed))

    def send_angles(self, deg: list[float], speed: int) -> None:
        self.sent.append(("angles", deg, speed))

    def get_angles(self) -> Any:
        return self.reply


class FakeMyCobot(_FakeMyCobotBase):
    """The well-behaved handle: a public `close()` that releases the port."""

    def __init__(self, *, reply: Any = _UNSET) -> None:
        super().__init__(reply=reply)
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class FakeMyCobotNoClose(_FakeMyCobotBase):
    """A handle exposing no close/disconnect -- what the fallback chain is for."""


# ------------------------------------------------- the ArmInterface contract

ARMS = {
    "dry_run": lambda jm: DryRunArm(jm, clock=FakeClock()),
    "mypalletizer": lambda jm: MyPalletizerArm(None, jm, transport=FakeMyCobot()),
}


@pytest.fixture(params=sorted(ARMS))
def arm(request: pytest.FixtureRequest, jmap: JointMap) -> ArmInterface:
    """One suite over every arm, the way test_sensor_contract.py does sensors."""
    return ARMS[request.param](jmap)


def test_every_arm_is_an_arminterface(arm: ArmInterface) -> None:
    assert isinstance(arm, ArmInterface)
    assert isinstance(arm.name, str) and arm.name
    assert isinstance(arm.jmap, JointMap)


def test_send_takes_model_radians_and_needs_no_speed(arm: ArmInterface) -> None:
    """The caller never converts to API degrees; that is the arm's job."""
    arm.send(np.deg2rad([10.0, 20.0, 30.0]))
    arm.send(np.deg2rad([10.0, 20.0, 30.0]), speed=50)


def test_read_is_four_api_degrees_or_none(arm: ArmInterface) -> None:
    arm.send(np.deg2rad([10.0, 20.0, 30.0]))
    out = arm.read()
    assert out is None or (out.shape == (4,) and out.dtype == np.float64)


def test_close_is_idempotent(arm: ArmInterface) -> None:
    arm.close()
    arm.close()


def test_the_context_manager_closes_even_when_the_body_raises(jmap: JointMap) -> None:
    """The notebook's bare `arm.close()` after the loop leaks the port on error."""
    mc = FakeMyCobot()
    with pytest.raises(RuntimeError, match="boom"), MyPalletizerArm(None, jmap, transport=mc):
        raise RuntimeError("boom")
    assert mc.closed == 1


def test_send_rejects_a_trajectory_with_the_wrong_width(arm: ArmInterface) -> None:
    """qpos has 4 entries; only 3 are commanded. `act` must not reach the port."""
    with pytest.raises(ValueError, match="4 columns"):
        arm.send(np.zeros(4))


# ---------------------------------------------------------------- JointMap

def test_the_map_round_trips(jmap: JointMap) -> None:
    q = np.deg2rad([[10.0, 20.0, 30.0], [-5.0, 0.0, 15.0]])
    assert np.allclose(jmap.to_model_rad(jmap.to_api_deg(q)), q, rtol=0, atol=1e-15)


def test_j4_is_held_and_is_not_a_model_joint(jmap: JointMap) -> None:
    api = jmap.to_api_deg(np.deg2rad([[10.0, 20.0, 30.0]]))
    assert api.shape == (1, 4)
    assert api[0, 3] == 0.0
    assert jmap.to_model_rad(api).shape == (1, 3)


def test_the_map_cannot_be_mutated_in_place(jmap: JointMap) -> None:
    """3.5's `SIGNS[:] = signos` must be impossible, not merely discouraged.

    `frozen=True` alone does not achieve this: it blocks rebinding the
    attribute, not writing through the array the attribute points at. The
    arrays are flagged read-only for the same reason `Measurement.R` is.
    """
    with pytest.raises(ValueError, match="read-only"):
        jmap.signs[:] = [1.0, -1.0, 1.0]
    with pytest.raises(ValueError, match="read-only"):
        jmap.offsets_deg[0] = 5.0
    with pytest.raises(AttributeError):
        jmap.vmax_deg_s = 10.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        jmap.api_limits_deg[2] = (-90.0, 90.0)  # type: ignore[index]


def test_a_zero_sign_is_refused():
    """to_model_rad divides by signs; a 0 makes the map quietly one-way."""
    with pytest.raises(ValueError, match="not invertible"):
        make_jmap(signs=(1.0, 0.0, 1.0))


def test_shape_mismatches_are_refused() -> None:
    with pytest.raises(ValueError, match="signs must be"):
        make_jmap(signs=(1.0, 1.0))
    with pytest.raises(ValueError, match="no entry for API joints"):
        make_jmap(api_limits_deg={1: (-1.0, 1.0)})


# ------------------------------------------- the guard, and its three failures

@pytest.mark.mujoco
def test_the_real_trajectory_passes(jmap: JointMap, model: Any) -> None:
    t, q = sine_trajectory((30.0, 10.0, 20.0), 3.0)
    api = jmap.validate(q, model, t, verbose=False)
    assert api.shape == (len(t), 4)


@pytest.mark.mujoco
def test_a_position_past_the_xml_range_is_rejected(jmap: JointMap, model: Any) -> None:
    """Case 1: J3 sweeps to 80 deg, past link2's XML range and the API limit."""
    t, q = sine_trajectory((30.0, 10.0, 40.0), 3.0)
    with pytest.raises(ValueError, match="J3/link2") as exc:
        jmap.validate(q, model, t, verbose=False)
    assert "outside the XML range" in str(exc.value)


@pytest.mark.mujoco
def test_an_inverted_joint_is_rejected(model: Any) -> None:
    """Case 2: an inverted J2 sweeps 0 -> -20 deg, through J2's floor of -2.

    The notebook did this by assigning into the SIGNS global under
    try/finally. A frozen JointMap makes that impossible, so the miswiring is
    expressed the way it should be: as a different map.
    """
    t, q = sine_trajectory((30.0, 10.0, 20.0), 3.0)
    with pytest.raises(ValueError, match="J2/link1") as exc:
        make_jmap(signs=(1.0, -1.0, 1.0)).validate(q, model, t, verbose=False)
    assert "outside the XML range" in str(exc.value)


@pytest.mark.mujoco
def test_a_too_fast_trajectory_is_rejected(jmap: JointMap, model: Any) -> None:
    """Case 3: the one only the velocity check can catch.

    Shortening the period moves no position at all -- the amplitude is
    unchanged -- so the position table still reports OK while peak speed grows
    as 1/period. 30 deg amplitude over 1.0 s peaks at 2*pi*30 = 188 deg/s.
    """
    t, q = sine_trajectory((30.0, 10.0, 20.0), 1.0)
    with pytest.raises(ValueError, match="over the arm's spec") as exc:
        jmap.validate(q, model, t, verbose=False)
    assert "188" in str(exc.value)


@pytest.mark.mujoco
def test_without_t_the_same_fast_trajectory_passes(jmap: JointMap, model: Any) -> None:
    """The positive control, and the reason both real call sites pass `t`.

    This is not a bug being tolerated: without a time axis there is no
    velocity to check. It is asserted so that the velocity check above is
    known to be what rejected the trajectory, rather than something else.
    """
    _, q = sine_trajectory((30.0, 10.0, 20.0), 1.0)
    assert jmap.validate(q, model, verbose=False).shape == (len(q), 4)


@pytest.mark.mujoco
def test_the_verbose_table_prints_a_verdict(
    jmap: JointMap, model: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """The table is the only thing a human sees before the arm moves."""
    t, q = sine_trajectory((30.0, 10.0, 20.0), 3.0)
    jmap.validate(q, model, t)
    out = capsys.readouterr().out
    assert "OUT OF RANGE" not in out
    for name in jmap.names:
        assert name in out
    assert "held at 0.0 deg" in out


@pytest.mark.mujoco
def test_ranges_are_looked_up_by_name_not_by_position(model: Any) -> None:
    """3.7's open item: `JOINT_NAMES[k]` and `model.jnt_range[k]` agreeing is luck.

    The XML happens to declare rot, link1, link2, act in that order, so
    positional lookup is right today and a regression would be invisible. A
    JointMap whose columns are in another order is a legitimate config, and it
    is where the two lookups part company:

        column 0 is `link1`, whose XML range is [0.0, 90.0] deg
        jnt_range[0] is  `rot`, whose XML range is [-159.9, 159.9] deg

    A trajectory reaching -1.5 deg on that column sits inside rot's range and
    inside J2's API limit of [-2, 90], and outside link1's. Positional lookup
    passes it; lookup by name rejects it.

    This used `link2` at -50 deg until 2026-10-21, when link2's XML range was
    widened to the API floor of -92 deg (J3 really does go negative, see
    docs/j2j3_envelope_handoff.md). Since then no J3 value is outside link2's
    range yet inside J3's API limit, so link2 can no longer tell the lookups
    apart. link1's XML floor of 0 still sits 2 deg inside J2's API floor of -2.
    """
    reordered = make_jmap(names=("link1", "rot", "link2"), api_ids=(2, 1, 3))
    t = np.linspace(0.0, 3.0, 1500)
    q = np.zeros((t.size, 3))
    q[:, 0] = np.deg2rad(-1.5 * np.sin(2 * np.pi / 3.0 * t) ** 2)

    # What the guard does now.
    with pytest.raises(ValueError, match="J2/link1") as exc:
        reordered.validate(q, model, t, verbose=False)
    assert "[0.0,90.0]" in str(exc.value)

    # And what the pre-P2 guard concluded on the same trajectory: nothing at
    # all. This is the whole of check_trajectory's position test, with the one
    # difference that made it wrong -- `model.jnt_range[k]` for k in 0..2.
    api = reordered.to_api_deg(q)
    complaints = []
    for k in range(reordered.n_joints):
        lo_xml, hi_xml = np.rad2deg(model.jnt_range[k])
        lo_api, hi_api = API_LIMITS_DEG[reordered.api_ids[k]]
        lo, hi = api[:, k].min(), api[:, k].max()
        if model.jnt_limited[k] and not (lo >= lo_xml - 1e-9 and hi <= hi_xml + 1e-9):
            complaints.append(f"XML {k}")
        if not (lo >= lo_api and hi <= hi_api):
            complaints.append(f"API {k}")
    qd = np.abs(np.gradient(np.rad2deg(q), t, axis=0)).max(axis=0)
    assert qd.max() == pytest.approx(3.14, abs=0.05)  # far under the 120 deg/s spec
    assert complaints == [], (
        "positional lookup was supposed to wave this trajectory through, and "
        f"it did not: {complaints}. The falsification no longer falsifies."
    )


# ------------------------------------------------------------- DryRunArm

def test_nothing_moves_before_the_transport_delay(jmap: JointMap) -> None:
    clock = FakeClock()
    arm = DryRunArm(jmap, latency_s=0.08, clock=clock)
    arm.send(np.deg2rad([30.0, 20.0, 10.0]))
    clock.t = 0.079
    assert np.all(arm.read() == 0.0)
    clock.t = 0.081
    assert np.any(arm.read() != 0.0)


def test_the_first_order_lag_approaches_without_overshoot(jmap: JointMap) -> None:
    """One tau reaches 1 - 1/e of the step; the state never passes the setpoint."""
    clock = FakeClock()
    arm = DryRunArm(jmap, latency_s=0.0, tau_s=0.06, rate_limit_dps=1e6, clock=clock)
    arm.send(np.deg2rad([30.0, 0.0, 0.0]))
    clock.t = 0.06
    assert float(arm.read()[0]) == pytest.approx(30.0 * (1 - np.exp(-1.0)), abs=0.01)
    for step in np.arange(0.07, 1.0, 0.01):
        clock.t = float(step)
        assert float(arm.read()[0]) <= 30.0 + 1e-9
    clock.t = 5.0
    assert float(arm.read()[0]) == pytest.approx(30.0, abs=1e-6)


def test_the_rate_limit_binds_on_a_large_step(jmap: JointMap) -> None:
    """Without the cap the lag alone would move 33.9 deg in the first 50 ms."""
    clock = FakeClock()
    capped = DryRunArm(jmap, latency_s=0.0, tau_s=0.06, rate_limit_dps=120.0, clock=clock)
    free = DryRunArm(jmap, latency_s=0.0, tau_s=0.06, rate_limit_dps=1e6, clock=clock)
    q = np.deg2rad([60.0, 0.0, 0.0])
    capped.send(q)
    free.send(q)
    clock.t = 0.05
    assert float(capped.read()[0]) == pytest.approx(120.0 * 0.05, abs=1e-9)
    assert float(free.read()[0]) == pytest.approx(60.0 * (1 - np.exp(-0.05 / 0.06)), abs=0.01)


def test_readback_is_quantised_to_the_api_resolution(jmap: JointMap) -> None:
    """The API transmits int(degrees * 100), so 0.01 deg is the real grid."""
    clock = FakeClock()
    arm = DryRunArm(jmap, latency_s=0.0, tau_s=0.06, clock=clock)
    arm.send(np.deg2rad([30.0, 20.0, 10.0]))
    clock.t = 0.037
    out = arm.read()
    assert np.allclose(out, np.round(out * 100.0) / 100.0, rtol=0, atol=1e-12)


def test_a_zero_tau_is_refused(jmap: JointMap) -> None:
    with pytest.raises(ValueError, match="tau_s must be"):
        DryRunArm(jmap, tau_s=0.0)


# ---------------------------------------------------------- MyPalletizerArm

def test_send_defaults_to_the_non_blocking_call(jmap: JointMap) -> None:
    """send_angles blocks on a reply per setpoint, which does not fit at 25 Hz."""
    mc = FakeMyCobot()
    MyPalletizerArm(None, jmap, transport=mc).send(np.deg2rad([10.0, 20.0, 30.0]))
    kind, payload, speed = mc.sent[0]
    assert kind == "radians"
    assert speed == 100
    # send_radians converts back to degrees internally, so the values handed
    # over are the API degrees expressed in radians.
    assert np.allclose(np.rad2deg(payload), [10.0, 20.0, 30.0, 0.0])


def test_blocking_send_uses_send_angles(jmap: JointMap) -> None:
    mc = FakeMyCobot()
    arm = MyPalletizerArm(None, jmap, transport=mc, blocking_send=True, speed=40)
    arm.send(np.deg2rad([10.0, 20.0, 30.0]), speed=55)
    kind, payload, speed = mc.sent[0]
    assert kind == "angles"
    assert speed == 55
    assert np.allclose(payload, [10.0, 20.0, 30.0, 0.0])


@pytest.mark.parametrize(
    "reply", [-1, None, [1.0, 2.0], "error"], ids=["minus_one", "none", "short", "text"]
)
def test_a_bad_reply_reads_as_none(jmap: JointMap, reply: Any) -> None:
    """get_angles returns -1 on error; None must stay distinct from a reading."""
    mc = FakeMyCobot(reply=reply)
    assert MyPalletizerArm(None, jmap, transport=mc).read() is None


def test_close_prefers_the_public_method(jmap: JointMap) -> None:
    mc = FakeMyCobot()
    arm = MyPalletizerArm(None, jmap, transport=mc)
    arm.close()
    arm.close()
    assert mc.closed == 1


def test_close_falls_back_to_the_private_port(jmap: JointMap) -> None:
    """What the notebook did, kept only as a fallback."""

    class Port:
        def __init__(self) -> None:
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    mc = FakeMyCobotNoClose()
    mc._serial_port = Port()  # type: ignore[attr-defined]
    MyPalletizerArm(None, jmap, transport=mc).close()
    assert mc._serial_port.closed == 1  # type: ignore[attr-defined]


def test_a_close_that_cannot_release_the_port_says_so(jmap: JointMap) -> None:
    """The notebook swallowed this under `except Exception: pass`.

    A close that silently does nothing leaves the device held until the
    interpreter exits, and the next run fails at construction with a message
    about the port being busy -- nowhere near the cause.
    """
    mc = FakeMyCobotNoClose()
    with pytest.raises(RuntimeError, match="still held"):
        MyPalletizerArm(None, jmap, transport=mc).close()


def test_constructing_without_a_port_or_transport_is_refused(jmap: JointMap) -> None:
    with pytest.raises(ValueError, match="port or a transport"):
        MyPalletizerArm(None, jmap)


def test_importing_the_package_does_not_need_pymycobot() -> None:
    """pymycobot is in the [app] extra and is not installed in CI."""
    import sys

    assert "pymycobot" not in sys.modules


# ------------------------------------------- the J3(J2) envelope and its guard

# A synthetic envelope with the measured shape (docs/j2j3_envelope_handoff.md):
# the upper stop flat at 58 deg to J2 = 40, then falling 0.93 deg per deg; the
# lower stop a straight line. Grid every 10 deg of J2 over [-10, 90].
ENV_J2 = np.arange(-10.0, 91.0, 10.0)
ENV_LO = -20.0 - 0.95 * ENV_J2
ENV_HI = np.where(ENV_J2 <= 40.0, 58.0, 58.0 - 0.93 * (ENV_J2 - 40.0))
TWO_EACH = {"max": ("a_max.csv", "b_max.csv"), "min": ("a_min.csv", "b_min.csv")}
ONE_EACH = {"max": ("a_max.csv",), "min": ("a_min.csv",)}

# J2 = 80 deg, J3 = 50 deg: inside every box (J3 <= 60) and above the envelope,
# whose upper bound there is 58 - 0.93 * 40 = 20.8 deg. The case the boxes miss.
OVER_THE_STOP = np.deg2rad([0.0, 80.0, 50.0])


def make_envelope(**kw: Any) -> CoupledLimit:
    return CoupledLimit(
        driver_deg=kw.pop("driver_deg", ENV_J2),
        lo_deg=kw.pop("lo_deg", ENV_LO),
        hi_deg=kw.pop("hi_deg", ENV_HI),
        recordings=kw.pop("recordings", TWO_EACH),
        source=kw.pop("source", "synthetic"),
        **kw,
    )


@pytest.fixture(scope="module")
def jmap_env() -> JointMap:
    return make_jmap(coupled=make_envelope())


def _sent(arm: ArmInterface) -> list[Any]:
    """What reached the transport (real arm) or the delay line (dry run)."""
    if isinstance(arm, MyPalletizerArm):
        return list(arm.mc.sent)  # type: ignore[attr-defined]
    assert isinstance(arm, DryRunArm)
    return list(arm._queue)


@pytest.mark.parametrize("kind", sorted(ARMS))
def test_every_arm_refuses_a_setpoint_over_the_stop(kind: str, jmap_env: JointMap) -> None:
    """The guard lives in each arm's `send`; one suite proves both run it."""
    arm = ARMS[kind](jmap_env)
    with pytest.raises(ValueError, match=r"envelope.*J3 = 50\.00 deg outside"):
        arm.send(OVER_THE_STOP)
    assert _sent(arm) == [], "a rejected setpoint still reached the transport"


@pytest.mark.parametrize("kind", sorted(ARMS))
def test_without_the_envelope_the_same_setpoint_is_sent(kind: str, jmap: JointMap) -> None:
    """The paired control: the boxes alone pass it, so the envelope rejected it."""
    arm = ARMS[kind](jmap)
    arm.send(OVER_THE_STOP)
    assert len(_sent(arm)) == 1


@pytest.mark.parametrize("kind", sorted(ARMS))
def test_every_arm_refuses_a_setpoint_outside_the_box(kind: str, jmap: JointMap) -> None:
    """`send` checked nothing before; J2 = 95 deg is past its API limit of 90."""
    arm = ARMS[kind](jmap)
    with pytest.raises(ValueError, match=r"J2/link1 = 95\.00 deg outside the API limit"):
        arm.send(np.deg2rad([0.0, 95.0, 0.0]))
    assert _sent(arm) == []


def test_a_setpoint_inside_the_envelope_is_sent(jmap_env: JointMap) -> None:
    mc = FakeMyCobot()
    MyPalletizerArm(None, jmap_env, transport=mc).send(np.deg2rad([0.0, 80.0, 15.0]))
    assert np.allclose(np.rad2deg(mc.sent[0][1]), [0.0, 80.0, 15.0, 0.0])


def test_the_envelope_interpolates_and_never_extrapolates() -> None:
    env = make_envelope()
    lo, hi = env.bounds([40.0, 45.0, 80.0])
    assert np.allclose(hi, [58.0, 58.0 - 0.93 * 5.0, 58.0 - 0.93 * 40.0])
    assert np.allclose(lo, -20.0 - 0.95 * np.array([40.0, 45.0, 80.0]))
    lo, hi = env.bounds([-10.5, 90.5])
    assert np.all(np.isnan(lo)) and np.all(np.isnan(hi))
    # What np.interp would have said just below the table: the first row's
    # bounds, i.e. an unmeasured pose reading as permitted.
    assert np.interp(-10.5, ENV_J2, ENV_HI) == pytest.approx(58.0)


def test_a_gap_in_the_envelope_forbids_its_interval_only() -> None:
    hi = ENV_HI.copy()
    hi[5] = np.nan  # J2 = 40
    env = make_envelope(hi_deg=hi)
    _, b = env.bounds([30.0, 35.0, 45.0, 50.0])
    assert b[0] == pytest.approx(58.0) and b[3] == pytest.approx(58.0 - 0.93 * 10.0)
    assert np.isnan(b[1]) and np.isnan(b[2])
    with pytest.raises(ValueError, match=r"no measured limit at J2 = 35\.00"):
        make_jmap(coupled=env).check_setpoint(np.deg2rad([0.0, 35.0, 0.0]))


def test_an_inverted_envelope_is_refused() -> None:
    """What the swapped max/min recordings of 2026-10-21 exported: lo > hi."""
    with pytest.raises(ValueError, match="inverted envelope"):
        make_envelope(lo_deg=ENV_HI, hi_deg=ENV_LO)


def test_a_malformed_envelope_is_refused() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        make_envelope(driver_deg=ENV_J2[::-1])
    with pytest.raises(ValueError, match="must match"):
        make_envelope(lo_deg=ENV_LO[:-1])
    with pytest.raises(ValueError, match="names API joints"):
        make_jmap(coupled=make_envelope(driver_api_id=4))


def test_the_envelope_cannot_be_mutated_in_place() -> None:
    env = make_envelope()
    with pytest.raises(ValueError, match="read-only"):
        env.hi_deg[0] = 90.0
    with pytest.raises(TypeError):
        env.recordings["max"] = ()  # type: ignore[index]


def _write_envelope_csv(path: Any, lo: np.ndarray, hi: np.ndarray, recordings: Any) -> None:
    """The notebook's export format, byte for byte where it matters."""
    columns = ["j2", "j3_min", "j3_max", "j3_min_seguro", "j3_max_seguro", "n_min", "n_max"]
    header = "\n".join([
        "Envolvente J2 -> J3 en grados API, generada por la prueba. NaN = no medido.",
        f"max: {', '.join(recordings['max'])}",
        f"min: {', '.join(recordings['min'])}",
        "cuantil 0.9, ancho_bin 2.5, margen min 1.00 max 1.00",
        ",".join(columns),
    ])
    n = np.full(len(ENV_J2), 10)
    data = np.column_stack([ENV_J2, lo - 1.0, hi + 1.0, lo, hi, n, n])
    np.savetxt(path, data, delimiter=",", header=header, fmt=["%.3f"] * 5 + ["%d"] * 2)


def test_the_envelope_loads_from_the_notebook_csv(tmp_path: Any) -> None:
    path = tmp_path / "j2j3_envelope.csv"
    _write_envelope_csv(path, ENV_LO, ENV_HI, TWO_EACH)
    env = CoupledLimit.from_csv(path)
    # The margin-applied columns, not the raw stops beside them.
    assert np.allclose(env.lo_deg, ENV_LO) and np.allclose(env.hi_deg, ENV_HI)
    assert dict(env.recordings) == TWO_EACH
    assert env.ready


def test_an_inverted_csv_is_refused_at_load(tmp_path: Any) -> None:
    path = tmp_path / "j2j3_envelope.csv"
    _write_envelope_csv(path, ENV_HI, ENV_LO, TWO_EACH)
    with pytest.raises(ValueError, match="inverted envelope"):
        CoupledLimit.from_csv(path)


def test_one_recording_per_side_is_not_ready() -> None:
    env = make_envelope(recordings=ONE_EACH)
    assert not env.ready
    assert "fewer than 2 recordings per side" in str(env.not_ready_reason)
    assert make_envelope(recordings={}).not_ready_reason is not None
    assert make_envelope().ready


def test_the_real_arm_refuses_an_unready_envelope_before_opening_a_port() -> None:
    """A serial error here would mean pymycobot ran first; the port is never touched."""
    import sys

    unready = make_jmap(coupled=make_envelope(recordings=ONE_EACH))
    with pytest.raises(ValueError, match="not ready to gate the real arm"):
        MyPalletizerArm("COMX", unready)
    with pytest.raises(ValueError, match="not ready to gate the real arm"):
        MyPalletizerArm(None, unready, transport=FakeMyCobot())
    assert "pymycobot" not in sys.modules


def test_the_dry_run_accepts_an_unready_envelope() -> None:
    unready = make_jmap(coupled=make_envelope(recordings=ONE_EACH))
    arm = DryRunArm(unready, clock=FakeClock())
    with pytest.raises(ValueError, match="envelope"):
        arm.send(OVER_THE_STOP)


def test_clamp_moves_an_outside_point_onto_the_envelope(jmap_env: JointMap) -> None:
    q = jmap_env.clamp_rad(OVER_THE_STOP)
    assert np.allclose(np.rad2deg(q), [0.0, 80.0, 58.0 - 0.93 * 40.0])
    assert jmap_env.check_setpoint(q).shape == (4,)  # the guard accepts it
    assert np.array_equal(jmap_env.clamp_rad(q), q)  # and clamping is idempotent


def test_clamp_leaves_an_inside_point_alone(jmap_env: JointMap) -> None:
    q = np.deg2rad([10.0, 30.0, 20.0])
    assert np.allclose(jmap_env.clamp_rad(q), q, rtol=0, atol=1e-15)


def test_clamp_brings_the_driver_into_the_measured_span() -> None:
    """J2 = 95 deg is past both the API box and the table; J3 follows it down."""
    env = make_envelope(driver_deg=ENV_J2[:-2], lo_deg=ENV_LO[:-2], hi_deg=ENV_HI[:-2])
    jm = make_jmap(coupled=env)
    q = jm.clamp_rad(np.deg2rad([[0.0, 95.0, 50.0], [0.0, -5.0, 0.0]]))
    assert q.shape == (2, 3)
    assert np.allclose(np.rad2deg(q[0]), [0.0, 70.0, 58.0 - 0.93 * 30.0])
    assert np.allclose(np.rad2deg(q[1]), [0.0, -2.0, 0.0])  # the box, not the table


@pytest.mark.mujoco
def test_the_real_trajectory_passes_with_the_envelope(jmap_env: JointMap, model: Any) -> None:
    t, q = sine_trajectory((30.0, 10.0, 20.0), 3.0)
    assert jmap_env.validate(q, model, t, verbose=False).shape == (len(t), 4)


@pytest.mark.mujoco
def test_a_trajectory_over_the_stop_is_rejected(
    jmap: JointMap, jmap_env: JointMap, model: Any
) -> None:
    """J2 to 80 and J3 to 50 together: every box passes it, the envelope does not.

    Peak speed is 2*pi*40/3 = 84 deg/s, under the spec, so the envelope is the
    only check that can fire -- which the plain map passing it confirms.
    """
    t, q = sine_trajectory((0.0, 40.0, 25.0), 3.0)
    assert jmap.validate(q, model, t, verbose=False).shape == (len(t), 4)
    with pytest.raises(ValueError, match=r"J3/link2: \d+ of \d+ setpoints outside the measured"):
        jmap_env.validate(q, model, t, verbose=False)


@pytest.mark.mujoco
def test_the_verbose_table_reports_the_envelope(
    jmap_env: JointMap, model: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    t, q = sine_trajectory((30.0, 10.0, 20.0), 3.0)
    jmap_env.validate(q, model, t)
    assert "J3(J2) envelope from synthetic: OK" in capsys.readouterr().out
