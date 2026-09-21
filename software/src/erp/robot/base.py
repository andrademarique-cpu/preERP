"""Arm command sink: the transport contract and the joint map that gates it.

An arm is a **command sink**, never a measurement source. Nothing in the
estimation stack -- ``core``, ``models``, ``sim``, ``estimators``, ``fusion``
-- may import this package; its only consumer inside ``erp`` is the runtime
that constructs it. :meth:`ArmInterface.read` exists so the loop can log what
the arm reports back, not to feed the filter: if encoder readback ever becomes
a measurement it enters through a ``Sensor`` like everything else.

Documented in English because this package sits beside :mod:`erp.sensors`
rather than under :mod:`erp.sim`. The measured constants lifted from the
notebook are carried unchanged in the docstrings that explain them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from erp.core.types import Array

# MuJoCo's Python bindings are untyped. Alias the one type used here so the
# signatures stay readable and `Any` does not travel past this module.
MjModel = Any

__all__ = ["ArmInterface", "JointMap"]


@dataclass(frozen=True)
class JointMap:
    """Model joints <-> API joints, and the range guard that runs before a send.

    Replaces the ``SIGNS`` / ``OFFSETS_DEG`` / ``API_LIMITS_DEG`` / ``J4_HOLD_DEG``
    module globals of the notebook, which a negative-test cell mutated in place
    under ``try/finally``.

    The correspondence, for this arm::

        API J1 <-> XML `rot`     base, yaw
        API J2 <-> XML `link1`   shoulder
        API J3 <-> XML `link2`   elbow
        API J4 <-> nothing       the XML's fourth joint is `act`, the passive
                                 parallelogram, not J4.

    **Why there is a guard here at all, rather than trusting pymycobot.** The
    library does not validate angles in this class: ``send_angles`` passes
    ``angles=...`` and ``send_angle`` passes ``angle=...``, but
    ``MyPalletizer260.calibration_parameters`` only inspects the ``degrees``
    and ``degree`` kwargs. Those checks never fire -- only ``id`` and ``speed``
    are really validated -- and the class-wide bound is +-170 deg, far wider
    than any real joint limit. An out-of-range value reaches the firmware with
    nothing having looked at it.

    ``signs`` and ``offsets_deg`` stay explicit even though both are identity
    today: if a joint is ever inverted it is changed here, and the guard
    catches it before anything moves (an inverted J2 sweeps 0 -> -20 deg,
    through J2's floor of -2 deg).

    Parameters
    ----------
    names:
        MuJoCo joint names, in the column order of the trajectories passed to
        :meth:`to_api_deg` and :meth:`validate`.
    api_ids:
        The API joint number each column commands, positionally aligned with
        ``names``. Used to look ``api_limits_deg`` up and to label errors.
    signs, offsets_deg:
        ``api_deg = rad2deg(q) * signs + offsets_deg``, applied per column.
    api_limits_deg:
        API joint number -> (lo, hi) in degrees. Keyed by API id, not by
        column, because that is how the arm's documentation states them.
    vmax_deg_s:
        Per-joint speed ceiling, deg/s. 120 is the arm's spec and the only
        hardware number we have for it.
    j4_hold_deg:
        J4 is unmodelled and held here. The API wants four values; the model
        has three actuated joints.
    """

    names: tuple[str, ...]
    api_ids: tuple[int, ...]
    signs: Array
    offsets_deg: Array
    api_limits_deg: Mapping[int, tuple[float, float]]
    vmax_deg_s: float = 120.0
    j4_hold_deg: float = 0.0

    def __post_init__(self) -> None:
        """Validate the shapes and make the arrays read-only.

        ``frozen=True`` alone is not enough to retire the ``SIGNS[:] = ...``
        mutation it was introduced to prevent: freezing blocks rebinding the
        attribute, not writing through the array it points at. The arrays are
        flagged non-writeable for the same reason ``shared_rows_R`` does it to
        ``rows``/``R`` -- a caller editing one in place would silently change
        the mapping for every command sent before and after.
        """
        n = len(self.names)
        if n == 0:
            raise ValueError("names must not be empty")
        if len(self.api_ids) != n:
            raise ValueError(f"api_ids has {len(self.api_ids)} entries for {n} names")
        signs = np.array(self.signs, dtype=np.float64).reshape(-1)
        offsets = np.array(self.offsets_deg, dtype=np.float64).reshape(-1)
        if signs.shape != (n,):
            raise ValueError(f"signs must be ({n},), got {signs.shape}")
        if offsets.shape != (n,):
            raise ValueError(f"offsets_deg must be ({n},), got {offsets.shape}")
        if np.any(signs == 0.0):
            # to_model_rad divides by signs; a zero makes the map one-way and
            # the reverse trip silently infinite rather than wrong-by-a-little.
            raise ValueError("a sign of 0 is not invertible; use +1 or -1")
        missing = sorted(set(self.api_ids) - set(self.api_limits_deg))
        if missing:
            raise ValueError(f"api_limits_deg has no entry for API joints {missing}")
        signs.flags.writeable = False
        offsets.flags.writeable = False
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "api_ids", tuple(self.api_ids))
        object.__setattr__(self, "signs", signs)
        object.__setattr__(self, "offsets_deg", offsets)
        object.__setattr__(self, "api_limits_deg", MappingProxyType(dict(self.api_limits_deg)))

    @property
    def n_joints(self) -> int:
        """Number of commanded model joints (3 here; J4 is not one of them)."""
        return len(self.names)

    def to_api_deg(self, q_rad: npt.ArrayLike) -> Array:
        """(N, n) model radians -> (N, 4) API degrees.

        The fourth column is J4, held at :attr:`j4_hold_deg`.
        """
        q = np.atleast_2d(np.asarray(q_rad, dtype=np.float64))
        if q.shape[1] != self.n_joints:
            raise ValueError(
                f"trajectory has {q.shape[1]} columns; expected {self.n_joints} "
                f"{self.names}. qpos[3] is `act` and is not commanded."
            )
        deg = np.rad2deg(q) * self.signs + self.offsets_deg
        out = np.column_stack([deg, np.full(len(deg), self.j4_hold_deg)])
        return np.asarray(out, dtype=np.float64)

    def to_model_rad(self, deg: npt.ArrayLike) -> Array:
        """(N, 4) API degrees -> (N, n) model radians. Drops J4."""
        d = np.atleast_2d(np.asarray(deg, dtype=np.float64))
        if d.shape[1] < self.n_joints:
            raise ValueError(f"expected at least {self.n_joints} columns, got {d.shape[1]}")
        d = d[:, : self.n_joints]
        return np.asarray(np.deg2rad((d - self.offsets_deg) / self.signs), dtype=np.float64)

    def validate(
        self,
        q_rad: npt.ArrayLike,
        model: MjModel,
        t: npt.ArrayLike | None = None,
        *,
        verbose: bool = True,
    ) -> Array:
        """Check positions against XML + API limits and, given ``t``, velocity.

        Was ``check_trajectory`` in the notebook. Raises ``ValueError`` before
        anything reaches the serial port, and returns the (N, 4) API-degree
        trajectory on success so the caller can log what was approved.

        The XML ranges are **read from the loaded model**, not copied here, so
        this stays true when the XML's ``range`` attributes change.

        **The velocity check is the one that matters when tuning speed.**
        Shortening the trajectory period does not move a single position --
        the amplitude is unchanged -- so the position table keeps reporting OK
        while peak velocity grows as ``1/period``. Without ``t`` the only
        guard in the system is blind to exactly the parameter being turned.
        Callers that have a time axis must pass it; :meth:`validate` says so
        out loud when they do not.

        Parameters
        ----------
        q_rad:
            (N, n) model radians, in ``names`` order.
        model:
            The loaded ``MjModel``. Ranges are looked up **by joint name**, not
            by position: ``JOINT_NAMES[k]`` and ``model.jnt_range[k]`` agree
            today, but nothing enforces that, and reordering the XML's joints
            would silently check each column against another joint's limits.
        t:
            (N,) seconds. Without it velocity cannot be checked.
        """
        q = np.atleast_2d(np.asarray(q_rad, dtype=np.float64))
        api = self.to_api_deg(q)

        rows: list[tuple[int, str, float, float, float, float, float, float, bool, bool]] = []
        bad: list[str] = []
        for k, name in enumerate(self.names):
            jid = self.api_ids[k]
            joint = model.joint(name)
            jnt_range = np.rad2deg(np.asarray(joint.range, dtype=np.float64))
            lo_xml, hi_xml = float(jnt_range[0]), float(jnt_range[1])
            limited = bool(model.jnt_limited[int(joint.id)])
            lo_api, hi_api = self.api_limits_deg[jid]
            lo, hi = float(api[:, k].min()), float(api[:, k].max())
            ok_xml = (not limited) or (lo >= lo_xml - 1e-9 and hi <= hi_xml + 1e-9)
            ok_api = lo >= lo_api and hi <= hi_api
            rows.append((jid, name, lo, hi, lo_xml, hi_xml, lo_api, hi_api, ok_xml, ok_api))
            if not ok_xml:
                bad.append(
                    f"J{jid}/{name}: travel [{lo:.1f},{hi:.1f}] deg outside the "
                    f"XML range [{lo_xml:.1f},{hi_xml:.1f}]"
                )
            if not ok_api:
                bad.append(
                    f"J{jid}/{name}: travel [{lo:.1f},{hi:.1f}] deg outside the "
                    f"API limit [{lo_api:.1f},{hi_api:.1f}]"
                )

        qd: Array | None = None
        if t is not None:
            t_arr = np.asarray(t, dtype=np.float64)
            qd = np.asarray(
                np.abs(np.gradient(np.rad2deg(q), t_arr, axis=0)).max(axis=0), dtype=np.float64
            )
            for k, name in enumerate(self.names):
                if qd[k] > self.vmax_deg_s + 1e-9:
                    bad.append(
                        f"J{self.api_ids[k]}/{name}: peak speed {qd[k]:.1f} deg/s "
                        f"over the arm's spec ({self.vmax_deg_s:.0f} deg/s) -- "
                        "lengthen the period"
                    )

        if verbose:
            self._print_table(rows, qd)
        if bad:
            raise ValueError("Trajectory out of range, nothing sent:\n  - " + "\n  - ".join(bad))
        return api

    def _print_table(
        self,
        rows: list[tuple[int, str, float, float, float, float, float, float, bool, bool]],
        qd: Array | None,
    ) -> None:
        """The notebook's verdict table. Kept on a library call on purpose.

        This is the only output a human sees before the arm moves, and the
        notebook's two real call sites rely on it. It moves behind a report
        object when there is an ``erp.analysis`` to put one in (P8).
        """
        print(
            f"{'API':>4} {'XML':<7}{'travel (deg)':>18}{'XML range':>18}"
            f"{'API limit':>18}{'peak deg/s':>12}   verdict"
        )
        for jid, name, lo, hi, lx, hx, la, ha, ok_xml, ok_api in rows:
            k = self.api_ids.index(jid)
            ok_v = qd is None or bool(qd[k] <= self.vmax_deg_s + 1e-9)
            verdict = "OK" if (ok_xml and ok_api and ok_v) else "OUT OF RANGE"
            vel = f"{qd[k]:11.1f}" if qd is not None else f"{'(no t)':>11}"
            print(
                f"J{jid:<3} {name:<7}[{lo:7.1f},{hi:7.1f}]  [{lx:7.1f},{hx:7.1f}]  "
                f"[{la:7.1f},{ha:7.1f}]{vel}   {verdict}"
            )
        j4_lo, j4_hi = self.api_limits_deg.get(4, (float("nan"), float("nan")))
        print(
            f"J4   {'-':<7}{'(not modelled)':>18}{'-':>18}"
            f"{f'[{j4_lo:7.1f},{j4_hi:7.1f}]':>18}{'-':>12}   held at {self.j4_hold_deg:.1f} deg"
        )
        if qd is None:
            print("  NOTE: without `t` speed was not checked. Pass the time axis.")
        else:
            print(f"  arm speed spec: {self.vmax_deg_s:.0f} deg/s")


class ArmInterface(ABC):
    """A command sink. The notebook's ``DryRunArm`` and ``SerialArm`` fit it.

    Implementations own a :class:`JointMap` and take commands in **model
    radians**, converting to API degrees themselves. The caller never handles
    API units, which is what lets the same loop drive a dry run and the real
    arm without branching.

    ``send`` must not block on a reply. The loop that calls it is also pacing
    the trajectory and draining sensors, so a round trip per setpoint caps the
    achievable rate -- see :class:`~erp.robot.mypalletizer.MyPalletizerArm`.
    """

    name: str
    jmap: JointMap

    @abstractmethod
    def send(self, q_rad: npt.ArrayLike, speed: int | None = None) -> None:
        """Command one setpoint, given as (n,) model radians."""

    @abstractmethod
    def read(self) -> Array | None:
        """(4,) API degrees as the arm reports them, or ``None`` if no reply.

        ``None`` means "nothing came back this time", which is a normal
        outcome on a shared port, not an error. It must stay distinguishable
        from a real reading, which is why this is not a zeros vector.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the transport. Must be safe to call twice."""

    def __enter__(self) -> ArmInterface:
        return self

    def __exit__(self, *exc: object) -> None:
        """Close on the way out, including when the body raised.

        The notebook calls ``arm.close()`` as a plain statement after
        ``run_trajectory``, so an exception mid-trajectory leaks the port and
        the next construction fails with the device busy.
        """
        self.close()
