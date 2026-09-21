"""The real arm, over pymycobot. Constructing one opens a serial port."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from erp.core.types import Array
from erp.robot.base import ArmInterface, JointMap

__all__ = ["ArmTransport", "MyPalletizerArm"]


class ArmTransport(Protocol):
    """What :class:`MyPalletizerArm` needs from a pymycobot handle.

    Stated here because pymycobot is untyped and optional, so this is the only
    written record of the surface this module depends on. It is also what the
    tests implement in place of the real library.

    ``close`` is deliberately **not** part of it: which of ``close``,
    ``disconnect`` or ``_serial_port`` the real class exposes is exactly the
    question :meth:`MyPalletizerArm.close` has to answer at runtime.
    """

    def send_radians(self, rad: list[float], speed: int) -> None: ...

    def send_angles(self, deg: list[float], speed: int) -> None: ...

    def get_angles(self) -> Any: ...


class MyPalletizerArm(ArmInterface):
    """Real arm via pymycobot. Only instantiated when you ask for one.

    **Writing.** ``send_radians`` by default: it writes and returns.
    ``send_angles`` uses ``has_reply=True``, which BLOCKS waiting for the
    firmware's reply on every setpoint, and that does not fit at 25 Hz. The
    price of ``send_radians`` is that it does not consume that reply, so if
    the firmware answers anyway the bytes sit in the input buffer and the next
    ``get_angles()`` can read garbage. If readback comes back strange, set
    ``blocking_send=True`` and lower the stream rate -- you lose rate, but the
    protocol stays synchronised.

    ``MyPalletizer260`` has no ``set_fresh_mode`` (``MyCobot280`` does), so
    setpoints can QUEUE in the firmware rather than replace one another, and
    the backlog accumulates over a run. There is no host-side fix. That is why
    the lag between command and motion is **measured**, at ~395 ms on this
    arm, rather than assumed absent.

    Parameters
    ----------
    port:
        Serial device, e.g. ``"COM6"``. Ignored when ``transport`` is given.
    jmap:
        Joint map. Commands arrive in model radians and leave in API degrees.
    speed:
        API speed scale, 1..100. A firmware scale, not deg/s.
    baudrate:
        Passed through to pymycobot.
    blocking_send:
        Use ``send_angles`` (waits for a reply) instead of ``send_radians``.
    transport:
        An already-constructed pymycobot handle, used instead of building one.
        This is the seam the tests drive, the same way ``SerialIMUSensor``
        takes a ``transport`` instead of opening a port.
    """

    def __init__(
        self,
        port: str | None,
        jmap: JointMap,
        *,
        speed: int = 100,
        baudrate: str = "115200",
        blocking_send: bool = False,
        transport: ArmTransport | None = None,
    ) -> None:
        if transport is None:
            if port is None:
                raise ValueError("give either a port or a transport")
            # Imported here, not at module scope: pymycobot is in the [app]
            # extra, and importing `erp.robot` must not require it.
            from pymycobot import MyPalletizer260

            transport = MyPalletizer260(port, baudrate=baudrate)
        self.mc: ArmTransport = transport
        self.jmap = jmap
        self.speed = int(speed)
        self.blocking_send = bool(blocking_send)
        self._closed = False
        self.name = f"MyPalletizerArm({port}, speed={speed}, blocking_send={blocking_send})"

    def send(self, q_rad: npt.ArrayLike, speed: int | None = None) -> None:
        """Command one setpoint, given as (n,) model radians."""
        deg = self.jmap.to_api_deg(q_rad)[0]
        sp = self.speed if speed is None else int(speed)
        if self.blocking_send:
            self.mc.send_angles([float(d) for d in deg], sp)
        else:
            # send_radians converts to degrees internally, so we hand it the
            # API degrees expressed in radians and the conversion back is exact.
            self.mc.send_radians([float(r) for r in np.deg2rad(deg)], sp)

    def read(self) -> Array | None:
        """(4,) API degrees, or ``None`` if the reply did not come or came short.

        ``get_angles`` returns ``-1`` on error, which is why the type is
        checked before the length: an int has no ``len``.
        """
        a = self.mc.get_angles()
        if not isinstance(a, list | tuple) or len(a) < 3:
            return None
        out = np.zeros(4)
        out[: min(4, len(a))] = a[:4]
        return np.asarray(out, dtype=np.float64)

    def close(self) -> None:
        """Release the serial port, idempotently.

        The notebook reached into ``mc._serial_port`` behind a bare
        ``except Exception``, which closes the port when that private
        attribute happens to exist and silently does nothing when it does not
        -- leaving the device held until the interpreter exits.

        This tries the public method first and keeps the private attribute
        only as a fallback. The fallback is still here because pymycobot is an
        optional dependency that is **not installed in this environment**, so
        which of these names its ``MyPalletizer260`` actually exposes has not
        been verified against the library. If it turns out to have ``close``,
        this already prefers it; if it does not, the previous behaviour still
        happens. What is new is that a close which finds no way to release the
        port now says so instead of passing silently.
        """
        if self._closed:
            return
        for name in ("close", "disconnect"):
            method = getattr(self.mc, name, None)
            if callable(method):
                method()
                self._closed = True
                return
        port = getattr(self.mc, "_serial_port", None)
        if port is not None and callable(getattr(port, "close", None)):
            port.close()
            self._closed = True
            return
        raise RuntimeError(
            f"{type(self.mc).__name__} exposes no close(), disconnect() or "
            "_serial_port to release; the port is still held"
        )
