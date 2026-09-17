"""Threaded base class for live sensors that block on I/O.

A serial ``readline`` blocks until a line arrives. Doing that in the trajectory
loop would tie the setpoint rate to the IMU rate, so each live sensor gets one
reader thread that blocks there instead, and the loop only ever calls the
non-blocking :meth:`StreamSensor.drain`. The blocking wait inside pyserial
releases the GIL, so the reader costs the loop nothing while idle.
"""

from __future__ import annotations

import threading
from abc import abstractmethod
from collections import deque
from operator import attrgetter
from types import TracebackType

import numpy.typing as npt

from erp.core.types import Array, IntArray, Measurement
from erp.sensors.base import Sensor, SensorError, shared_rows_R

__all__ = ["StreamSensor"]

_by_time = attrgetter("timestamp")


class StreamSensor(Sensor):
    """A :class:`Sensor` fed by a background reader thread.

    Subclasses implement :meth:`_poll` (one blocking read -> one sample or
    ``None``) and optionally :meth:`_open` / :meth:`_close`. Everything about
    buffering, threading, counters and error propagation lives here, so every
    live sensor behaves the same way under load.

    Counters (read them; they are the only evidence of lost data):

    - ``received``: samples pushed into the buffer.
    - ``rejected``: lines that looked like data but did not decode.
    - ``overruns``: samples pushed while the buffer was full, each of which
      evicted the oldest unread sample. Non-zero means the loop is not draining
      fast enough for ``buffer_len``.
    """

    def __init__(
        self, name: str, rows: npt.ArrayLike, R: npt.ArrayLike, *, buffer_len: int = 4096
    ) -> None:
        """
        Parameters
        ----------
        name:
            Sensor name, copied into ``Measurement.source``.
        rows:
            (k,) indices into ``model.sensordata`` the samples correspond to.
        R:
            (k, k) noise covariance, units of ``z`` squared.
        buffer_len:
            Maximum unread samples kept. At 1 kHz the default holds 4 s, far
            longer than any sane gap between two drains.
        """
        if buffer_len < 1:
            raise ValueError(f"buffer_len must be >= 1, got {buffer_len}")
        self.name = name
        self._rows, self._R = shared_rows_R(rows, R)
        self._buf: deque[Measurement] = deque(maxlen=buffer_len)
        self.received = 0
        self.rejected = 0
        self.overruns = 0
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._first = threading.Event()
        self._error: BaseException | None = None

    # -- contract ---------------------------------------------------------

    @property
    def rows(self) -> IntArray:
        return self._rows

    @property
    def R(self) -> Array:
        return self._R

    def set_R(self, R: npt.ArrayLike) -> None:
        """Replace the noise covariance for samples produced from now on.

        Already-buffered samples keep the array they were created with: the
        shared array is swapped, never written in place.
        """
        self._rows, self._R = shared_rows_R(self._rows, R)

    def read(self) -> Measurement | None:
        try:
            return self._buf.popleft()
        except IndexError:
            self._raise_if_dead()
            return None

    def drain(self) -> list[Measurement]:
        """Return every buffered sample, ordered by timestamp.

        Samples already received are handed out even if the reader has since
        died; the :class:`SensorError` is raised on the first call that finds
        the buffer empty, so no data is lost and no failure is silent.
        """
        buf = self._buf
        out: list[Measurement] = []
        try:
            while True:
                out.append(buf.popleft())
        except IndexError:
            pass
        if not out:
            self._raise_if_dead()
            return out
        # Arrival order is almost always timestamp order already, so this is a
        # linear pass. It is not guaranteed: clock sync can lower its offset
        # estimate between two samples.
        out.sort(key=_by_time)
        return out

    # -- lifecycle --------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Open the device and start the reader thread. Idempotent."""
        if self.running:
            return
        self._error = None
        self._stopping.clear()
        self._open()
        self._thread = threading.Thread(target=self._run, name=f"{self.name}-reader", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        """Stop the reader thread and close the device. Idempotent."""
        self._stopping.set()
        try:
            self._close()
        finally:
            if self._thread is not None:
                self._thread.join(timeout)
                self._thread = None

    def wait_first(self, timeout: float | None = None) -> bool:
        """Block until the first valid sample arrives; ``False`` on timeout."""
        ok = self._first.wait(timeout)
        if not ok:
            self._raise_if_dead()
        return ok

    def __enter__(self) -> StreamSensor:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    # -- subclass hooks ---------------------------------------------------

    def _open(self) -> None:
        """Acquire the device. Runs in the caller's thread, before reading."""

    def _close(self) -> None:
        """Release the device. Must unblock a pending :meth:`_poll`."""

    @abstractmethod
    def _poll(self) -> Measurement | None:
        """Block for at most a short timeout; return one sample or ``None``."""

    # -- internals --------------------------------------------------------

    def _run(self) -> None:
        buf = self._buf
        maxlen = buf.maxlen
        try:
            while not self._stopping.is_set():
                m = self._poll()
                if m is None:
                    continue
                if len(buf) == maxlen:
                    self.overruns += 1
                buf.append(m)
                self.received += 1
                if not self._first.is_set():
                    self._first.set()
        except BaseException as exc:
            # Closing the port from stop() makes a pending readline raise; that
            # is the shutdown path, not a failure.
            if not self._stopping.is_set():
                self._error = exc

    def _raise_if_dead(self) -> None:
        if self._error is not None:
            raise SensorError(f"{self.name}: reader thread stopped") from self._error
