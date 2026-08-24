"""Real-time PyQt viewer/logger for two potentiometer channels on an Arduino.

Reads CSV lines from the sketch in ``firmware/potentiometer_logger/``:

    <micros>,<a0>,<a1>

maps raw ADC counts to joint angles (A0 -> theta1, A1 -> theta2, degrees),
differentiates them to angular rates (deg/s), and plots both live.

Run:
    python notebooks/Potenctiometerlogging.py                 # pick the port in the GUI
    python notebooks/Potenctiometerlogging.py --port COM5
    python notebooks/Potenctiometerlogging.py --simulate      # no hardware needed

Not part of the ``erp`` package -- exploratory tooling, per CLAUDE.md.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

import pyqtgraph as pg  # noqa: E402
import serial  # noqa: E402
import serial.tools.list_ports  # noqa: E402
from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402

# A sample as it leaves a source: (t_seconds, adc0_counts, adc1_counts).
Sample = tuple[float, int, int]

HISTORY_CAPACITY = 120_000  # samples kept in RAM (~10 min at 200 Hz)
UI_PERIOD_MS = 33  # ~30 fps repaint, decoupled from the serial rate
MICROS_WRAP = 1 << 32


# --------------------------------------------------------------------------
# Signal processing
# --------------------------------------------------------------------------


@dataclass
class ChannelCal:
    """Linear map from raw ADC counts to a joint angle in degrees.

    ``adc_min``/``adc_max`` are the counts observed at the two mechanical
    endpoints of the joint; ``angle_min``/``angle_max`` are the angles those
    endpoints correspond to. Nothing here assumes the pot is wired in any
    particular direction -- swap the endpoints, or set ``invert``, if the
    angle runs backwards.
    """

    name: str
    adc_min: float = 0.0
    adc_max: float = 1023.0
    angle_min: float = 0.0
    angle_max: float = 90.0
    invert: bool = False
    clamp: bool = True

    def to_angle(self, adc: float) -> float:
        """Convert raw ADC counts to degrees."""
        span = self.adc_max - self.adc_min
        if abs(span) < 1e-9:
            return self.angle_min
        u = (adc - self.adc_min) / span
        if self.invert:
            u = 1.0 - u
        if self.clamp:
            u = min(1.0, max(0.0, u))
        return self.angle_min + u * (self.angle_max - self.angle_min)


class LowPass:
    """One-pole low-pass with a fixed cutoff, correct under variable dt.

    The usual ``y += alpha * (x - y)`` with a constant alpha changes its
    cutoff whenever the sample rate drifts. Deriving alpha from the actual
    dt of each sample keeps the cutoff where you asked for it, which matters
    because USB serial delivery is not evenly spaced.
    """

    def __init__(self, cutoff_hz: float) -> None:
        self.cutoff_hz = cutoff_hz
        self._y: float | None = None

    def reset(self) -> None:
        self._y = None

    def update(self, x: float, dt: float) -> float:
        """Filter one sample taken ``dt`` seconds after the previous one."""
        if self._y is None or dt <= 0.0 or self.cutoff_hz <= 0.0:
            self._y = x
            return x
        tau = 1.0 / (2.0 * math.pi * self.cutoff_hz)
        alpha = dt / (tau + dt)
        self._y += alpha * (x - self._y)
        return self._y


def lsq_slope(t: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of ``y`` against ``t`` (units of y per unit of t).

    Used instead of a two-point difference because a 10-bit pot quantises to
    ~0.1 deg steps; dividing one quantisation step by a 5 ms dt gives ~20
    deg/s of pure noise. Fitting a line across a window averages that down
    and, unlike a difference plus a smoother, does not assume uniform dt.
    """
    n = t.size
    if n < 2:
        return 0.0
    dt = t - t.mean()
    den = float(dt @ dt)
    if den <= 1e-12:
        return 0.0
    return float(dt @ (y - y.mean()) / den)


class RingBuffer:
    """Fixed-length history of floats with amortised O(1) append.

    Backed by an array of twice the capacity: appends run forward until it
    fills, then the newest half is copied down. That costs one memmove every
    ``capacity`` appends and keeps ``data()`` a contiguous view, which is
    what the plot widget wants.
    """

    def __init__(self, capacity: int) -> None:
        self._cap = capacity
        self._buf = np.empty(capacity * 2, dtype=np.float64)
        self._n = 0

    def append(self, value: float) -> None:
        if self._n == self._buf.size:
            self._buf[: self._cap] = self._buf[self._cap :]
            self._n = self._cap
        self._buf[self._n] = value
        self._n += 1

    def clear(self) -> None:
        self._n = 0

    def data(self) -> np.ndarray:
        """The most recent samples, oldest first."""
        return self._buf[max(0, self._n - self._cap) : self._n]


# --------------------------------------------------------------------------
# Sample sources
# --------------------------------------------------------------------------


class SerialSource:
    """Line-oriented reader for the Arduino CSV stream.

    Accepts either ``micros,a0,a1`` or a bare ``a0,a1``. With a device
    timestamp, dt comes from the Arduino's own clock and the derivative is
    honest; without one, dt is host arrival time, which USB buffering makes
    lumpy -- so prefer the 3-field form.
    """

    def __init__(self, port: str, baud: int) -> None:
        self.port = port
        self.baud = baud
        self._ser: serial.Serial | None = None
        self._buf = bytearray()
        self._prev_raw_us: int | None = None
        self._wraps = 0
        self._t0: float | None = None
        self.bad_lines = 0

    def open(self) -> None:
        self._ser = serial.Serial(self.port, self.baud, timeout=0.05)
        # Opening the port asserts DTR, which resets most Arduino boards.
        # Anything read before the bootloader hands over is garbage.
        time.sleep(2.0)
        self._ser.reset_input_buffer()
        self._buf.clear()

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    def read(self) -> list[Sample]:
        assert self._ser is not None
        chunk = self._ser.read(max(1, self._ser.in_waiting))
        if not chunk:
            return []
        self._buf.extend(chunk)

        out: list[Sample] = []
        while True:
            nl = self._buf.find(b"\n")
            if nl < 0:
                break
            line = bytes(self._buf[:nl])
            del self._buf[: nl + 1]
            parsed = self._parse(line)
            if parsed is not None:
                out.append(parsed)
        # A stream that never sends '\n' would otherwise grow without bound.
        if len(self._buf) > 4096:
            del self._buf[:-256]
        return out

    def _parse(self, line: bytes) -> Sample | None:
        parts = line.strip().split(b",")
        try:
            if len(parts) == 3:
                raw_us, a0, a1 = int(parts[0]), int(parts[1]), int(parts[2])
                t = self._unwrap(raw_us) * 1e-6
            elif len(parts) == 2:
                a0, a1 = int(parts[0]), int(parts[1])
                t = time.perf_counter()
            else:
                self.bad_lines += 1
                return None
        except ValueError:
            self.bad_lines += 1  # partial first line, or a stray Serial.print
            return None

        if self._t0 is None:
            self._t0 = t
        return (t - self._t0, a0, a1)

    def _unwrap(self, raw_us: int) -> float:
        """Undo the ~71.6 min rollover of the Arduino's ``micros()``."""
        if self._prev_raw_us is not None and raw_us < self._prev_raw_us - MICROS_WRAP // 2:
            self._wraps += 1
        self._prev_raw_us = raw_us
        return float(raw_us + self._wraps * MICROS_WRAP)


class SimSource:
    """Synthetic two-channel signal, for working on the GUI without hardware.

    Two out-of-phase sinusoids over the full 0..1023 range plus a couple of
    counts of noise, so the derivative pane shows something with a known
    shape (a cosine) to check the estimator against.
    """

    def __init__(self, rate_hz: float = 200.0) -> None:
        self.rate_hz = rate_hz
        self._t0 = 0.0
        self._next = 0.0
        self._rng = np.random.default_rng(0)

    def open(self) -> None:
        self._t0 = time.perf_counter()
        self._next = 0.0

    def close(self) -> None:
        pass

    def read(self) -> list[Sample]:
        now = time.perf_counter() - self._t0
        out: list[Sample] = []
        while self._next <= now:
            t = self._next
            a0 = 512 + 480 * math.sin(2 * math.pi * 0.25 * t)
            a1 = 512 + 350 * math.sin(2 * math.pi * 0.4 * t + 1.0)
            noise = self._rng.normal(0.0, 1.5, 2)
            out.append((t, int(round(a0 + noise[0])), int(round(a1 + noise[1]))))
            self._next += 1.0 / self.rate_hz
        if not out:
            time.sleep(0.002)
        return out


class ReaderThread(threading.Thread):
    """Pumps a source into a queue so the GUI thread never blocks on I/O."""

    def __init__(self, source: SerialSource | SimSource, out: queue.Queue) -> None:
        super().__init__(daemon=True)
        self.source = source
        self.out = out
        self.error: str | None = None
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            self.source.open()
        except Exception as exc:
            self.error = str(exc)
            return
        try:
            while not self._stop.is_set():
                for sample in self.source.read():
                    self.out.put(sample)
        except Exception as exc:
            self.error = str(exc)
        finally:
            try:
                self.source.close()
            except Exception:
                pass


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------

COLOR_T1 = "#4c8fd8"
COLOR_T2 = "#e08a3c"


class ChannelPanel(QtWidgets.QGroupBox):
    """Calibration controls for one channel."""

    changed = QtCore.pyqtSignal()

    def __init__(self, cal: ChannelCal, color: str) -> None:
        super().__init__(f"{cal.name} calibration")
        self.cal = cal
        self._learning = False
        self._seen_min = math.inf
        self._seen_max = -math.inf

        self.adc_min = QtWidgets.QDoubleSpinBox()
        self.adc_max = QtWidgets.QDoubleSpinBox()
        for box, val in ((self.adc_min, cal.adc_min), (self.adc_max, cal.adc_max)):
            box.setRange(0.0, 65535.0)
            box.setDecimals(0)
            box.setValue(val)

        self.angle_min = QtWidgets.QDoubleSpinBox()
        self.angle_max = QtWidgets.QDoubleSpinBox()
        for box, val in ((self.angle_min, cal.angle_min), (self.angle_max, cal.angle_max)):
            box.setRange(-3600.0, 3600.0)
            box.setDecimals(2)
            box.setSuffix(" deg")
            box.setValue(val)

        self.invert = QtWidgets.QCheckBox("Invert direction")
        self.invert.setChecked(cal.invert)
        self.clamp = QtWidgets.QCheckBox("Clamp to range")
        self.clamp.setChecked(cal.clamp)

        self.learn_btn = QtWidgets.QPushButton("Learn ADC range")
        self.learn_btn.setCheckable(True)
        self.learn_btn.setToolTip(
            "Press, sweep the joint through its full travel, press again.\n"
            "The min/max counts seen are written into the fields above."
        )

        self.readout = QtWidgets.QLabel("--")
        self.readout.setStyleSheet(f"color: {color}; font-family: monospace;")

        form = QtWidgets.QFormLayout(self)
        form.addRow("ADC at min", self.adc_min)
        form.addRow("ADC at max", self.adc_max)
        form.addRow("Angle at min", self.angle_min)
        form.addRow("Angle at max", self.angle_max)
        form.addRow(self.invert)
        form.addRow(self.clamp)
        form.addRow(self.learn_btn)
        form.addRow("Live", self.readout)

        for box in (self.adc_min, self.adc_max, self.angle_min, self.angle_max):
            box.valueChanged.connect(self._apply)
        self.invert.toggled.connect(self._apply)
        self.clamp.toggled.connect(self._apply)
        self.learn_btn.toggled.connect(self._toggle_learn)

    def _apply(self) -> None:
        self.cal.adc_min = self.adc_min.value()
        self.cal.adc_max = self.adc_max.value()
        self.cal.angle_min = self.angle_min.value()
        self.cal.angle_max = self.angle_max.value()
        self.cal.invert = self.invert.isChecked()
        self.cal.clamp = self.clamp.isChecked()
        self.changed.emit()

    def _toggle_learn(self, on: bool) -> None:
        self._learning = on
        if on:
            self._seen_min, self._seen_max = math.inf, -math.inf
            self.learn_btn.setText("Sweep the joint, then press again")
        else:
            self.learn_btn.setText("Learn ADC range")
            if self._seen_max > self._seen_min:
                self.adc_min.setValue(self._seen_min)
                self.adc_max.setValue(self._seen_max)

    def observe(self, adc: float, angle: float, rate: float) -> None:
        """Feed the newest sample in for the live readout and range learning."""
        if self._learning:
            self._seen_min = min(self._seen_min, adc)
            self._seen_max = max(self._seen_max, adc)
            self.readout.setText(f"{adc:6.0f} cts  [{self._seen_min:.0f}..{self._seen_max:.0f}]")
        else:
            self.readout.setText(f"{adc:6.0f} cts  {angle:8.2f} deg  {rate:9.2f} deg/s")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, port: str | None, baud: int, simulate: bool) -> None:
        super().__init__()
        self.setWindowTitle("Potentiometer logger - theta1 / theta2")
        self.resize(1280, 760)

        self.cal1 = ChannelCal("theta1 (A0)", angle_min=0.0, angle_max=90.0)
        self.cal2 = ChannelCal("theta2 (A1)", angle_min=0.0, angle_max=90.0)

        self.t = RingBuffer(HISTORY_CAPACITY)
        self.th1 = RingBuffer(HISTORY_CAPACITY)
        self.th2 = RingBuffer(HISTORY_CAPACITY)
        self.w1 = RingBuffer(HISTORY_CAPACITY)
        self.w2 = RingBuffer(HISTORY_CAPACITY)

        self.lpf1 = LowPass(5.0)
        self.lpf2 = LowPass(5.0)
        self._t_prev: float | None = None
        self._raw_hist: list[Sample] = []

        self.queue: queue.Queue = queue.Queue()
        self.reader: ReaderThread | None = None
        self._rate_marks: list[float] = []
        self._csv_file: object = None
        self._csv_writer: object = None

        self._build_ui(simulate)
        if port:
            idx = self.port_box.findText(port)
            if idx >= 0:
                self.port_box.setCurrentIndex(idx)
            else:
                self.port_box.setEditText(port)
        self.baud_box.setCurrentText(str(baud))

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(UI_PERIOD_MS)

    # -- construction ------------------------------------------------------

    def _build_ui(self, simulate: bool) -> None:
        pg.setConfigOptions(antialias=True, background="w", foreground="k")

        self.plot_angle = pg.PlotWidget()
        self.plot_angle.setLabel("left", "angle", units="deg")
        self.plot_angle.addLegend(offset=(-10, 10))
        self.plot_angle.showGrid(x=True, y=True, alpha=0.3)
        self.curve_th1 = self.plot_angle.plot(pen=pg.mkPen(COLOR_T1, width=2), name="theta1 (A0)")
        self.curve_th2 = self.plot_angle.plot(pen=pg.mkPen(COLOR_T2, width=2), name="theta2 (A1)")

        self.plot_rate = pg.PlotWidget()
        self.plot_rate.setLabel("left", "angular rate", units="deg/s")
        self.plot_rate.setLabel("bottom", "time", units="s")
        self.plot_rate.addLegend(offset=(-10, 10))
        self.plot_rate.showGrid(x=True, y=True, alpha=0.3)
        self.curve_w1 = self.plot_rate.plot(pen=pg.mkPen(COLOR_T1, width=2), name="dtheta1/dt")
        self.curve_w2 = self.plot_rate.plot(pen=pg.mkPen(COLOR_T2, width=2), name="dtheta2/dt")
        self.plot_rate.setXLink(self.plot_angle)

        for plot in (self.plot_angle, self.plot_rate):
            plot.setDownsampling(auto=True, mode="peak")
            plot.setClipToView(True)

        plots = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        plots.addWidget(self.plot_angle)
        plots.addWidget(self.plot_rate)
        plots.setSizes([400, 300])

        side = QtWidgets.QWidget()
        side.setFixedWidth(320)
        col = QtWidgets.QVBoxLayout(side)
        col.addWidget(self._build_connection_box(simulate))

        self.panel1 = ChannelPanel(self.cal1, COLOR_T1)
        self.panel2 = ChannelPanel(self.cal2, COLOR_T2)
        for panel in (self.panel1, self.panel2):
            panel.changed.connect(self._recompute)
            col.addWidget(panel)

        col.addWidget(self._build_filter_box())
        col.addWidget(self._build_view_box())
        col.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(False)
        scroll.setFixedWidth(345)

        root = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(root)
        row.addWidget(scroll)
        row.addWidget(plots, 1)
        self.setCentralWidget(root)

        self.status = self.statusBar()
        self.status.showMessage("Disconnected")

    def _build_connection_box(self, simulate: bool) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Connection")

        self.port_box = QtWidgets.QComboBox()
        self.port_box.setEditable(True)
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self._refresh_ports)

        self.baud_box = QtWidgets.QComboBox()
        self.baud_box.addItems(["9600", "57600", "115200", "230400", "250000", "500000"])
        self.baud_box.setCurrentText("115200")

        self.sim_box = QtWidgets.QCheckBox("Simulate (no hardware)")
        self.sim_box.setChecked(simulate)

        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.setCheckable(True)
        self.connect_btn.toggled.connect(self._toggle_connection)

        self.log_btn = QtWidgets.QPushButton("Start CSV log")
        self.log_btn.setCheckable(True)
        self.log_btn.toggled.connect(self._toggle_logging)

        port_row = QtWidgets.QHBoxLayout()
        port_row.addWidget(self.port_box, 1)
        port_row.addWidget(refresh)

        form = QtWidgets.QFormLayout(box)
        form.addRow("Port", port_row)
        form.addRow("Baud", self.baud_box)
        form.addRow(self.sim_box)
        form.addRow(self.connect_btn)
        form.addRow(self.log_btn)

        self._refresh_ports()
        return box

    def _build_filter_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Filtering / derivative")

        self.cutoff = QtWidgets.QDoubleSpinBox()
        self.cutoff.setRange(0.0, 200.0)
        self.cutoff.setDecimals(1)
        self.cutoff.setSingleStep(0.5)
        self.cutoff.setValue(5.0)
        self.cutoff.setSuffix(" Hz")
        self.cutoff.setToolTip("Low-pass on the angle before differentiating. 0 disables it.")

        self.deriv_window = QtWidgets.QSpinBox()
        self.deriv_window.setRange(5, 2000)
        self.deriv_window.setSingleStep(5)
        self.deriv_window.setValue(250)
        self.deriv_window.setSuffix(" ms")
        self.deriv_window.setToolTip(
            "Width of the least-squares fit used for d(theta)/dt.\n"
            "Rate noise falls as roughly window^-1.5, so widening this is the\n"
            "main lever on a jittery rate trace -- at the cost of lag."
        )

        self.lag_label = QtWidgets.QLabel()
        self.lag_label.setStyleSheet("color: #666;")

        for widget in (self.cutoff, self.deriv_window):
            widget.valueChanged.connect(self._recompute)
            widget.valueChanged.connect(self._update_lag_label)

        form = QtWidgets.QFormLayout(box)
        form.addRow("LPF cutoff", self.cutoff)
        form.addRow("Deriv. window", self.deriv_window)
        form.addRow("Rate lag", self.lag_label)
        self._update_lag_label()
        return box

    def _update_lag_label(self) -> None:
        """Show what the current smoothing costs in delay.

        Half the fit window (the fit is centred, but plotted at 'now') plus
        the one-pole group delay 1/(2*pi*fc).
        """
        lag = 0.5 * self.deriv_window.value() * 1e-3
        if self.cutoff.value() > 0:
            lag += 1.0 / (2.0 * math.pi * self.cutoff.value())
        self.lag_label.setText(f"~{lag * 1e3:.0f} ms behind")

    def _build_view_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("View")

        self.span = QtWidgets.QDoubleSpinBox()
        self.span.setRange(1.0, 600.0)
        self.span.setDecimals(1)
        self.span.setValue(15.0)
        self.span.setSuffix(" s")

        self.full_range = QtWidgets.QCheckBox("Angle: fix to calibrated range")
        self.full_range.setChecked(True)
        self.full_range.setToolTip(
            "Pin the angle axis to the range you calibrated, so you can see\n"
            "where in the joint's travel you are. Uncheck to zoom to the\n"
            "signal instead."
        )

        self.min_angle_span = QtWidgets.QDoubleSpinBox()
        self.min_angle_span.setRange(0.0, 360.0)
        self.min_angle_span.setValue(5.0)
        self.min_angle_span.setSuffix(" deg")

        self.min_rate_span = QtWidgets.QDoubleSpinBox()
        self.min_rate_span.setRange(0.0, 3600.0)
        self.min_rate_span.setValue(20.0)
        self.min_rate_span.setSuffix(" deg/s")

        for widget in (self.min_angle_span, self.min_rate_span):
            widget.setToolTip(
                "Floor on how far the axis will zoom in. Without one, a\n"
                "stationary sensor fills the plot with magnified ADC noise."
            )

        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.clicked.connect(self._clear)

        for widget in (self.span, self.min_angle_span, self.min_rate_span):
            widget.valueChanged.connect(self._redraw)
        self.full_range.toggled.connect(self._redraw)

        form = QtWidgets.QFormLayout(box)
        form.addRow("Time span", self.span)
        form.addRow(self.full_range)
        form.addRow("Min. angle span", self.min_angle_span)
        form.addRow("Min. rate span", self.min_rate_span)
        form.addRow(self.pause_btn)
        form.addRow(clear_btn)
        return box

    # -- connection --------------------------------------------------------

    def _refresh_ports(self) -> None:
        current = self.port_box.currentText()
        self.port_box.clear()
        for info in serial.tools.list_ports.comports():
            self.port_box.addItem(info.device, info.description)
            self.port_box.setItemData(
                self.port_box.count() - 1, f"{info.device} - {info.description}",
                QtCore.Qt.ToolTipRole,
            )
        if current:
            self.port_box.setEditText(current)

    def _toggle_connection(self, on: bool) -> None:
        if on:
            self._clear()
            if self.sim_box.isChecked():
                source: SerialSource | SimSource = SimSource()
                label = "simulated source"
            else:
                port = self.port_box.currentText().strip()
                if not port:
                    self.connect_btn.setChecked(False)
                    QtWidgets.QMessageBox.warning(self, "No port", "Pick a serial port first.")
                    return
                source = SerialSource(port, int(self.baud_box.currentText()))
                label = f"{port} @ {self.baud_box.currentText()} baud"
            self.reader = ReaderThread(source, self.queue)
            self.reader.start()
            self.connect_btn.setText("Disconnect")
            self.status.showMessage(f"Connected to {label}")
        else:
            if self.reader is not None:
                self.reader.stop()
                self.reader.join(timeout=2.0)
                self.reader = None
            self.connect_btn.setText("Connect")
            self.status.showMessage("Disconnected")

    def _toggle_logging(self, on: bool) -> None:
        if on:
            out_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"pot_log_{datetime.now():%Y%m%d_%H%M%S}.csv"
            self._csv_file = open(path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(
                ["t_s", "adc0", "adc1", "theta1_deg", "theta2_deg", "omega1_dps", "omega2_dps"]
            )
            self.log_btn.setText("Stop CSV log")
            self.status.showMessage(f"Logging to {path}")
        else:
            if self._csv_file is not None:
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None
            self.log_btn.setText("Start CSV log")

    def _clear(self) -> None:
        for buf in (self.t, self.th1, self.th2, self.w1, self.w2):
            buf.clear()
        self._raw_hist.clear()
        self.lpf1.reset()
        self.lpf2.reset()
        self._t_prev = None
        self._rate_marks.clear()
        self._redraw()

    # -- processing --------------------------------------------------------

    def _ingest(self, sample: Sample) -> None:
        """Map one raw sample to angles + rates and push it onto the history."""
        t, a0, a1 = sample
        dt = 0.0 if self._t_prev is None else t - self._t_prev
        self._t_prev = t

        th1 = self.lpf1.update(self.cal1.to_angle(a0), dt)
        th2 = self.lpf2.update(self.cal2.to_angle(a1), dt)

        self.t.append(t)
        self.th1.append(th1)
        self.th2.append(th2)

        # Slope over the trailing window. The buffers already hold this
        # sample, so the fit is centred on "now" minus half a window -- that
        # half-window lag is the price of not amplifying quantisation noise.
        ts = self.t.data()
        win = self.deriv_window.value() * 1e-3
        i = int(np.searchsorted(ts, ts[-1] - win, side="left"))
        self.w1.append(lsq_slope(ts[i:], self.th1.data()[i:]))
        self.w2.append(lsq_slope(ts[i:], self.th2.data()[i:]))

    def _recompute(self) -> None:
        """Re-derive the whole history after a calibration/filter change.

        Cheap enough at these buffer sizes, and it means the plot always
        shows what the *current* settings produce rather than a splice of
        old and new ones -- which is what you want while calibrating.
        """
        raw = list(self._raw_hist)
        for buf in (self.t, self.th1, self.th2, self.w1, self.w2):
            buf.clear()
        for lpf in (self.lpf1, self.lpf2):
            lpf.cutoff_hz = self.cutoff.value()
            lpf.reset()
        self._t_prev = None
        for sample in raw:
            self._ingest(sample)
        self._redraw()

    def _tick(self) -> None:
        new: list[Sample] = []
        try:
            while True:
                new.append(self.queue.get_nowait())
        except queue.Empty:
            pass

        if self.reader is not None and self.reader.error:
            err = self.reader.error
            self.reader = None
            self.connect_btn.setChecked(False)
            QtWidgets.QMessageBox.critical(self, "Serial error", err)
            return

        now = time.perf_counter()
        for sample in new:
            self._ingest(sample)
            self._raw_hist.append(sample)
            self._rate_marks.append(now)
            if self._csv_writer is not None:
                self._csv_writer.writerow(
                    [
                        f"{sample[0]:.6f}", sample[1], sample[2],
                        f"{self.th1.data()[-1]:.4f}", f"{self.th2.data()[-1]:.4f}",
                        f"{self.w1.data()[-1]:.4f}", f"{self.w2.data()[-1]:.4f}",
                    ]
                )

        if len(self._raw_hist) > HISTORY_CAPACITY:
            del self._raw_hist[:-HISTORY_CAPACITY]
        self._rate_marks = [m for m in self._rate_marks if now - m < 1.0]

        if new:
            self._update_readouts()
        if not self.pause_btn.isChecked():
            self._redraw()

    def _update_readouts(self) -> None:
        a0, a1 = self._raw_hist[-1][1], self._raw_hist[-1][2]
        self.panel1.observe(a0, self.th1.data()[-1], self.w1.data()[-1])
        self.panel2.observe(a1, self.th2.data()[-1], self.w2.data()[-1])
        if self.connect_btn.isChecked():
            msg = f"{len(self._rate_marks)} Hz   |   {len(self._raw_hist)} samples"
            if self._csv_file is not None:
                msg += f"   |   logging -> {Path(self._csv_file.name).name}"
            self.status.showMessage(msg)

    @staticmethod
    def _floored_range(values: np.ndarray, floor: float) -> tuple[float, float]:
        """Y limits that fit ``values`` but never zoom in tighter than ``floor``.

        Plain autoscaling is unusable for a sensor at rest: the visible
        spread is then just ADC noise, so the axis magnifies a few counts of
        jitter to full height and the trace looks like it is thrashing. The
        floor keeps a still signal reading as a flat line.
        """
        lo, hi = float(values.min()), float(values.max())
        mid = 0.5 * (lo + hi)
        half = max(0.5 * (hi - lo), 0.5 * floor)
        pad = 0.08 * half
        return mid - half - pad, mid + half + pad

    def _redraw(self) -> None:
        ts = self.t.data()
        th1, th2 = self.th1.data(), self.th2.data()
        w1, w2 = self.w1.data(), self.w2.data()
        # Copies, not ring-buffer views: the curves hold these until the next
        # redraw, and the buffers are written from under them meanwhile.
        self.curve_th1.setData(ts, th1.copy())
        self.curve_th2.setData(ts, th2.copy())
        self.curve_w1.setData(ts, w1.copy())
        self.curve_w2.setData(ts, w2.copy())
        if not ts.size:
            return

        t1 = float(ts[-1])
        t0 = max(0.0, t1 - self.span.value())
        self.plot_angle.setXRange(t0, t1, padding=0.0)

        # Scale Y over the visible window only -- otherwise one fast sweep
        # early on flattens everything that comes after it.
        vis = ts >= t0
        if self.full_range.isChecked():
            bounds = [
                self.cal1.angle_min, self.cal1.angle_max,
                self.cal2.angle_min, self.cal2.angle_max,
            ]
            lo, hi = min(bounds), max(bounds)
            pad = 0.04 * max(hi - lo, 1.0)
            self.plot_angle.setYRange(lo - pad, hi + pad, padding=0.0)
        else:
            both = np.concatenate((th1[vis], th2[vis]))
            self.plot_angle.setYRange(
                *self._floored_range(both, self.min_angle_span.value()), padding=0.0
            )
        rates = np.concatenate((w1[vis], w2[vis]))
        self.plot_rate.setYRange(
            *self._floored_range(rates, self.min_rate_span.value()), padding=0.0
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.connect_btn.setChecked(False)
        self.log_btn.setChecked(False)
        super().closeEvent(event)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", help="serial port, e.g. COM5 or /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--simulate", action="store_true", help="synthetic data, no hardware")
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(args.port, args.baud, args.simulate)
    win.show()
    if args.port or args.simulate:
        win.connect_btn.setChecked(True)
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
