#!/usr/bin/env python3
"""
jetson_tegrastats_gui.py
- Launch tegrastats on Jetson and receive/parse/display via SSH on PC
- Real-time graph display using PyQt5
- Graph visualization of power, CPU usage, GPU usage, RAM usage, temperature

Dependencies: paramiko, PyQt5, matplotlib, numpy
    pip install paramiko PyQt5 matplotlib numpy

Usage example:
    python jetson_tegrastats_gui.py --host 10.41.66.20 --user ryogayuzawa --interval 1000
"""

import argparse
import json
import os
import re
import sys
import time
import signal
import threading
from datetime import datetime
from getpass import getpass
from statistics import mean
from collections import deque
import queue

import paramiko
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QPushButton, QLineEdit, QSpinBox,
                             QTextEdit, QTabWidget, QGridLayout, QGroupBox, QCheckBox, QSlider,
                             QSizePolicy)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.animation as animation

# ----------------------------- Parser (same as original) -----------------------------

RE_TEMP_KV   = re.compile(r'(\w+)@([0-9.]+)C')
RE_POWER     = re.compile(r'(\w+)\s+(\d+)mW(?:/(\d+)mW)?')
RE_CPU_BLOCK = re.compile(r'CPU\s+\[([^\]]+)\]')
RE_CPU_ELEM  = re.compile(r'(\d+)%@(\d+)')
RE_GR3D      = re.compile(r'GR3D_FREQ\s+(\d+)%')
RE_RAM       = re.compile(r'RAM\s+(\d+)/(\d+)MB')
RE_SWAP      = re.compile(r'SWAP\s+(\d+)/(\d+)MB')
RE_LFB       = re.compile(r'\(lfb\s+([^)]+)\)')

def parse_cpu_detail(block: str):
    cores = []
    for tok in block.split(','):
        tok = tok.strip()
        m = RE_CPU_ELEM.fullmatch(tok)
        if m:
            cores.append({'util_pct': int(m.group(1)), 'freq_mhz': int(m.group(2))})
    return cores

def parse_tegrastats_line(line: str):
    line = line.strip()
    if not line:
        return None

    rec = {
        'ts': time.time(),
        'raw': line,
        'temps_C': {},
        'power_mW': {},
    }

    m = RE_RAM.search(line)
    if m:
        rec['ram_MB'] = {'used': int(m.group(1)), 'total': int(m.group(2))}

    m = RE_SWAP.search(line)
    if m:
        rec['swap_MB'] = {'used': int(m.group(1)), 'total': int(m.group(2))}

    m = RE_LFB.search(line)
    if m:
        rec['lfb'] = m.group(1)

    m = RE_CPU_BLOCK.search(line)
    if m:
        rec['cpu_cores'] = parse_cpu_detail(m.group(1))

    m = RE_GR3D.search(line)
    if m:
        rec['gr3d_freq_pct'] = int(m.group(1))

    for key, val in RE_TEMP_KV.findall(line):
        rec['temps_C'][key] = float(val)

    for key, cur, avg in RE_POWER.findall(line):
        rec['power_mW'][key] = {'inst': int(cur), 'avg': int(avg) if avg else None}

    return rec

# ----------------------------- SSH Worker Thread -----------------------------

class TegraStatsWorker(QThread):
    data_received = pyqtSignal(dict)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, host, user, password, port=22, interval=1000):
        super().__init__()
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.interval = interval
        self.client = None
        self.running = False

    def run(self):
        self.running = True
        self.status_changed.emit("Testing network connectivity...")
        
        import socket
        import time
        
        try:
            # Test basic connectivity first with more robust method
            try:
                sock = socket.create_connection((self.host, self.port), timeout=10)
                sock.close()
            except socket.timeout:
                self.error_occurred.emit(f"Connection timeout to {self.host}:{self.port}")
                return
            except socket.error as e:
                self.error_occurred.emit(f"Cannot reach {self.host}:{self.port} - {str(e)}")
                return
        
            self.status_changed.emit("Network OK, establishing SSH connection...")
            time.sleep(0.1)
            
            self.client = None
            try:
                self.client = paramiko.SSHClient()
                self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

                # Conservative connection settings for Windows
                self.client.connect(
                    hostname=self.host, 
                    port=self.port, 
                    username=self.user, 
                    password=self.password, 
                    timeout=20.0,
                    banner_timeout=60.0,
                    auth_timeout=60.0,
                    look_for_keys=False,
                    allow_agent=False,
                    disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']}
                )
                self.status_changed.emit(f"SSH connected: {self.user}@{self.host}")
                
            except paramiko.AuthenticationException:
                self.error_occurred.emit("Authentication failed")
                return
            except paramiko.SSHException as e:
                self.error_occurred.emit(f"SSH protocol error: {e}")
                return
            except socket.error as e:
                self.error_occurred.emit(f"Socket error during SSH: {e}")
                return
            except Exception as e:
                self.error_occurred.emit(f"SSH connection failed: {e}")
                return

            cmd = f"tegrastats --interval {self.interval}"
            self.status_changed.emit(f"Starting: {cmd}")

            try:
                stdin, stdout, stderr = self.client.exec_command(cmd, timeout=30)

                buf = ""
                ch = stdout.channel
                ch.settimeout(1.0)  # Short timeout for recv operations

                while self.running:
                    try:
                        if ch.recv_ready():
                            data = ch.recv(4096).decode('utf-8', errors='ignore')
                            if not data:
                                self.status_changed.emit("Remote connection terminated")
                                break
                            
                            data = data.replace('\r', '\n')
                            buf += data
                            while '\n' in buf:
                                line, buf = buf.split('\n', 1)
                                s = line.strip()
                                if not s:
                                    continue
                                rec = parse_tegrastats_line(s)
                                if rec:
                                    self.data_received.emit(rec)

                        if ch.exit_status_ready():
                            if buf.strip():
                                for s in buf.splitlines():
                                    rec = parse_tegrastats_line(s)
                                    if rec:
                                        self.data_received.emit(rec)
                            code = ch.recv_exit_status()
                            self.status_changed.emit(f"tegrastats terminated with code={code}")
                            break
                            
                    except socket.timeout:
                        # This is expected for recv operations, continue
                        pass
                    except Exception as e:
                        self.error_occurred.emit(f"Data reception error: {e}")
                        break

                    time.sleep(0.05)

            except Exception as e:
                self.error_occurred.emit(f"Command execution error: {e}")
        finally:
            if self.client:
                try:
                    self.client.close()
                except:
                    pass

    def stop(self):
        self.running = False
        if self.client:
            try:
                self.client.close()
            except:
                pass

# ----------------------------- Graph Widget -----------------------------

class GraphWidget(QWidget):
    def __init__(self, title, ylabel, max_points=100, smoothing_alpha=0.3, fixed_height=320):
        super().__init__()
        self.title = title
        self.ylabel = ylabel
        self.max_points = max_points
        self.fixed_height = fixed_height  # 固定高さ (ユーザー要望)
        
        # Smoothing parameters
        # Exponential Moving Average (EMA): s_t = a * x_t + (1-a) * s_{t-1}
        # O(1) per point per series, minimal CPU & memory.
        self.smoothing_alpha = smoothing_alpha  # 0 < a <= 1; closer to 1 = less smoothing
        self.enable_smoothing = True
        
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # 固定高さ設定（幅はリサイズ可 / 高さは一定）
        if self.fixed_height:
            self.canvas.setFixedHeight(self.fixed_height)
            self.setMinimumHeight(self.fixed_height + 10)
            size_policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            self.setSizePolicy(size_policy)
            self.canvas.setSizePolicy(size_policy)
        
        self.times = deque(maxlen=max_points)
        # Store raw values
        self.values_raw = {}
        # Store smoothed values (EMA)
        self.values_ema = {}
        
        self.setup_plot()

    def setup_plot(self):
        # Background and border for Task Manager style
        self.ax.set_facecolor('#18191c')
        self.figure.set_facecolor('#18191c')
        for spine in self.ax.spines.values():
            spine.set_color('#333')
            spine.set_linewidth(1.2)

        self.ax.set_title(self.title, fontsize=9, color='#e0e0e0', pad=6, fontweight='bold')
        self.ax.set_ylabel(self.ylabel, fontsize=10, color='#b0b0b0', labelpad=10)
        self.ax.grid(True, color='#333', alpha=0.35, linewidth=0.7)

        # X axis: small, light, few ticks
        self.ax.tick_params(axis='x', labelsize=8, colors='#888', rotation=0, length=0, pad=2)
        self.ax.tick_params(axis='y', labelsize=9, colors='#aaa', length=3, pad=2)

        # Remove top/right ticks
        self.ax.tick_params(top=False, right=False)

        # Add padding around plot (increase left/bottom to prevent tick label clipping)
        self.figure.subplots_adjust(left=0.13, right=0.98, top=0.98, bottom=0.22)

    def add_data(self, timestamp, **kwargs):
        self.times.append(timestamp)

        for key, value in kwargs.items():
            # Raw queue
            if key not in self.values_raw:
                self.values_raw[key] = deque(maxlen=self.max_points)
            self.values_raw[key].append(value)

            # EMA queue (same maxlen)
            if key not in self.values_ema:
                self.values_ema[key] = deque(maxlen=self.max_points)
                self.values_ema[key].append(value)  # first value = raw
            else:
                # Compute EMA incrementally
                prev = self.values_ema[key][-1]
                a = self.smoothing_alpha
                smoothed = a * value + (1 - a) * prev
                self.values_ema[key].append(smoothed)

    # --- Keep only the latest 30 seconds ---
        cutoff = timestamp - 30.0
        while self.times and self.times[0] < cutoff:
            self.times.popleft()
            for vq in self.values_raw.values():
                if vq:
                    vq.popleft()
            for vq in self.values_ema.values():
                if vq:
                    vq.popleft()

        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        self.setup_plot()

        if len(self.times) < 2:
            self.canvas.draw()
            return

        times_list = list(self.times)
        if len(times_list) > 0:
            t0 = times_list[0]
            x_plot = [t - t0 for t in times_list]
        else:
            x_plot = []

        value_source = self.values_ema if self.enable_smoothing else self.values_raw
        stats_texts = []
        for key, values_deque in value_source.items():
            if len(values_deque) > 0:
                values_list = list(values_deque)
                self.ax.plot(x_plot, values_list, label=key, linewidth=1.2)
                # Calculate Ave, Min, Max (1 decimal place)
                ave = np.mean(values_list) if values_list else 0
                vmin = np.min(values_list) if values_list else 0
                vmax = np.max(values_list) if values_list else 0
                stats_texts.append(f"{key}: Ave {ave:.1f}  Min {vmin:.1f}  Max {vmax:.1f}")

    # Show statistics at the top right of the graph (position always fixed)
        if stats_texts:
            stats_str = "\n".join(stats_texts)
            self.ax.text(0.99, 0.99, stats_str, transform=self.ax.transAxes,
                         fontsize=10, color="#e0e0e0", ha="right", va="top",
                         bbox=dict(facecolor="#222", alpha=0.7, edgecolor="#444", boxstyle="round,pad=0.3"))

    # X axis: 0-30, step 5
        self.ax.set_xlim(0, 30)
        xticks = list(range(0, 31, 5))
        self.ax.set_xticks(xticks)
        self.ax.set_xticklabels([str(x) for x in xticks])

    # --- Fixed Y axis: usage, temperature, and power graphs ---
        if self.title in ["CPU Core Usage", "GPU Usage", "RAM Usage"]:
            self.ax.set_ylim(-5, 105)
            yticks = list(range(0, 101, 20))
            self.ax.set_yticks(yticks)
            self.ax.set_yticklabels([str(y) for y in yticks])
        elif self.title == "Temperature":
            self.ax.set_ylim(-5, 105)
            yticks = list(range(0, 101, 20))
            self.ax.set_yticks(yticks)
            self.ax.set_yticklabels([str(y) for y in yticks])
        elif self.title == "Power Consumption":
            self.ax.set_ylim(-1, 31)
            yticks = list(range(0, 31, 5))
            self.ax.set_yticks(yticks)
            self.ax.set_yticklabels([str(y) for y in yticks])

        if value_source:
            if self.title == "CPU Core Usage":
                legend = self.ax.legend(ncol=2, loc='upper left', fontsize=9)
            else:
                legend = self.ax.legend()
            if legend is not None:
                legend.get_frame().set_facecolor('#18191c')  # Dark color
                legend.get_frame().set_edgecolor('none')
                legend.get_frame().set_alpha(0.2)  # High transparency
                for text in legend.get_texts():
                    text.set_color('#e0e0e0')

        self.canvas.draw()

    def set_smoothing(self, enabled: bool, alpha: float | None = None):
        """Enable/disable smoothing or adjust alpha.
        alpha: new smoothing factor (optional). Higher = follow raw more.
        """
        self.enable_smoothing = enabled
        if alpha is not None and 0 < alpha <= 1:
            self.smoothing_alpha = alpha
        # No need to recompute historical EMA for simplicity (low cost tradeoff).
        self.update_plot()

# ----------------------------- Connection Test Worker -----------------------------

class ConnectionTestWorker(QThread):
    connection_result = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, host, user, password):
        super().__init__()
        self.host = host
        self.user = user
        self.password = password
        
    def run(self):
        import socket
        import time
        
        try:
            # First test basic network connectivity with more robust approach
            self.connection_result.emit(False, f"Testing network connectivity to {self.host}...")
            time.sleep(0.1)  # Give UI time to update
            
            # Test TCP connection to port 22
            try:
                sock = socket.create_connection((self.host, 22), timeout=10)
                sock.close()
            except socket.timeout:
                self.connection_result.emit(False, f"Connection timeout to {self.host}:22")
                return
            except socket.error as e:
                self.connection_result.emit(False, f"Cannot reach {self.host}:22 - {str(e)}")
                return
            
            self.connection_result.emit(False, "Network connectivity OK, testing SSH...")
            time.sleep(0.1)
            
            # Now test SSH connection with more conservative settings
            client = None
            try:
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
                # Very conservative connection settings for Windows
                client.connect(
                    hostname=self.host,
                    port=22,
                    username=self.user,
                    password=self.password,
                    timeout=20.0,
                    banner_timeout=60.0,
                    auth_timeout=60.0,
                    look_for_keys=False,
                    allow_agent=False,
                    disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']}
                )
                
                self.connection_result.emit(False, "SSH connection OK, testing tegrastats...")
                time.sleep(0.1)
                
                # Test tegrastats command with simple approach
                stdin, stdout, stderr = client.exec_command("which tegrastats", timeout=10)
                
                # Read output with timeout
                stdout.channel.settimeout(10.0)
                result_bytes = stdout.read()
                result = result_bytes.decode('utf-8', errors='ignore') if result_bytes else ""
                
                # Get exit status
                exit_code = stdout.channel.recv_exit_status()
                
                if exit_code == 0 and 'tegrastats' in result:
                    self.connection_result.emit(True, "All tests passed! Connection successful.")
                else:
                    self.connection_result.emit(False, "tegrastats command not found on remote system")
                    
            except paramiko.AuthenticationException:
                self.connection_result.emit(False, "Authentication failed - check username/password")
            except paramiko.SSHException as e:
                self.connection_result.emit(False, f"SSH protocol error: {str(e)}")
            except socket.error as e:
                self.connection_result.emit(False, f"Socket error during SSH: {str(e)}")
            except Exception as e:
                self.connection_result.emit(False, f"SSH connection failed: {str(e)}")
            finally:
                if client:
                    try:
                        client.close()
                    except:
                        pass
                        
        except Exception as e:
            self.connection_result.emit(False, f"Unexpected error: {str(e)}")

# ----------------------------- Connection Dialog -----------------------------

class ConnectionDialog(QWidget):
    connection_established = pyqtSignal(str, str, str, int)  # host, user, password, interval

    def __init__(self):
        super().__init__()
        self.test_worker = None
        self._config_path = os.path.join(os.path.expanduser("~"), ".jetson_tegrastats_monitor.json")
        self.init_ui()
        self._apply_dark_theme()
        self._load_last_settings()

    def _apply_dark_theme(self):
        font_css = "font-family: 'Consolas', 'monospace';"
        palette_css = f"""
        QWidget {{ background-color: #1e1f23; color: #e0e0e0; {font_css} }}
        QGroupBox {{ border: 1px solid #3a3d41; border-radius: 6px; margin-top: 6px; padding: 6px; {font_css} }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; {font_css} }}
        QPushButton {{ background:#2d2f33; border:1px solid #4a4d52; border-radius:4px; padding:6px 10px; {font_css} }}
        QPushButton:hover {{ background:#3a3d41; }}
        QPushButton:pressed {{ background:#2a2c2f; }}
        QLineEdit, QSpinBox, QTextEdit {{ background:#2a2c30; border:1px solid #4a4d52; border-radius:4px; {font_css} }}
        QTabBar::tab {{ background:#2d2f33; padding:6px 12px; border:1px solid #3a3d41; border-bottom:none; {font_css} }}
        QTabBar::tab:selected {{ background:#3a3d41; }}
        QScrollBar:vertical {{ background:#2a2c30; width:12px; }}
        QSlider::groove:horizontal {{ height:6px; background:#3a3d41; border-radius:3px; }}
        QSlider::handle:horizontal {{ background:#5c97ff; width:14px; margin:-4px 0; border-radius:7px; }}
        QLabel#statusLabel {{ background:#2a2c30; {font_css} }}
        """
        self.setStyleSheet(palette_css)

    def init_ui(self):
        self.setWindowTitle("Jetson TegraStats Monitor - Connection")
        # Enlarged initial dialog window size
        self.setGeometry(300, 240, 640, 420)
        self.setFixedSize(640, 420)
        self.setWindowTitle("Jetson TegraStats Monitor - Connection")
        # Enlarged initial dialog window size
        self.setGeometry(300, 240, 640, 420)
        self.setFixedSize(640, 420)

        layout = QVBoxLayout()

        # Title
        title_label = QLabel("Jetson TegraStats Monitor")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        layout.addWidget(QLabel(""))  # Spacer

        # Connection form
        form_layout = QGridLayout()
        form_layout.addWidget(QLabel("Host:"), 0, 0)
        self.host_edit = QLineEdit("")
        form_layout.addWidget(self.host_edit, 0, 1)

        form_layout.addWidget(QLabel("User:"), 1, 0)
        self.user_edit = QLineEdit("")
        form_layout.addWidget(self.user_edit, 1, 1)

        form_layout.addWidget(QLabel("Password:"), 2, 0)
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.returnPressed.connect(self.connect_clicked)
        form_layout.addWidget(self.password_edit, 2, 1)

        form_layout.addWidget(QLabel("Interval (ms):"), 3, 0)
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(100, 10000)
        self.interval_spin.setValue(1000)
        form_layout.addWidget(self.interval_spin, 3, 1)

        layout.addLayout(form_layout)
        layout.addWidget(QLabel(""))  # Spacer

        # Status and buttons
        self.status_label = QLabel("Enter connection details and click Connect")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        button_layout = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_clicked)
        self.connect_btn.setDefault(True)
        button_layout.addWidget(self.connect_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.close)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        if self.host_edit.text() and self.user_edit.text():
            self.password_edit.setFocus()

    def connect_clicked(self):
        host = self.host_edit.text().strip()
        user = self.user_edit.text().strip()
        password = self.password_edit.text()
        interval = self.interval_spin.value()

        if not host:
            self.status_label.setText("Please enter host address")
            self.host_edit.setFocus()
            return
        if not user:
            self.status_label.setText("Please enter username")
            self.user_edit.setFocus()
            return
        if not password:
            self.status_label.setText("Please enter password")
            self.password_edit.setFocus()
            return

        self.status_label.setText("Testing connection...")
        self.connect_btn.setEnabled(False)

        self.test_worker = ConnectionTestWorker(host, user, password)
        self.test_worker.connection_result.connect(self.on_connection_result)
        self.test_worker.start()

    def on_connection_result(self, success, message):
        self.status_label.setText(message)
        if success:
            host = self.host_edit.text().strip()
            user = self.user_edit.text().strip()
            password = self.password_edit.text()
            interval = self.interval_spin.value()
            self._save_last_settings(host, user, interval)
            QTimer.singleShot(1000, lambda: self.emit_and_close(host, user, password, interval))
        else:
            if not message.startswith("Testing") and not message.startswith("Network") and not message.startswith("SSH"):
                self.connect_btn.setEnabled(True)
        if self.test_worker and self.test_worker.isFinished():
            self.test_worker.wait()
            self.test_worker = None

    def emit_and_close(self, host, user, password, interval):
        self.connection_established.emit(host, user, password, interval)
        QTimer.singleShot(100, self.close)

    def _load_last_settings(self):
        try:
            if os.path.isfile(self._config_path):
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                host = data.get('host', '')
                user = data.get('user', '')
                interval = data.get('interval')
                if host:
                    self.host_edit.setText(host)
                if user:
                    self.user_edit.setText(user)
                if isinstance(interval, int):
                    self.interval_spin.setValue(interval)
        except Exception:
            pass

    def _save_last_settings(self, host, user, interval):
        try:
            data = {"host": host, "user": user, "interval": int(interval)}
            with open(self._config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception:
            pass

# ----------------------------- Main Monitor Window -----------------------------

class TegraStatsMonitor(QMainWindow):
    def __init__(self, host, user, password, interval):
        super().__init__()
        self.host = host
        self.user = user
        self.password = password
        self.interval = interval
        self.worker = None
        self.current_theme = 'dark'
        # intervalはmsなので、30秒分のデータ点数を計算
        self._graph_max_points = max(30_000 // max(1, self.interval), 10)  # 最低10点は確保
        self.init_ui()
        self.start_monitoring()
        
    def init_ui(self):
        self.setWindowTitle(f"Jetson TegraStats Monitor - {self.user}@{self.host}")
        # Set fixed window size (e.g., 1400x900) and disable resizing
        self.setFixedSize(1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)  # Add more space around the edges
        central_widget.setLayout(layout)
        # (Optional) top style
        self.apply_theme(self.current_theme)

        # Current values display
        current_group = QGroupBox("Current Status")
        current_layout = QGridLayout()

        self.power_label = QLabel("Power: -")
        self.gpu_label = QLabel("GPU: -")
        self.ram_label = QLabel("RAM: -")
        self.swap_label = QLabel("SWAP: -")
        self.temp_label = QLabel("Temperature: -")

        # CPU core labels (6 cores)
        self.cpu_labels = []
        for i in range(6):
            label = QLabel(f"CPU{i}: -")
            self.cpu_labels.append(label)

        font = QFont()
        font.setPointSize(10)
        font.setBold(True)

        # Arrange labels in grid
        current_layout.addWidget(self.power_label, 0, 0)
        current_layout.addWidget(self.gpu_label, 0, 1)
        current_layout.addWidget(self.ram_label, 0, 2)
        current_layout.addWidget(self.swap_label, 0, 3)
        current_layout.addWidget(self.temp_label, 0, 4)

        # CPU cores in second row
        for i, label in enumerate(self.cpu_labels):
            label.setFont(font)
            current_layout.addWidget(label, 1, i)

        for label in [self.power_label, self.gpu_label, self.ram_label, self.swap_label, self.temp_label]:
            label.setFont(font)

        current_group.setLayout(current_layout)
        layout.addWidget(current_group)

        # 5 Graphs in 2x2+1 grid
        mp = self._graph_max_points
        self.power_graph = GraphWidget("Power Consumption", "Power (W)", max_points=mp, fixed_height=260)
        self.cpu_graph = GraphWidget("CPU Core Usage", "Usage (%)", max_points=mp, fixed_height=260)
        self.gpu_graph = GraphWidget("GPU Usage", "Usage (%)", max_points=mp, fixed_height=260)
        self.ram_graph = GraphWidget("RAM Usage", "Usage (%)", max_points=mp, fixed_height=260)
        self.temp_graph = GraphWidget("Temperature", "Temperature (°C)", max_points=mp, fixed_height=260)

        # Set all graph widgets to expanding size policy
        for graph in [self.power_graph, self.cpu_graph, self.gpu_graph, self.ram_graph, self.temp_graph]:
            graph.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        graph_grid = QGridLayout()
        graph_grid.setSpacing(12)
        graph_grid.setContentsMargins(12, 12, 12, 12)  # Add space around the graphs
        # 2x2 grid
        graph_grid.addWidget(self.power_graph, 0, 0)
        graph_grid.addWidget(self.cpu_graph, 0, 1)
        graph_grid.addWidget(self.gpu_graph, 1, 0)
        graph_grid.addWidget(self.ram_graph, 1, 1)
        # Last graph centered below
        graph_grid.addWidget(self.temp_graph, 2, 0, 1, 2)

        # Set stretch so graphs expand and fill space
        graph_grid.setRowStretch(0, 1)
        graph_grid.setRowStretch(1, 1)
        graph_grid.setRowStretch(2, 2)  # Give more space to the temperature graph
        graph_grid.setColumnStretch(0, 1)
        graph_grid.setColumnStretch(1, 1)

        layout.addLayout(graph_grid)
        # Add extra stretch to create bottom margin
        layout.addStretch(1)
    # ---------------- Theme & UI Enhancements -----------------
    def toggle_theme(self):
        self.current_theme = 'light' if self.current_theme == 'dark' else 'dark'
        self.apply_theme(self.current_theme)
        self.theme_btn.setText("Switch Dark Theme" if self.current_theme == 'light' else "Switch Light Theme")

    def apply_theme(self, theme: str):
        # Consolas font for all widgets
        font_css = "font-family: 'Consolas', 'monospace';"
        if theme == 'dark':
            palette_css = f"""
            QWidget {{ background-color: #1e1f23; color: #e0e0e0; {font_css} }}
            QGroupBox {{ border: 1px solid #3a3d41; border-radius: 6px; margin-top: 6px; padding: 6px; {font_css} }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; {font_css} }}
            QPushButton {{ background:#2d2f33; border:1px solid #4a4d52; border-radius:4px; padding:6px 10px; {font_css} }}
            QPushButton:hover {{ background:#3a3d41; }}
            QPushButton:pressed {{ background:#2a2c2f; }}
            QLineEdit, QSpinBox, QTextEdit {{ background:#2a2c30; border:1px solid #4a4d52; border-radius:4px; {font_css} }}
            QTabBar::tab {{ background:#2d2f33; padding:6px 12px; border:1px solid #3a3d41; border-bottom:none; {font_css} }}
            QTabBar::tab:selected {{ background:#3a3d41; }}
            QScrollBar:vertical {{ background:#2a2c30; width:12px; }}
            QSlider::groove:horizontal {{ height:6px; background:#3a3d41; border-radius:3px; }}
            QSlider::handle:horizontal {{ background:#5c97ff; width:14px; margin:-4px 0; border-radius:7px; }}
            QLabel#statusLabel {{ background:#2a2c30; {font_css} }}
            """
        else:
            palette_css = f"""
            QWidget {{ background:#f6f7fb; color:#202124; {font_css} }}
            QGroupBox {{ border:1px solid #c9ccd1; border-radius:6px; margin-top:6px; padding:6px; {font_css} }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding:0 4px; {font_css} }}
            QPushButton {{ background:#ffffff; border:1px solid #c9ccd1; border-radius:4px; padding:6px 10px; {font_css} }}
            QPushButton:hover {{ background:#e9eef4; }}
            QPushButton:pressed {{ background:#dde3ea; }}
            QLineEdit, QSpinBox, QTextEdit {{ background:#ffffff; border:1px solid #c9ccd1; border-radius:4px; {font_css} }}
            QTabBar::tab {{ background:#e2e6ea; padding:6px 12px; border:1px solid #c9ccd1; border-bottom:none; {font_css} }}
            QTabBar::tab:selected {{ background:#ffffff; }}
            QSlider::groove:horizontal {{ height:6px; background:#d0d4d9; border-radius:3px; }}
            QSlider::handle:horizontal {{ background:#4285f4; width:14px; margin:-4px 0; border-radius:7px; }}
            QLabel#statusLabel {{ background:#eceff3; {font_css} }}
            """
        self.setStyleSheet(palette_css)

    def on_smoothing_changed(self):
        enabled = self.smooth_check.isChecked()
        alpha = self.smooth_slider.value() / 100.0
        graphs = [self.power_graph, self.cpu_graph, self.gpu_graph, self.ram_graph, self.temp_graph]
        for g in graphs:
            g.set_smoothing(enabled, alpha)

    def start_monitoring(self):
        self.worker = TegraStatsWorker(self.host, self.user, self.password, interval=self.interval)
        self.worker.data_received.connect(self.update_data)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.start()
        
    def update_data(self, data):
        timestamp = data['ts']
        
        # Update current values
        p = data.get('power_mW', {})
        vdd_in = p.get('VDD_IN', {})
        power_w = vdd_in.get('inst', 0) / 1000.0 if 'inst' in vdd_in else 0
        self.power_label.setText(f"Power: {power_w:.2f}W")
        
        cores = data.get('cpu_cores', [])
        
        # Update individual CPU core labels
        for i in range(6):
            if i < len(cores):
                core_util = cores[i]['util_pct']
                core_freq = cores[i]['freq_mhz']
                self.cpu_labels[i].setText(f"CPU{i}: {core_util}%@{core_freq}MHz")
            else:
                self.cpu_labels[i].setText(f"CPU{i}: -")
        
        cpu_avg = mean([c['util_pct'] for c in cores]) if cores else 0
        
        gpu_util = data.get('gr3d_freq_pct', 0)
        self.gpu_label.setText(f"GPU: {gpu_util}%")
        
        ram = data.get('ram_MB', {})
        ram_used = ram.get('used', 0)
        ram_total = ram.get('total', 1)
        ram_pct = (ram_used / ram_total) * 100 if ram_total > 0 else 0
        self.ram_label.setText(f"RAM: {ram_used}/{ram_total}MB ({ram_pct:.1f}%)")

        swap = data.get('swap_MB', {})
        swap_used = swap.get('used', 0)
        swap_total = swap.get('total', 1)
        swap_pct = (swap_used / swap_total) * 100 if swap_total > 0 else 0
        self.swap_label.setText(f"SWAP: {swap_used}/{swap_total}MB ({swap_pct:.1f}%)")
        
        temps = data.get('temps_C', {})
        temp_strs = []
        for key in ['cpu', 'gpu', 'tj']:
            if key in temps:
                if key == 'cpu':
                    display_name = 'CPU'
                elif key == 'gpu':
                    display_name = 'GPU'
                elif key == 'tj':
                    display_name = 'Thermal Junction'
                else:
                    display_name = key
                temp_strs.append(f"{display_name}:{temps[key]:.1f}°C")
        self.temp_label.setText(f"Temperature: {' '.join(temp_strs)}")
        
        # Update graphs
        if power_w > 0:
            self.power_graph.add_data(timestamp, Power=power_w)
            
        # Add individual CPU core data to graph
        if cores:
            cpu_data = {}
            for i, core in enumerate(cores):
                if i < 6:  # Limit to 6 cores
                    cpu_data[f"CPU{i}"] = core['util_pct']
            if cpu_data:
                self.cpu_graph.add_data(timestamp, **cpu_data)
            
        if gpu_util is not None:
            self.gpu_graph.add_data(timestamp, GPU=gpu_util)
            
        if ram_total > 0:
            self.ram_graph.add_data(timestamp, RAM_Usage=ram_pct)
            
        if temps:
            temp_data = {}
            for key, val in temps.items():
                if key in ['cpu', 'gpu', 'tj']:
                    display_name = "Thermal Junction" if key == "tj" else key
                    temp_data[display_name] = val
            if temp_data:
                self.temp_graph.add_data(timestamp, **temp_data)
        
    # ...Log出力を削除...
            
    def handle_error(self, error):
        # エラーは何もしない（または必要なら別途UIに表示）
        pass
        
    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        event.accept()

def build_argparser():
    p = argparse.ArgumentParser(description="Jetson tegrastats GUI Monitor")
    p.add_argument("--host", default="192.168.1.100", help="Jetson IP address or hostname")
    p.add_argument("--user", default="default", help="SSH username")
    p.add_argument("--interval", type=int, default=1000, help="tegrastats interval in milliseconds")
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    
    app = QApplication(sys.argv)
    
    # Global reference to monitor window to prevent garbage collection
    monitor_window = None
    
    def on_connection_established(host, user, password, interval):
        global monitor_window
        # Create and show monitor window
        monitor_window = TegraStatsMonitor(host, user, password, interval)
        monitor_window.show()
        # Bring window to front
        monitor_window.raise_()
        monitor_window.activateWindow()
    
    # Show connection dialog first
    connection_dialog = ConnectionDialog()
    
    # Set default values from command line arguments
    connection_dialog.host_edit.setText(args.host)
    connection_dialog.user_edit.setText(args.user)
    connection_dialog.interval_spin.setValue(args.interval)
    
    connection_dialog.connection_established.connect(on_connection_established)
    connection_dialog.show()
    
    sys.exit(app.exec_())
