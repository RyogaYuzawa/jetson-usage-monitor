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
                             QTextEdit, QTabWidget, QGridLayout, QGroupBox)
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
    def __init__(self, title, ylabel, max_points=100):
        super().__init__()
        self.title = title
        self.ylabel = ylabel
        self.max_points = max_points
        
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        self.times = deque(maxlen=max_points)
        self.values = {}
        
        self.setup_plot()

    def setup_plot(self):
        self.ax.set_title(self.title)
        self.ax.set_ylabel(self.ylabel)
        self.ax.grid(True, alpha=0.3)
        self.ax.tick_params(axis='x', rotation=45)
        self.figure.tight_layout()

    def add_data(self, timestamp, **kwargs):
        self.times.append(timestamp)
        
        for key, value in kwargs.items():
            if key not in self.values:
                self.values[key] = deque(maxlen=self.max_points)
            self.values[key].append(value)
        
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        self.setup_plot()
        
        if len(self.times) < 2:
            self.canvas.draw()
            return
            
        times_list = list(self.times)
        time_labels = [datetime.fromtimestamp(t).strftime('%H:%M:%S') for t in times_list]
        
        for key, values_deque in self.values.items():
            if len(values_deque) > 0:
                values_list = list(values_deque)
                self.ax.plot(range(len(values_list)), values_list, label=key, marker='o', markersize=2)
        
        if len(time_labels) > 1:
            step = max(1, len(time_labels) // 10)
            tick_positions = range(0, len(time_labels), step)
            tick_labels = [time_labels[i] for i in tick_positions]
            self.ax.set_xticks(tick_positions)
            self.ax.set_xticklabels(tick_labels)
        
        if self.values:
            self.ax.legend()
        
        self.canvas.draw()

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
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Jetson TegraStats Monitor - Connection")
        self.setGeometry(300, 300, 400, 300)
        self.setFixedSize(400, 300)
        
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
        self.host_edit = QLineEdit("10.41.66.20")
        form_layout.addWidget(self.host_edit, 0, 1)
        
        form_layout.addWidget(QLabel("User:"), 1, 0)
        self.user_edit = QLineEdit("ryogayuzawa")
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
        self.status_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; }")
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
        
        # Focus on password field if other fields are filled
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
        
        # Start connection test in background thread
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
            # Give user a moment to see the success message
            QTimer.singleShot(1000, lambda: self.emit_and_close(host, user, password, interval))
        else:
            # Only re-enable button if it's a final failure (not progress update)
            if not message.startswith("Testing") and not message.startswith("Network") and not message.startswith("SSH"):
                self.connect_btn.setEnabled(True)
        
        if self.test_worker and self.test_worker.isFinished():
            self.test_worker.wait()
            self.test_worker = None
    
    def emit_and_close(self, host, user, password, interval):
        # Emit the signal before closing
        self.connection_established.emit(host, user, password, interval)
        # Close the dialog after a short delay to ensure signal is processed
        QTimer.singleShot(100, self.close)

# ----------------------------- Main Monitor Window -----------------------------

class TegraStatsMonitor(QMainWindow):
    def __init__(self, host, user, password, interval):
        super().__init__()
        self.host = host
        self.user = user
        self.password = password
        self.interval = interval
        self.worker = None
        self.init_ui()
        self.start_monitoring()
        
    def init_ui(self):
        self.setWindowTitle(f"Jetson TegraStats Monitor - {self.user}@{self.host}")
        self.setGeometry(100, 100, 1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Current values display
        current_group = QGroupBox("Current Status")
        current_layout = QGridLayout()
        
        self.power_label = QLabel("Power: -")
        self.gpu_label = QLabel("GPU: -")
        self.ram_label = QLabel("RAM: -")
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
        current_layout.addWidget(self.temp_label, 0, 3)
        
        # CPU cores in second row
        for i, label in enumerate(self.cpu_labels):
            label.setFont(font)
            current_layout.addWidget(label, 1, i)
        
        for label in [self.power_label, self.gpu_label, self.ram_label, self.temp_label]:
            label.setFont(font)
        
        current_group.setLayout(current_layout)
        layout.addWidget(current_group)
        
        # Graph tabs
        self.tab_widget = QTabWidget()
        
        # Power graph
        self.power_graph = GraphWidget("Power Consumption", "Power (W)")
        self.tab_widget.addTab(self.power_graph, "Power")
        
        # CPU cores graph
        self.cpu_graph = GraphWidget("CPU Core Usage", "Usage (%)")
        self.tab_widget.addTab(self.cpu_graph, "CPU Cores")
        
        # GPU graph
        self.gpu_graph = GraphWidget("GPU Usage", "Usage (%)")
        self.tab_widget.addTab(self.gpu_graph, "GPU")
        
        # RAM graph
        self.ram_graph = GraphWidget("RAM Usage", "Usage (%)")
        self.tab_widget.addTab(self.ram_graph, "RAM")
        
        # Temperature graph
        self.temp_graph = GraphWidget("Temperature", "Temperature (°C)")
        self.tab_widget.addTab(self.temp_graph, "Temperature")
        
        layout.addWidget(self.tab_widget)
        
        # Log display
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

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
        
        temps = data.get('temps_C', {})
        temp_strs = []
        for key in ['cpu', 'gpu', 'tj']:
            if key in temps:
                display_name = "Thermal Junction" if key == "tj" else key
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
        
        # Add log entry
        time_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
        core_strs = []
        for i, core in enumerate(cores[:6]):  # Show first 6 cores
            core_strs.append(f"C{i}:{core['util_pct']}%")
        cpu_detail = " ".join(core_strs) if core_strs else "CPU:-"
        
        log_msg = f"[{time_str}] P:{power_w:.2f}W {cpu_detail} GPU:{gpu_util}% RAM:{ram_pct:.1f}%"
        self.log_text.append(log_msg)
        
        # Limit log length
        if self.log_text.document().blockCount() > 100:
            cursor = self.log_text.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()
            
    def handle_error(self, error):
        # Log error to the log area instead of status label
        time_str = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{time_str}] ERROR: {error}")
        
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
