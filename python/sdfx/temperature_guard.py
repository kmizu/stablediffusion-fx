"""Interruptible temperature hysteresis; hardware providers are separate."""
import math
import time
import subprocess
import json


def ohm_temperature(sensor_id):
    """Read an exact GPU sensor from a running OpenHardwareMonitor WMI service.

    No binaries are downloaded, loaded in-process, or started by this function.
    """
    if not isinstance(sensor_id, str) or not sensor_id.startswith(('/amdgpu/', '/nvidiagpu/')) or '/temperature/' not in sensor_id:
        raise ValueError('an explicit OpenHardwareMonitor GPU temperature sensor ID is required')
    command = ("$ErrorActionPreference='Stop'; "
               "Get-CimInstance -Namespace root/OpenHardwareMonitor -ClassName Sensor | "
               "Select-Object Identifier,SensorType,Value | ConvertTo-Json -Compress")
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
        check=True, capture_output=True, text=True, timeout=3, stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    rows = json.loads(result.stdout)
    if isinstance(rows, dict):
        rows = [rows]
    matches = [row for row in (rows or []) if row.get('Identifier') == sensor_id and row.get('SensorType') == 'Temperature']
    if len(matches) != 1 or matches[0].get('Value') is None or isinstance(matches[0].get('Value'), bool):
        raise ValueError('selected OpenHardwareMonitor temperature sensor is unavailable or ambiguous')
    value = float(matches[0]['Value'])
    # OHM's WMI wrapper encodes a missing nullable hardware reading as zero.
    if value == 0:
        raise ValueError('OpenHardwareMonitor temperature unavailable (zero sentinel)')
    if not math.isfinite(value) or not -50 <= value <= 150:
        raise ValueError('invalid GPU temperature reading')
    return value


def rocm_temperature(device=0):
    """Read the selected ROCm card's edge temperature; never substitute zero."""
    if isinstance(device, bool) or not isinstance(device, int) or device < 0:
        raise ValueError('GPU device index must be a nonnegative integer')
    result = subprocess.run(['rocm-smi', '--showtemp', '--json'], check=True,
                            capture_output=True, text=True, timeout=2,
                            stdin=subprocess.DEVNULL,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    data = json.loads(result.stdout)
    card = data.get(f'card{device}', {})
    reading = card.get('Temperature (Sensor edge) (C)')
    if reading is None or isinstance(reading, bool):
        raise ValueError('selected ROCm edge temperature sensor is unavailable')
    value = float(reading)
    if not math.isfinite(value) or not -50 <= value <= 150:
        raise ValueError('invalid GPU temperature reading')
    return value


def nvidia_temperature(device=0):
    """Query one explicitly selected physical NVIDIA GPU, with a bounded wait."""
    if isinstance(device, bool) or not isinstance(device, int) or device < 0:
        raise ValueError('GPU device index must be a nonnegative integer')
    result = subprocess.run(['nvidia-smi', f'--id={device}', '--query-gpu=temperature.gpu',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True,
                            text=True, timeout=2, stdin=subprocess.DEVNULL,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    value = float(result.stdout.strip())
    if not math.isfinite(value) or not -50 <= value <= 150:
        raise ValueError('invalid GPU temperature reading')
    return value


class TemperatureGuard:
    def __init__(self, read_temperature, sleep_temperature=80, wake_temperature=70,
                 interval=1, sleep_time=1, max_sleep=0, clock=time.monotonic, wait=time.sleep,
                 on_event=lambda kind, temperature: None):
        values = (sleep_temperature, wake_temperature, interval, sleep_time, max_sleep)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError('temperature guard settings must be finite')
        if interval < 0 or sleep_time <= 0 or max_sleep < 0:
            raise ValueError('invalid temperature guard timing')
        self.read = read_temperature
        self.hot, self.cool = sleep_temperature, wake_temperature
        self.interval, self.delay, self.limit = interval, sleep_time, max_sleep
        self.clock, self.wait = clock, wait
        self.on_event = on_event
        self.last = None

    def step(self, check_interrupt):
        check_interrupt()
        if self.interval == 0 or self.cool > self.hot:
            return
        now = self.clock()
        if self.last is not None and now - self.last < self.interval:
            return
        self.last = now
        temperature = self.read()
        if not math.isfinite(temperature):
            raise ValueError('temperature reading must be finite')
        if temperature <= self.hot:
            return
        started = now
        self.on_event('pause', temperature)
        timed_out = False
        while temperature >= self.cool:
            check_interrupt()
            if self.limit and self.clock() - started >= self.limit:
                timed_out = True
                break
            remaining = self.delay
            if self.limit:
                remaining = min(remaining, max(0, self.limit - (self.clock() - started)))
            while remaining > 1e-9:
                check_interrupt()
                chunk = min(remaining, 0.1)
                self.wait(chunk)
                remaining -= chunk
            check_interrupt()
            temperature = self.read()
            if not math.isfinite(temperature):
                raise ValueError('temperature reading must be finite')
            self.on_event('reading', temperature)
        self.on_event('timeout' if timed_out else 'resume', temperature)
        self.last = self.clock()
