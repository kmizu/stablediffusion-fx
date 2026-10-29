from sdfx.temperature_guard import TemperatureGuard
import pytest


@pytest.mark.parametrize('value', [True, False, None, 'N/A', 'NaN'])
def test_ohm_rejects_unavailable_or_non_numeric_values(monkeypatch, value):
    import json
    from types import SimpleNamespace
    from sdfx import temperature_guard as module
    row = dict(Identifier='/amdgpu/0/temperature/0', SensorType='Temperature', Value=value)
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=json.dumps(row)))
    with pytest.raises(ValueError):
        module.ohm_temperature('/amdgpu/0/temperature/0')


@pytest.mark.parametrize('settings', [dict(interval=-1), dict(sleep_time=0), dict(max_sleep=-1),
    dict(sleep_temperature=float('nan')), dict(wake_temperature=float('inf'))])
def test_invalid_settings_fail_before_sensor_access(settings):
    with pytest.raises(ValueError):
        TemperatureGuard(lambda: pytest.fail('sensor must not be queried'), **settings)


def test_wake_above_sleep_disables_temperature_queries():
    guard = TemperatureGuard(lambda: pytest.fail('disabled sensor was queried'),
                             sleep_temperature=70, wake_temperature=80)
    guard.step(lambda: None)


def test_sensor_failure_is_not_replaced_with_cool_temperature():
    def failed():
        raise OSError('monitor unavailable')
    guard = TemperatureGuard(failed)
    with pytest.raises(OSError, match='monitor unavailable'):
        guard.step(lambda: None)


def test_openhardwaremonitor_requires_exact_temperature_sensor(monkeypatch):
    from sdfx import temperature_guard as module
    from types import SimpleNamespace
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='[{"Identifier":"/amdgpu/0/temperature/0","SensorType":"Temperature","Value":55},{"Identifier":"/intelcpu/0/temperature/0","SensorType":"Temperature","Value":70}]'))
    assert module.ohm_temperature('/amdgpu/0/temperature/0') == 55
    with pytest.raises(ValueError, match='sensor'):
        module.ohm_temperature('/amdgpu/1/temperature/0')
    with pytest.raises(ValueError, match='GPU'):
        module.ohm_temperature('/intelcpu/0/temperature/0')
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='{"Identifier":"/amdgpu/0/temperature/0","SensorType":"Temperature","Value":0}'))
    with pytest.raises(ValueError, match='unavailable'):
        module.ohm_temperature('/amdgpu/0/temperature/0')


def test_reports_pause_resume_and_timeout():
    readings = iter([90, 60])
    events = []
    guard = TemperatureGuard(lambda: next(readings), wait=lambda _: None,
                             on_event=lambda kind, value: events.append((kind, value)))
    guard.step(lambda: None)
    assert events == [('pause', 90), ('reading', 60), ('resume', 60)]
    now = [0.0]
    events.clear()
    def wait(seconds):
        now[0] += seconds
    guard = TemperatureGuard(lambda: 90, sleep_time=1, max_sleep=0.2,
        clock=lambda: now[0], wait=wait, on_event=lambda kind, value: events.append((kind, value)))
    guard.step(lambda: None)
    assert events[-1] == ('timeout', 90)


def test_rocm_selects_requested_device_and_edge_sensor(monkeypatch):
    from sdfx import temperature_guard as module
    from types import SimpleNamespace
    calls = []
    def run(command, **options):
        calls.append(command)
        return SimpleNamespace(stdout='{"card1":{"Temperature (Sensor edge) (C)":"54.5"},"card0":{"Temperature (Sensor edge) (C)":"40"}}')
    monkeypatch.setattr(module.subprocess, 'run', run)
    assert module.rocm_temperature(1) == 54.5
    assert '--json' in calls[0]
    with pytest.raises(ValueError, match='sensor'):
        module.rocm_temperature(2)


@pytest.mark.parametrize('reading', [float('nan'), float('inf')])
def test_invalid_readings_stop_processing(reading):
    guard = TemperatureGuard(lambda: reading)
    with pytest.raises(ValueError, match='reading'):
        guard.step(lambda: None)


def test_nvidia_provider_selects_one_device_and_validates_output(monkeypatch):
    from sdfx import temperature_guard as module
    from types import SimpleNamespace
    calls = []
    def run(command, **options):
        calls.append((command, options))
        return SimpleNamespace(stdout='42\n')
    monkeypatch.setattr(module.subprocess, 'run', run)
    assert module.nvidia_temperature(2) == 42
    assert '--id=2' in calls[0][0]
    assert calls[0][1]['timeout'] == 2
    with pytest.raises(ValueError, match='index'):
        module.nvidia_temperature(-1)
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='N/A'))
    with pytest.raises(ValueError):
        module.nvidia_temperature()


def test_guard_hysteresis_and_interruptible_wait():
    temperatures = iter([85, 79, 69])
    waits, checked = [], []
    guard = TemperatureGuard(lambda: next(temperatures), sleep_temperature=80, wake_temperature=70,
                             interval=0.1, sleep_time=1, max_sleep=10,
                             clock=lambda: 0, wait=lambda seconds: waits.append(seconds))
    guard.step(lambda: checked.append(True))
    assert abs(sum(waits) - 2) < 1e-6
    assert len(checked) >= 3


def test_zero_interval_disables_reading():
    guard = TemperatureGuard(lambda: (_ for _ in ()).throw(AssertionError('disabled')), interval=0)
    guard.step(lambda: None)


def test_wait_is_bounded_and_interruptible():
    import pytest
    waits = []
    def wait(seconds):
        waits.append(seconds)
    def check():
        if waits:
            raise RuntimeError('cancelled')
    guard = TemperatureGuard(lambda: 90, sleep_time=60, wait=wait, clock=lambda: 0)
    with pytest.raises(RuntimeError, match='cancelled'):
        guard.step(check)
    assert waits == [0.1]


def test_maximum_sleep_and_minimum_measurement_interval():
    now, readings = [0.0], []
    def read():
        readings.append(now[0])
        return 90
    def wait(seconds):
        now[0] += seconds
    guard = TemperatureGuard(read, sleep_time=10, max_sleep=0.3, interval=1,
                             clock=lambda: now[0], wait=wait)
    guard.step(lambda: None)
    assert now[0] <= 0.300001
    count = len(readings)
    guard.step(lambda: None)
    assert len(readings) == count
