"""Test-first source-contract tests. CPU only; no parent WebUI or Torch imports required."""
from types import SimpleNamespace
import sys
import pytest
from sdfx import hardware_info as hardware


def test_compact_hardware_normalizes_brand_frequency_and_rounds_total_gib():
    text = hardware.compact(gpu_name='NVIDIA GeForce RTX 3060', vram_bytes=12 * 2**30,
        cpu_name='Intel(R) Core(TM) i5-10400F CPU @ 2.90GHz', ram_bytes=63.6 * 2**30)
    assert text == 'RTX 3060 12GB, i5-10400F, 64GB RAM'


def test_compact_cpu_only_and_newer_vendor_names_keep_actual_model_identity():
    assert hardware.compact(gpu_name=None, vram_bytes=None,
        cpu_name='13th Gen Intel(R) Core(TM) i7-13700', ram_bytes=32 * 2**30) == 'i7-13700, 32GB RAM'
    assert hardware.compact(gpu_name='AMD Radeon RX 7900 XTX', vram_bytes=24 * 2**30,
        cpu_name='AMD Ryzen 9 7950X CPU', ram_bytes=64 * 2**30) == 'RX 7900 XTX 24GB, 9 7950X, 64GB RAM'


def test_profile_is_cached_and_failed_detection_is_nonfatal_unknown():
    engine = SimpleNamespace()
    calls = []
    def collect():
        calls.append('collect')
        return 'RTX 3060 12GB, i5-10400F, 64GB RAM'
    assert hardware.cached_profile(engine, collect) == hardware.cached_profile(engine, collect)
    assert calls == ['collect']
    def broken():
        raise OSError('provider unavailable')
    assert hardware.cached_profile(SimpleNamespace(), broken) == 'unknown'


def test_system_info_facts_use_active_engine_device_and_reuse_system_ram_provider(monkeypatch):
    from sdfx import system_info
    monkeypatch.setattr(system_info, '_cpu_brand', lambda: 'Actual branded CPU')
    monkeypatch.setattr(system_info, '_memory', lambda engine: {'ram_total_mb': 65536})
    queried = []
    def properties(index):
        queried.append(index)
        return SimpleNamespace(name='GPU at active device', total_memory=12 * 2**30)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        current_device=lambda: 2, get_device_properties=properties)))
    assert system_info.hardware_facts(SimpleNamespace(cuda=True)) == {
        'gpu_name': 'GPU at active device', 'vram_bytes': 12 * 2**30,
        'cpu_name': 'Actual branded CPU', 'ram_bytes': 64 * 2**30}
    assert queried == [2]
    assert system_info.hardware_facts(SimpleNamespace(cuda=False))['gpu_name'] is None
    assert queried == [2]


def test_timing_can_include_setup_before_parsing_without_sharing_global_clock():
    timing = hardware.ImageTiming(batch_size=2, clock=lambda: 8.0, started_at=0.0)
    assert timing.next_image() == '4.0 sec.'


def test_each_postprocessed_image_averages_by_batch_and_resets_after_batch_boundary():
    clock = [0.0]
    timing = hardware.ImageTiming(batch_size=2, clock=lambda: clock[0])
    values = []
    for now in (8.0, 8.2, 12.2, 12.4):
        clock[0] = now
        values.append(timing.next_image())
    assert values == ['4.0 sec.', '4.1 sec.', '2.0 sec.', '2.1 sec.']


def test_separate_generation_requests_do_not_share_iteration_clock_state():
    clock = [0.0]
    first = hardware.ImageTiming(batch_size=1, clock=lambda: clock[0])
    clock[0] = 62.5
    assert first.next_image() == '1 min. 2.5 sec.'
    clock[0] = 100.0
    second = hardware.ImageTiming(batch_size=1, clock=lambda: clock[0])
    clock[0] = 102.0
    assert second.next_image() == '2.0 sec.'


def test_extras_uses_whole_operation_time_instead_of_dividing_by_generation_batch_size():
    clock = [10.0]
    timing = hardware.ImageTiming(batch_size=1, clock=lambda: clock[0])
    clock[0] = 135.5
    assert timing.next_image() == '2 min. 5.5 sec.'


def test_record_keeps_remark_literal_and_omits_empty_remark():
    record = hardware.record('GPU, CPU, RAM', '2.0 sec.', 'quiet, 日本語\nsecond line')
    assert record == {'hardware_info': 'GPU, CPU, RAM', 'time_taken': '2.0 sec.', 'remark': 'quiet, 日本語\nsecond line'}
    assert hardware.record('unknown', '0.0 sec.', '')['remark'] is None


@pytest.mark.parametrize('batch_size', [0, -1, True, 1.5])
def test_invalid_batch_size_cannot_divide_by_zero_or_silently_change_timing(batch_size):
    with pytest.raises(ValueError):
        hardware.ImageTiming(batch_size=batch_size, clock=lambda: 0.0)
