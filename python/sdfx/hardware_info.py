"""Compact image provenance and source-style postprocessing clocks.

Independent behavior rewrite of Hardware Info in metadata (no source license grant).
This module does not import Torch, select devices, load models or affect random state.
"""
from __future__ import annotations

import logging
import math
import re
import time

log = logging.getLogger(__name__)


def compact(*, gpu_name, vram_bytes, cpu_name, ram_bytes):
    """Describe total memory in rounded GiB, keeping the source's historical GB labels."""
    if not isinstance(cpu_name, str) or not cpu_name.strip() or not math.isfinite(ram_bytes) or ram_bytes <= 0:
        raise ValueError('CPU brand and total system RAM are unavailable')
    parts = []
    if gpu_name:
        if not math.isfinite(vram_bytes) or vram_bytes <= 0:
            raise ValueError('GPU total memory is unavailable')
        parts.append(f'{gpu_name} {vram_bytes / 2**30:.0f}GB')
    parts.extend((cpu_name, f'{ram_bytes / 2**30:.0f}GB RAM'))
    text = ', '.join(parts)
    # Product/vendor boilerplate is removed; model numbers and capacity remain literal provenance.
    boilerplate = ('nvidia', 'geforce', '(r)', '(tm)', '(c)', 'cpu', 'gpu', '@', 'gen',
                   'amd', 'vega', 'ryzen', 'radeon', 'intel', 'core', 'arc')
    text = re.sub('|'.join(re.escape(word) for word in boilerplate), '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+th|\d+\.\d+ghz', '', text, flags=re.IGNORECASE)
    return re.sub(r'\s+,', ',', re.sub(r'\s+', ' ', text).strip())


def collect(engine):
    from .system_info import hardware_facts
    return compact(**hardware_facts(engine))


def cached_profile(engine, supplier=None):
    """Collect once per engine lifetime, including a nonfatal unknown fallback after failure."""
    if not hasattr(engine, '_hardware_info_profile'):
        try:
            engine._hardware_info_profile = (supplier or (lambda: collect(engine)))()
        except Exception as error:
            log.warning('Hardware metadata unavailable (%s)', type(error).__name__)
            engine._hardware_info_profile = 'unknown'
    return engine._hardware_info_profile


class ImageTiming:
    def __init__(self, batch_size, clock=time.perf_counter, started_at=None):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        self._size = batch_size
        self._clock = clock
        self._start = clock() if started_at is None else started_at
        self._count = 0

    def next_image(self):
        elapsed = max(0.0, self._clock() - self._start) / self._size
        self._count += 1
        if self._count % self._size == 0:
            self._start = self._clock()
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        return (f'{minutes} min. ' if minutes else '') + f'{seconds:.1f} sec.'


def record(hardware, time_taken, remark=''):
    return {'hardware_info': hardware, 'time_taken': time_taken, 'remark': remark or None}
