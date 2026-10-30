"""Dedicated-process real CUDA allocator OOM acceptance; never changes product limits."""
import io
import json
import sys
from pathlib import Path
import torch
from sdfx.engine import Engine
from sdfx.protocol import Emitter, JobContext
from sdfx.vram_estimator import CudaMeter, benchmark

output, checkpoint = Path(sys.argv[1]).resolve(), str(Path(sys.argv[2]).resolve())
output.mkdir(parents=True, exist_ok=True)
engine = Engine()
ctx = JobContext(Emitter(io.BytesIO()), 'bounded-oom')
engine.load_model(dict(path=checkpoint, dtype='fp16', vram_mode='gpu'), ctx)
device_index = torch.cuda.current_device()

class BoundedMeter(CudaMeter):
    def measure(self, generate):
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        total = torch.cuda.get_device_properties(self.device).total_memory
        # Only this process is capped, just above resident model weights. The
        # actual vanilla generation must encounter the CUDA allocator limit.
        limit = min(.99, (torch.cuda.memory_reserved(self.device) + 2**20) / total)
        torch.cuda.set_per_process_memory_fraction(limit, device_index)
        try:
            return super().measure(generate)
        finally:
            torch.cuda.set_per_process_memory_fraction(1.0, device_index)

try:
    record = benchmark(engine, dict(output_dir=str(output), max_size=320, max_batch=1, steps=1), ctx,
                       meter=BoundedMeter(engine.device))
    assert record['status'] == 'stopped_after_error', record
    assert len(record['rows']) == 4 and all(row['status'] == 'oom' for row in record['rows']), record['rows']
    assert not hasattr(engine, '_benchmark_no_save')
    recovered = benchmark(engine, dict(output_dir=str(output), max_size=256, max_batch=1, steps=1,
                                      save_samples=True), ctx)
    assert recovered['status'] == 'complete' and len(recovered['rows']) == 4, recovered
    (output / 'bounded-oom-report.json').write_text(json.dumps(dict(
        scope='Dedicated-process CUDA allocator cap; not physical GPU capacity',
        oom=record, recovered=recovered), indent=2), encoding='utf-8')
    print('PASS actual bounded CUDA OOM in four modes, larger-size skipping, flag restoration and four-mode recovery')
finally:
    torch.cuda.set_per_process_memory_fraction(1.0, device_index)
    engine._free()
