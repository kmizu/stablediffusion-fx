"""Native System Info, independently implemented from vladmandic's MIT extension behaviour.

No environment dump, username discovery, parent web-UI import, automatic file export or network send.
Legacy benchmark submission is a separate, one-use preview/token action.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import secrets
import socket
import subprocess
import sys
import tempfile
import time

from . import ENGINE_VERSION, models
from .protocol import EngineError, INVALID_PARAMS, UNSUPPORTED, INTERRUPTED

LEVELS = {'quick': [1], 'normal': [1, 2, 4], 'extensive': [1, 2, 4, 8, 16], 'ludicrous': [1, 2, 4, 8, 16, 32, 64]}
STEPS = {'turbo': 10, 'default': 20, 'long': 50}
LIBRARIES = ('torch', 'diffusers', 'transformers', 'accelerate', 'safetensors', 'numpy', 'pillow', 'xformers', 'psutil')
METADATA_FIELDS = ('version.app', 'version.engine', 'version.hash', 'platform.arch', 'platform.cpu', 'platform.system',
    'platform.release', 'platform.python', 'gpu.device', 'gpu.cuda', 'gpu.cudnn', 'gpu.hip', 'runtime.attention',
    'runtime.backend', 'runtime.pipeline', 'runtime.dtype', 'runtime.vram_mode', 'libraries.diffusers', 'libraries.transformers', 'libraries.torch')
METADATA_ALIASES = {'torch.version': 'libraries.torch', 'crossattention': 'runtime.attention', 'backend': 'runtime.backend',
    'pipeline': 'runtime.pipeline', 'flags': 'runtime.flags', 'libs.xformers': 'libraries.xformers',
    'libs.diffusers': 'libraries.diffusers', 'libs.transformers': 'libraries.transformers'}
METADATA_FIELDS += ('version.tag', 'version.tags', 'version.updated', 'version.url', 'gpu.ipex', 'gpu.openvino', *METADATA_ALIASES)
DESTINATION = ('logs3.papertrailapp.com', 32554)
# The same fixed scene/negative prompt as the source benchmark, so repeated runs use a stable workload.
PROMPT = ('postapocalyptic steampunk city, exploration, cinematic, realistic, hyper detailed, photorealistic maximum detail, '
          'volumetric light, (((focus))), wide-angle, (((brightly lit))), (((vegetation))), lightning, vines, destruction, devastation, wartorn, ruins')
NEGATIVE = '(((blurry))), ((foggy)), (((dark))), ((monochrome)), sun, (((depth of field)))'


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command(args, cwd=None):
    try:
        # On Windows a spawned utility can keep inherited pipe handles open after exit.
        # A temporary output file avoids communicate() reader threads blocking the JSON Lines worker.
        with tempfile.TemporaryFile() as output:
            p = subprocess.run(args, cwd=cwd, stdin=subprocess.DEVNULL, stdout=output,
                stderr=subprocess.DEVNULL, timeout=2, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            output.seek(0)
            return output.read(65536).decode('utf-8', errors='replace').strip() if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _memory(engine):
    from .base import ram_used_mb
    memory = {'process_rss_mb': ram_used_mb()}
    try:
        import psutil
        system = psutil.virtual_memory()
        memory.update(ram_total_mb=round(system.total / 2**20, 2), ram_available_mb=round(system.available / 2**20, 2), ram_percent=system.percent)
    except Exception:
        memory['ram_status'] = 'psutil unavailable'
    torch = sys.modules.get('torch')
    if torch is not None and getattr(engine, 'cuda', False):
        try:
            free, total = torch.cuda.mem_get_info()
            stats = torch.cuda.memory_stats()
            memory['gpu'] = dict(total_mb=total / 2**20, free_mb=free / 2**20, used_mb=(total-free) / 2**20,
                allocated_mb=torch.cuda.memory_allocated() / 2**20, allocated_peak_mb=torch.cuda.max_memory_allocated() / 2**20,
                reserved_mb=torch.cuda.memory_reserved() / 2**20, reserved_peak_mb=torch.cuda.max_memory_reserved() / 2**20,
                active_mb=stats.get('active_bytes.all.current', 0) / 2**20,
                inactive_split_mb=stats.get('inactive_split_bytes.all.current', 0) / 2**20,
                allocation_retries=stats.get('num_alloc_retries', 0), oom_events=stats.get('num_ooms', 0))
            try:
                memory['gpu']['utilization_percent'] = torch.cuda.utilization()
            except Exception:
                memory['gpu']['utilization_status'] = 'driver utilization unavailable'
        except Exception as error:
            memory['gpu_status'] = f'GPU statistics unavailable ({type(error).__name__})'
    else:
        memory['gpu_status'] = 'CUDA not active in this engine'
    return memory


def _cpu_brand():
    """Read the processor's model label without importing a hardware survey package."""
    if sys.platform == 'win32':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                    r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                value, _ = winreg.QueryValueEx(key, 'ProcessorNameString')
                if isinstance(value, str) and value.strip():
                    return value.strip()
        except OSError:
            pass
    elif sys.platform.startswith('linux'):
        try:
            with Path('/proc/cpuinfo').open(encoding='utf-8', errors='replace') as stream:
                for line in stream:
                    key, _, value = line.partition(':')
                    if key.strip() in ('model name', 'Hardware') and value.strip():
                        return value.strip()
        except OSError:
            pass
    elif sys.platform == 'darwin':
        value = _command(['sysctl', '-n', 'machdep.cpu.brand_string'])
        if value:
            return value
    return platform.processor().strip()


def hardware_facts(engine):
    """Small raw fact subset for image provenance; no Git/library report or model loading."""
    memory = _memory(engine)
    facts = dict(gpu_name=None, vram_bytes=None, cpu_name=_cpu_brand(),
                 ram_bytes=memory.get('ram_total_mb', 0) * 2**20)
    torch = sys.modules.get('torch')
    if getattr(engine, 'cuda', False):
        if torch is None:
            raise RuntimeError('Active CUDA device information unavailable')
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        facts.update(gpu_name=props.name, vram_bytes=props.total_memory)
    return facts


def snapshot(engine, full=False):
    now = time.time()
    started = getattr(engine, 'system_started_at', now)
    pipe = getattr(engine, 'pipe', None)
    model = getattr(engine, 'model', None) or {}
    result = dict(timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
        started_at=dt.datetime.fromtimestamp(started, dt.timezone.utc).isoformat(), uptime_seconds=max(0, now-started),
        memory=_memory(engine), model={k: model[k] for k in ('name', 'kind', 'hash') if k in model},
        runtime=dict(pipeline='dry-run generator (no diffusion)' if engine.dry_run else 'diffusers', backend='cuda' if getattr(engine, 'cuda', False) else 'cpu',
            dtype='not applicable (dry run)' if engine.dry_run else str(getattr(engine, 'dtype', engine.load_opts.dtype)), vram_mode=engine.load_opts.vram_mode,
            vae_tiling=engine.load_opts.vae_tiling, dry_run=bool(engine.dry_run),
            flags=[engine.load_opts.vram_mode, str(engine.load_opts.dtype)] + (['vae_tiling'] if engine.load_opts.vae_tiling else []),
            attention=type(next(iter(pipe.unet.attn_processors.values()))).__name__ if pipe is not None and getattr(pipe.unet, 'attn_processors', None) else 'not loaded'))
    vae = getattr(engine, 'vae_info', None) or {}
    result['model'].update({k: vae[k] for k in ('vae_name', 'vae_hash') if k in vae})
    def component_dtype(component):
        dtype = getattr(getattr(pipe, component, None), 'dtype', None)
        return str(dtype) if dtype is not None else None
    result['device'] = dict(active=str(getattr(engine, 'device', 'cpu')), dtype=result['runtime']['dtype'],
                            vae_dtype=component_dtype('vae'), unet_dtype=component_dtype('unet'))
    if not full:
        return result
    root = Path(__file__).resolve().parents[2]
    result.update(version=dict(app='stablediffusion-fx', engine=ENGINE_VERSION,
                               hash=_command(['git', 'rev-parse', 'HEAD'], root),
                               tag=_command(['git', 'describe', '--exact-match', '--tags'], root),
                               tags=_command(['git', 'tag', '--points-at', 'HEAD'], root),
                               updated=_command(['git', 'show', '-s', '--format=%cs', 'HEAD'], root), url=None),
        platform=dict(arch=platform.machine(), cpu=platform.processor(), system=platform.system(),
                      release=platform.release(), python=platform.python_version()),
        libraries={name: _version(name) for name in LIBRARIES}, gpu={'status': 'CUDA not active'},
        schedulers=engine.list_samplers({}, None), features=engine.features())
    result['repos'] = {'app_commit': result['version']['hash'], 'library_versions': result['libraries'],
                       'note': 'Standalone installed packages replace parent web-UI repository checkouts'}
    torch = sys.modules.get('torch')
    if torch is not None and getattr(engine, 'cuda', False):
        try:
            index = torch.cuda.current_device()
            result['gpu'] = dict(device=torch.cuda.get_device_name(index), count=torch.cuda.device_count(),
                total_mb=torch.cuda.get_device_properties(index).total_memory / 2**20,
                capability=list(torch.cuda.get_device_capability(index)), cuda=torch.version.cuda,
                hip=getattr(torch.version, 'hip', None), cudnn=torch.backends.cudnn.version(),
                driver=_command(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']))
        except Exception as error:
            result['gpu'] = {'status': f'unavailable ({type(error).__name__})'}
    return result


def inventory(engine, dirs, hashes, ctx):
    if not isinstance(dirs, dict) or not isinstance(hashes, bool):
        raise EngineError(INVALID_PARAMS, 'inventory dirs must be an object and hashes must be boolean')
    if any(not isinstance(v, list) or not all(isinstance(p, str) for p in v) for v in dirs.values()):
        raise EngineError(INVALID_PARAMS, 'inventory directories must be lists of strings')
    groups = {}
    for kind in ('checkpoints', 'lora', 'embeddings', 'hypernetworks', 'vae'):
        files = models.scan_models(dirs.get(kind, [])) if kind == 'checkpoints' else models.scan_files(
            kind if kind != 'hypernetworks' else 'embeddings', dirs.get(kind, []))
        items = []
        for entry in files:
            ctx.check_interrupt()
            path = Path(entry['path'])
            item = {k: entry[k] for k in ('name', 'kind', 'size_mb') if k in entry}
            item['hash'] = None
            if (engine.model or {}).get('path') == str(path):
                item['hash'] = engine.model.get('hash')
            if hashes:
                item['hash'] = models.path_hash(path)
            item['hash_status'] = 'SHA256 prefix' if item['hash'] else 'not calculated; use Hash inventory'
            if kind == 'embeddings':
                item['vectors'] = None
                if path.suffix.lower() == '.safetensors':
                    try:
                        header = models.read_safetensors_header(path)
                        shapes = [v['shape'] for k, v in header.items() if k != '__metadata__' and isinstance(v, dict) and 'shape' in v]
                        item['vectors'] = max((shape[0] if len(shape) > 1 else 1 for shape in shapes), default=None)
                    except (OSError, ValueError, KeyError):
                        item['vector_status'] = 'unreadable safetensors header'
                else:
                    item['vector_status'] = 'pickle is not loaded for inspection'
            items.append(item)
        groups[kind] = items
    groups['hypernetworks_status'] = 'Inventory only; hypernetwork inference is not implemented'
    return groups


def metadata(data, fields):
    result = {}
    for field in fields:
        if field not in METADATA_FIELDS:
            continue
        value = data
        for key in METADATA_ALIASES.get(field, field).split('.'):
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            result[field] = ' '.join(map(str, value)) if isinstance(value, list) else value
    return result


def benchmark_config(params):
    level, steps = params.get('level', 'normal'), params.get('steps', 'default')
    if not isinstance(level, str) or level not in LEVELS or not isinstance(steps, str) or steps not in STEPS:
        raise EngineError(INVALID_PARAMS, 'Unknown benchmark level or step preset')
    warmup = params.get('warmup', True)
    if not isinstance(warmup, bool):
        raise EngineError(INVALID_PARAMS, 'Benchmark warmup must be boolean')
    result = dict(level=level, steps=steps, inference_steps=STEPS[steps], batches=LEVELS[level], warmup=warmup,
                  sampler='Euler a', schedule='Automatic', cfg_scale=7, seed=42)
    for key in ('width', 'height'):
        value = params.get(key, 512)
        if isinstance(value, bool) or not isinstance(value, int) or value < 64 or value > 8192 or value % 8:
            raise EngineError(INVALID_PARAMS, 'Benchmark dimensions must be multiples of 8 from 64 to 8192')
        result[key] = value
    if result['width'] * result['height'] < 65536:
        raise EngineError(INVALID_PARAMS, 'Benchmark images must contain at least 65536 pixels, matching the source benchmark')
    return result


def benchmark(engine, params, ctx):
    if engine.dry_run:
        raise EngineError(UNSUPPORTED, 'Dry-run images are not a hardware benchmark; start the real engine')
    if engine.model is None:
        raise EngineError('no_model', 'Load a checkpoint before benchmarking')
    config = benchmark_config(params)
    system = snapshot(engine, full=True)
    stable = {key: system[key] for key in ('version', 'platform', 'libraries', 'gpu', 'runtime', 'model')}
    fingerprint = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:12]
    rows = []
    note = params.get('note', '')
    if not isinstance(note, str) or len(note) > 4096:
        raise EngineError(INVALID_PARAMS, 'Benchmark note must be text of at most 4096 characters')
    with tempfile.TemporaryDirectory(prefix='sdfx-benchmark-') as output:
        def run(batch, measured):
            ctx.check_interrupt()
            started = time.perf_counter()
            result = engine.txt2img({'output_dir': output, 'preview_method': 'none', 'params': dict(prompt=PROMPT,
                negative_prompt=NEGATIVE, seed=42, steps=config['inference_steps'], width=config['width'],
                height=config['height'], batch_size=batch, n_iter=1, sampler_name=config['sampler'],
                scheduler=config['schedule'], cfg_scale=config['cfg_scale'])}, ctx)
            elapsed = time.perf_counter() - started
            if len(result.get('images', [])) != batch or not math.isfinite(elapsed) or elapsed <= 0:
                raise EngineError(UNSUPPORTED, 'Benchmark returned an unexpected image count or invalid duration')
            for image in result['images']:
                if image.get('path'):
                    Path(image['path']).unlink(missing_ok=True)
                if image.get('width', config['width']) != config['width'] or image.get('height', config['height']) != config['height']:
                    raise EngineError(UNSUPPORTED, 'Benchmark returned unexpected output dimensions')
            if measured:
                rows.append(dict(batch=batch, seconds=elapsed, iterations_per_second=config['inference_steps'] * batch / elapsed,
                                 megapixels_per_second=config['width'] * config['height'] * batch / 2**20 / elapsed, status='complete'))
        try:
            if config['warmup']:
                ctx.log('info', 'System benchmark warmup (not included in scores)')
                run(1, False)
            for batch in config['batches']:
                ctx.log('info', f'System benchmark batch {batch}')
                run(batch, True)
        except Exception as error:
            if isinstance(error, EngineError) and error.code == INTERRUPTED:
                raise
            ctx.check_interrupt()
            oom = 'out of memory' in str(error).lower() or type(error).__name__ == 'OutOfMemoryError'
            if oom:
                engine.on_oom()
            rows.append(dict(batch=locals().get('batch', 1), status='oom' if oom else 'error', error=str(error)[:2048]))
    return dict(timestamp=system['timestamp'], fingerprint=fingerprint, system=stable, config=config, results=rows, note=note,
                status='complete' if all(r['status'] == 'complete' for r in rows) else 'stopped_after_error')


def _compact(value):
    # Match the upstream eleven-column logger's space-separated key:value representation.
    if isinstance(value, dict):
        return ' '.join(f'{key}:{_compact(item)}' for key, item in value.items())
    if isinstance(value, list):
        return ','.join(map(str, value))
    return str(value)


def submission_preview(engine, params):
    username, record = params.get('username', ''), params.get('record')
    if not isinstance(username, str) or not username.strip() or len(username) > 128 or any(c in username for c in '\r\n|\x00'):
        raise EngineError(INVALID_PARAMS, 'Enter a username without line breaks or separators to preview submission')
    if not isinstance(record, dict) or not isinstance(record.get('results'), list):
        raise EngineError(INVALID_PARAMS, 'Choose a benchmark record to submit')
    if (not all(isinstance(row, dict) for row in record['results']) or
            not isinstance(record.get('system', {}), dict) or not isinstance(record.get('config', {}), dict) or
            not isinstance(record.get('fingerprint', ''), str)):
        raise EngineError(INVALID_PARAMS, 'Malformed benchmark record')
    if not isinstance(params.get('include_hostname', False), bool):
        raise EngineError(INVALID_PARAMS, 'include_hostname must be boolean')
    hostname = platform.node() if params.get('include_hostname', False) else 'anonymous'
    system = record.get('system', {})
    runtime = system.get('runtime', {})
    config = record.get('config', {})
    model = system.get('model', {})
    optimizations = dict(pipeline=runtime.get('pipeline', ''), crossattention=runtime.get('attention', ''),
        flags=runtime.get('flags', []), compute=runtime.get('backend', ''), compile='none', dtype=runtime.get('dtype', ''))
    settings = dict(pipeline=runtime.get('pipeline', ''), model=model.get('name', ''), vae=model.get('vae_name', ''),
                    width=config.get('width', 512), height=config.get('height', 512))
    fields = [record.get('timestamp', ''), ' / '.join(str(r.get('iterations_per_second', 'error')) for r in record['results']),
        system.get('version', {}), system.get('platform', {}), system.get('libraries', {}), system.get('gpu', {}),
        optimizations, settings, username.strip(), record.get('note', ''), record.get('fingerprint', '')[:6]]
    line = '|'.join(_compact(f).replace('|', '/').replace('\r', ' ').replace('\n', ' ').replace('\x00', ' ') for f in fields).replace('  ', ' ').replace('"', '').strip()
    line += '|' + hashlib.sha256(line.encode()).hexdigest()[:6]
    stamp = dt.datetime.now().strftime('%b %d %H:%M:%S')
    packet = f'<14>{stamp} {hostname} SDBENCHMARK: {username.strip()} {line}\x00'
    if len(packet.encode('utf-8')) > 60000:
        raise EngineError(INVALID_PARAMS, 'Benchmark submission exceeds the UDP payload limit')
    preview = dict(token=secrets.token_urlsafe(24), destination=f'{DESTINATION[0]}:{DESTINATION[1]}/udp', packets=[packet])
    engine._system_submission = (time.monotonic(), preview)
    return preview


def submit(engine, params):
    pending = getattr(engine, '_system_submission', None)
    if not pending or params.get('token') != pending[1]['token'] or time.monotonic() - pending[0] > 300:
        raise EngineError(INVALID_PARAMS, 'Preview the exact benchmark payload again before submitting')
    engine._system_submission = None  # one use, including network failures
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as transport:
            transport.settimeout(5)
            for packet in pending[1]['packets']:
                transport.sendto(packet.encode('utf-8'), DESTINATION)
    except OSError as error:
        raise EngineError(UNSUPPORTED, f'Legacy submission unavailable: {error}; export the benchmark JSON locally')
    return dict(status='sent_unconfirmed', message='UDP packet sent. The legacy logger provides no receipt confirmation.')
