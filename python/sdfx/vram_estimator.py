"""Independent VRAM workload planning; measured execution and fitting are separate layers."""
import math
from .protocol import EngineError, INVALID_PARAMS

MODES = ('txt2img', 'txt2img_hr', 'txt2img_hr_latent', 'img2img')


def normalize(raw):
    config = dict(max_size=960, max_batch=8, steps=1)
    config.update(raw)
    for name, lower, upper in [('max_size', 256, 2048), ('max_batch', 1, 16), ('steps', 1, 100)]:
        try:
            value = config[name]
            if isinstance(value, bool):
                raise ValueError()
            number = float(value)
            if not math.isfinite(number) or number != int(number) or not lower <= number <= upper:
                raise ValueError()
            config[name] = int(number)
        except (ValueError, TypeError, OverflowError):
            raise EngineError(INVALID_PARAMS, f'VRAM estimator: invalid {name}')
    if config['max_size'] % 64:
        raise EngineError(INVALID_PARAMS, 'VRAM estimator: maximum size must be a multiple of 64')
    return config


def cases(config):
    result = []
    for mode in MODES:
        for batch in range(1, config['max_batch'] + 1):
            for target in range(256, config['max_size'] + 1, 64):
                hires = mode in ('txt2img_hr', 'txt2img_hr_latent')
                size = target // 2 if hires else target
                result.append(dict(mode=mode, batch=batch, target=target, width=size, height=size,
                                   upscaler=('Latent' if mode == 'txt2img_hr_latent' else 'Lanczos') if hires else None))
    return result



def _number(value, name, minimum=0, integer=False, positive=False):
    try:
        if isinstance(value, bool):
            raise ValueError()
        number = float(value)
        if (not math.isfinite(number) or number < minimum or (positive and number <= 0)
                or (integer and number != int(number))):
            raise ValueError()
        return number
    except (TypeError, ValueError, OverflowError):
        raise EngineError(INVALID_PARAMS, f'VRAM estimator: invalid {name}')


def estimate(rows, mode, width, height, batch, system_used_mb, total_mb):
    """Predict incremental reserved peak; never treat an estimate as measured capacity.

    Normalized coordinates avoid ill-conditioning at real image sizes. The cubic
    surface is used only at full column rank; sparse same-batch fits reduce degree.
    """
    import numpy as np
    if mode not in MODES:
        raise EngineError(INVALID_PARAMS, 'VRAM estimator: unknown operation')
    width = _number(width, 'width', integer=True, positive=True)
    height = _number(height, 'height', integer=True, positive=True)
    batch = _number(batch, 'batch', integer=True, positive=True)
    used = _number(system_used_mb, 'current used memory')
    total = _number(total_mb, 'total memory', positive=True)
    if used > total:
        raise EngineError(INVALID_PARAMS, 'VRAM estimator: current used memory exceeds total')
    unavailable = dict(available=False, mode=mode)
    selected = []
    if not isinstance(rows, list):
        raise EngineError(INVALID_PARAMS, 'VRAM estimator: samples must be a list')
    for row in rows:
        if not isinstance(row, dict):
            raise EngineError(INVALID_PARAMS, 'VRAM estimator: malformed sample')
        if row.get('mode') != mode or row.get('status') != 'complete':
            continue
        pixels = _number(row.get('pixels'), 'sample pixels', integer=True, positive=True)
        row_batch = _number(row.get('batch_size'), 'sample batch', integer=True, positive=True)
        delta = _number(row.get('reserved_delta_mb'), 'sample reserved memory')
        selected.append((math.sqrt(pixels), row_batch, delta))
    if not selected:
        return dict(unavailable, reason='no_matching_samples')
    point = math.sqrt(width * height)
    same = [row for row in selected if row[1] == batch]
    degree = None
    if same:
        data = np.asarray(same, dtype=np.float64)
        x, values = data[:, 0], data[:, 2]
        unique = np.unique(x)
        if len(unique) == 1:
            if not math.isclose(point, float(unique[0]), rel_tol=1e-12):
                return dict(unavailable, reason='insufficient_size_samples')
            prediction = float(values.mean())
            fitted = np.full_like(values, prediction)
            method = 'measured_point'
            degree = 0
        else:
            degree = min(3, len(unique) - 1)
            center, scale = float(x.mean()), float(x.std())
            coefficients = np.polynomial.polynomial.polyfit((x-center)/scale, values, degree)
            prediction = float(np.polynomial.polynomial.polyval((point-center)/scale, coefficients))
            fitted = np.polynomial.polynomial.polyval((x-center)/scale, coefficients)
            method = 'per_batch_polynomial'
        extrapolated = point < float(x.min()) or point > float(x.max())
        samples = len(same)
        size_range = [float(x.min()), float(x.max())]
    else:
        data = np.asarray(selected, dtype=np.float64)
        coordinates, values = data[:, :2], data[:, 2]
        center, scale = coordinates.mean(axis=0), coordinates.std(axis=0)
        if len(selected) < 10 or np.any(scale == 0):
            return dict(unavailable, reason='insufficient_surface_samples')
        normalized = (coordinates-center)/scale
        terms = [(power, degree-power) for degree in range(4) for power in range(degree+1)]
        matrix = np.column_stack([normalized[:, 0]**i * normalized[:, 1]**j for i,j in terms])
        coefficients, _, rank, _ = np.linalg.lstsq(matrix, values, rcond=None)
        if rank != len(terms):
            return dict(unavailable, reason='rank_deficient_surface')
        query = (np.asarray([point, batch])-center)/scale
        prediction = float(np.dot([query[0]**i * query[1]**j for i,j in terms], coefficients))
        fitted = matrix @ coefficients
        extrapolated = bool(np.any(query < normalized.min(axis=0)) or np.any(query > normalized.max(axis=0)))
        method, degree, samples = 'cubic_surface', 3, len(selected)
        size_range = [float(coordinates[:,0].min()), float(coordinates[:,0].max())]
    if not math.isfinite(prediction) or prediction < 0:
        return dict(unavailable, reason='nonphysical_fit')
    rmse = float(np.sqrt(np.mean((fitted-values)**2)))
    predicted_total = used + 1.1 * prediction
    if not math.isfinite(predicted_total) or not math.isfinite(rmse):
        return dict(unavailable, reason='nonfinite_fit')
    return dict(available=True, mode=mode, method=method, degree=degree, samples=samples,
                incremental_mb=prediction, system_used_mb=used, estimated_total_mb=predicted_total,
                total_mb=total, fraction=predicted_total/total, over_capacity=predicted_total>total,
                extrapolated=extrapolated, training_rmse_mb=rmse, measured_size_range=size_range,
                shape_assumption='square_calibration_by_pixel_area' if width != height else 'square',
                buffer_fraction=.1)



def write_profile(path, profile):
    """Atomic same-directory publication; a failed write never replaces old evidence."""
    import json
    import os
    from pathlib import Path
    import tempfile
    payload = (json.dumps(profile, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.vram-profile-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run_sweep(config, measure_case, ctx, on_oom, on_update):
    """Execute one owned serial workload, retaining each outcome before the next case.

    measure_case owns device-specific peak measurement; this orchestration is torch-free.
    Source behavior stops larger sizes after failure within that mode/batch, then continues
    other groups. Cancellation stays cancellation, with an explicitly partial saved state.
    """
    import time
    from .protocol import INTERRUPTED, OOM
    config = normalize(config)
    plan, rows, stopped = cases(config), [], set()
    def publish(status):
        state = dict(version=1, status=status, config=dict(config), rows=[dict(row) for row in rows],
                     planned_cases=len(plan), attempted_cases=len(rows))
        on_update(state)
        return state
    publish('running')
    try:
        for index, case in enumerate(plan):
            ctx.check_interrupt()
            group = case['mode'], case['batch']
            if group in stopped:
                continue
            row = dict(mode=case['mode'], width=case['width'], height=case['height'],
                       pixels=case['target']**2, target=case['target'], batch_size=case['batch'],
                       upscaler=case['upscaler'])
            started = time.perf_counter()
            try:
                metrics = measure_case(dict(case))
                delta = _number(metrics.get('reserved_delta_mb'), 'measured reserved memory')
                row.update(metrics)
                row.update(reserved_delta_mb=delta, status='complete')
            except Exception as error:
                if isinstance(error, EngineError) and error.code == INTERRUPTED:
                    raise
                ctx.check_interrupt()
                oom = ((isinstance(error, EngineError) and error.code == OOM)
                       or isinstance(error, MemoryError) or type(error).__name__ == 'OutOfMemoryError'
                       or 'out of memory' in str(error).lower())
                if oom:
                    on_oom()
                row.update(status='oom' if oom else 'error', error=str(error)[:2048])
                stopped.add(group)
            row['seconds'] = time.perf_counter() - started
            rows.append(row)
            publish('running')
            ctx.progress(index+1, len(plan), 'vram_estimator')
    except EngineError as error:
        if error.code == INTERRUPTED:
            publish('interrupted')
        raise
    return publish('stopped_after_error' if stopped else 'complete')



def generation_request(case, steps, output_dir, init_image):
    """Build an explicit vanilla workload; no user extension hooks enter calibration."""
    params = dict(prompt='an architectural city scene, daylight, detailed', negative_prompt='blurry, monochrome',
                  seed=42, steps=steps, sampler_name='Euler a', scheduler='Automatic', cfg_scale=15,
                  width=case['width'], height=case['height'], batch_size=case['batch'], n_iter=1,
                  hooks={}, loras=[])
    operation = 'img2img' if case['mode'] == 'img2img' else 'txt2img'
    if operation == 'img2img':
        params['img2img'] = dict(init_image=str(init_image), denoising_strength=.75)
    elif case['mode'] in ('txt2img_hr', 'txt2img_hr_latent'):
        params['hires'] = dict(enabled=True, scale=2, resize_x=case['target'], resize_y=case['target'],
                               upscaler=case['upscaler'], second_pass_steps=steps, denoising_strength=.7)
    return operation, dict(params=params, output_dir=str(output_dir), preview_method='none', preview_every=0,
                           lora_paths={}, embedding_paths=[], controlnet_paths={}, upscaler_paths={})



def fingerprint(identity):
    import hashlib, json
    def canonical(value):
        if isinstance(value, dict):
            return {key: canonical(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return int(value)
        return value
    return hashlib.sha256(json.dumps(canonical(identity), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()[:24]


def runtime_identity(engine):
    from .system_info import snapshot
    system = snapshot(engine, full=True)
    return {key: system[key] for key in ('version', 'libraries', 'gpu', 'runtime', 'model')}


class CudaMeter:
    def __init__(self, device):
        import torch
        self.cuda, self.device = torch.cuda, device

    def measure(self, generate):
        import gc
        cuda, device = self.cuda, self.device
        gc.collect()
        cuda.synchronize(device)
        cuda.empty_cache()
        cuda.reset_peak_memory_stats(device)
        baseline_reserved, baseline_allocated = cuda.memory_reserved(device), cuda.memory_allocated(device)
        free_before, total = cuda.mem_get_info(device)
        result = generate()
        cuda.synchronize(device)
        peak_reserved, peak_allocated = cuda.max_memory_reserved(device), cuda.max_memory_allocated(device)
        free_after, _ = cuda.mem_get_info(device)
        mib = 2**20
        return result, dict(reserved_delta_mb=(peak_reserved-baseline_reserved)/mib,
            allocated_delta_mb=(peak_allocated-baseline_allocated)/mib,
            reserved_peak_mb=peak_reserved/mib, allocated_peak_mb=peak_allocated/mib,
            baseline_reserved_mb=baseline_reserved/mib, baseline_allocated_mb=baseline_allocated/mib,
            system_used_before_mb=(total-free_before)/mib, system_used_after_mb=(total-free_after)/mib,
            total_mb=total/mib)


def benchmark(engine, params, ctx, meter=None, identity=None):
    """Actual serial calibration. Optional meter/identity injection is for device-free tests."""
    from pathlib import Path
    import uuid
    from .protocol import UNSUPPORTED, NO_MODEL
    if getattr(engine, 'dry_run', False) or not getattr(engine, 'cuda', False):
        raise EngineError(UNSUPPORTED, 'VRAM estimator requires a real CUDA-compatible GPU engine')
    if not getattr(engine, 'model', None):
        raise EngineError(NO_MODEL, 'Load a checkpoint before VRAM calibration')
    config = normalize(params)
    root_value = params.get('output_dir')
    if not isinstance(root_value, str) or not root_value.strip():
        raise EngineError(INVALID_PARAMS, 'VRAM estimator requires an output directory')
    if not isinstance(params.get('save_samples', False), bool):
        raise EngineError(INVALID_PARAMS, 'VRAM estimator save_samples must be boolean')
    ctx.check_interrupt()
    identity = runtime_identity(engine) if identity is None else identity
    meter = CudaMeter(engine.device) if meter is None else meter
    run_dir = Path(root_value).resolve() / 'vram-estimator' / ('run-' + uuid.uuid4().hex)
    run_dir.mkdir(parents=True)
    profile_path, init_path = run_dir / 'profile.json', run_dir / 'synthetic-input.png'
    from PIL import Image
    Image.new('RGB', (512, 512)).save(init_path)
    absent = object()
    names = ('_benchmark_no_save', '_vram_capture_sample', '_vram_sample')
    previous = {name: getattr(engine, name, absent) for name in names}
    import datetime
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    samples, latest = set(), {}
    def publish(state):
        record = dict(state, started_at=started_at, identity=identity, fingerprint=fingerprint(identity), profile_path=str(profile_path),
                      assumptions=dict(sampler='Euler a', cfg_scale=15, vanilla_hooks=True, input='synthetic black RGB 512x512'))
        write_profile(profile_path, record)
        latest.clear(); latest.update(record)
        engine._vram_last_profile = record
    def measure(case):
        old_sample = getattr(engine, '_vram_sample', None)
        if old_sample is not None:
            old_sample.close()
        engine._vram_sample = None
        engine._vram_capture_sample = params.get('save_samples', False) and case['mode'] not in samples
        operation, request = generation_request(case, config['steps'], run_dir, init_path)
        result, metrics = meter.measure(lambda: getattr(engine, operation)(request, ctx))
        images = result.get('images', [])
        if (len(images) != case['batch'] or any(image.get('width') != case['target'] or
                image.get('height') != case['target'] or image.get('path') for image in images)):
            raise EngineError(UNSUPPORTED, 'VRAM calibration returned unexpected dimensions/count or saved samples during measurement')
        captured = getattr(engine, '_vram_sample', None)
        if engine._vram_capture_sample and captured is not None:
            ctx.check_interrupt()
            folder = run_dir / 'samples'; folder.mkdir(exist_ok=True)
            sample_path = folder / (case['mode'] + '.png')
            captured.save(sample_path)
            metrics['sample_path'] = str(sample_path)
            samples.add(case['mode'])
        return metrics
    try:
        engine._benchmark_no_save, engine._vram_sample = True, None
        run_sweep(config, measure, ctx, engine.on_oom, publish)
        return dict(latest)
    finally:
        sample = getattr(engine, '_vram_sample', None)
        if sample is not None and sample is not previous['_vram_sample']:
            sample.close()
        for name, value in previous.items():
            if value is absent:
                if hasattr(engine, name): delattr(engine, name)
            else:
                setattr(engine, name, value)
        init_path.unlink(missing_ok=True)



def validate_profile(profile):
    if (not isinstance(profile, dict) or type(profile.get('version')) is not int or profile['version'] != 1
            or profile.get('status') not in ('running', 'complete', 'interrupted', 'stopped_after_error')
            or not isinstance(profile.get('identity'), dict) or not isinstance(profile.get('rows'), list)
            or len(profile['rows']) > 4096):
        raise EngineError(INVALID_PARAMS, 'VRAM estimator: malformed or unsupported profile')
    try:
        computed = fingerprint(profile['identity'])
    except (ValueError, TypeError, OverflowError):
        raise EngineError(INVALID_PARAMS, 'VRAM estimator: invalid profile identity')
    if profile.get('fingerprint') != computed:
        raise EngineError(INVALID_PARAMS, 'VRAM estimator: profile identity fingerprint does not match its contents')
    for row in profile['rows']:
        if (not isinstance(row, dict) or row.get('mode') not in MODES
                or row.get('status') not in ('complete', 'oom', 'error')):
            raise EngineError(INVALID_PARAMS, 'VRAM estimator: invalid profile row')
        if row['status'] == 'complete':
            _number(row.get('pixels'), 'profile pixels', integer=True, positive=True)
            _number(row.get('batch_size'), 'profile batch', integer=True, positive=True)
            _number(row.get('reserved_delta_mb'), 'profile reserved memory')
    return profile


def current_memory(engine):
    from .protocol import UNSUPPORTED
    try:
        import torch
        free, total = torch.cuda.mem_get_info(engine.device)
        return dict(system_used_mb=(total-free)/2**20, total_mb=total/2**20)
    except Exception as error:
        raise EngineError(UNSUPPORTED, 'VRAM estimator: current CUDA memory is unavailable', str(error))


def dispatch(engine, params, ctx, identity=None, memory=None):
    operation = params.get('operation', 'benchmark')
    if operation == 'benchmark':
        return benchmark(engine, params, ctx, identity=identity)
    if operation not in ('profile', 'estimate'):
        raise EngineError(INVALID_PARAMS, 'VRAM estimator: unknown operation')
    ctx.check_interrupt()
    profile = params.get('profile', getattr(engine, '_vram_last_profile', None))
    if profile is not None:
        validate_profile(profile)
    supported = not getattr(engine, 'dry_run', False) and getattr(engine, 'cuda', False)
    loaded = bool(getattr(engine, 'model', None))
    current_identity = (runtime_identity(engine) if identity is None else identity) if supported and loaded else None
    current_fingerprint = fingerprint(current_identity) if current_identity is not None else None
    compatible = profile is not None and profile['fingerprint'] == current_fingerprint
    if operation == 'profile':
        return dict(profile=profile, current_identity=current_identity, current_fingerprint=current_fingerprint,
                    compatible=compatible, supported=bool(supported), model_loaded=loaded)
    if profile is None:
        return dict(available=False, reason='no_profile')
    if not compatible:
        return dict(available=False, reason='profile_identity_mismatch', current_fingerprint=current_fingerprint,
                    profile_fingerprint=profile['fingerprint'])
    usage = current_memory(engine) if memory is None else memory()
    result = estimate(profile['rows'], params.get('mode', 'txt2img'), params.get('width', 512),
                      params.get('height', 512), params.get('batch', 1), usage['system_used_mb'], usage['total_mb'])
    return dict(result, fingerprint=current_fingerprint, profile_status=profile['status'])
