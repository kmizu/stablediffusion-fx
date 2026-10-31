"""Coordinated, reversible real PhotoMaker candidate diagnostic; never enables support.

Run from the repository root only after obtaining the shared GPU slot. This
temporarily bypasses the indicated compatibility guards, restores exact source
bytes on ordinary exit/error, and uses a new output path. A forced process kill
cannot run finally: verify source hashes and restore from the printed backup.
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('case', choices=('signed', 'multidiffusion'))
parser.add_argument('python', help='isolated GPU venv Python executable')
parser.add_argument('output', help='new diagnostic output directory')
parser.add_argument('--size', type=int, default=768)
parser.add_argument('--steps', type=int, default=20)
parser.add_argument('--tile', type=int, default=512)
parser.add_argument('--overlap', type=int, default=64)
parser.add_argument('--vram-mode', choices=('gpu', 'model_offload', 'sequential_offload'), default='model_offload')
parser.add_argument('--controls', action='store_true', help='Run PhotoMaker-only and MultiDiffusion-only comparisons')
args = parser.parse_args()
root = Path.cwd()
out = Path(args.output).resolve()
if out.exists() or out.with_suffix('.trace.jsonl').exists() or out.with_suffix('.source-backup.zip').exists():
    raise SystemExit('Output, trace and source backup must be new')
if args.size <= 0 or args.size % 8 or args.steps < 4:
    raise SystemExit('Size must be positive and divisible by 8; steps must be at least four')
if args.case == 'multidiffusion' and not (0 < args.overlap < args.tile < args.size):
    raise SystemExit('MultiDiffusion requires 0 < overlap < tile < image size')
if args.controls and args.case != 'multidiffusion':
    raise SystemExit('Visual controls require the multidiffusion case')
for key in ('SDFX_PHOTOMAKER_MODEL', 'SDFX_KEYWORD_LORA'):
    if not os.environ.get(key) or not Path(os.environ[key]).is_file():
        raise SystemExit(f'Set {key} to the existing verified local file')

paths = [root / 'python/sdfx/clora/runtime.py', root / 'python/sdfx/engine.py']
if args.case == 'multidiffusion':
    paths.append(root / 'python/sdfx/embedding_merge_pipeline.py')
original = {path: path.read_bytes() for path in paths}
backup = out.with_suffix('.source-backup.zip')
backup.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(backup, 'x', zipfile.ZIP_DEFLATED) as z:
    for path, data in original.items(): z.writestr(path.relative_to(root).as_posix(), data)
with zipfile.ZipFile(backup) as z:
    assert all(z.read(path.relative_to(root).as_posix()) == data for path, data in original.items())
print('Verified source backup', backup, hashlib.sha256(backup.read_bytes()).hexdigest(), flush=True)

def replace_once(source, old, new):
    if source.count(old) != 1: raise ValueError('Expected guard does not match current source: ' + old)
    return source.replace(old, new)

try:
    path = paths[0]
    source = original[path].decode('utf8')
    old = "if self.adapters and hook_config(gp.hooks, 'photomaker', {}) is not None:"
    path.write_bytes(replace_once(source, old, 'if False and ' + old[3:]).encode())
    path = paths[1]
    source = original[path].decode('utf8')
    old = "neutral_runtime.validate(gp.hooks, controlnet=bool(cn and cn[0]), pag=bool(kwargs.get('pag')))"
    source = replace_once(source, old, "neutral_runtime.validate(gp.hooks, controlnet=bool(cn and cn[0]), pag=bool(kwargs.get('pag')), allow_photomaker=getattr(self, '_composable_job', None) is not None)")
    source = replace_once(source, '                    reject_unverified_merge_composable(raw_texts)',
                          '                    # Controlled diagnostic bypass; restored in finally.')
    if args.case == 'signed':
        old = 'if (hooks_mod.hook_config(gp.hooks, "negpip", {}) is not None'
        source = replace_once(source, old, 'if (False and ' + old[4:])
    path.write_bytes(source.encode())
    if args.case == 'multidiffusion':
        path = paths[2]
        source = original[path].decode('utf8')
        old = 'if offload and photomaker and multidiffusion and any(expressions(text) for text in prompts):'
        path.write_bytes(replace_once(source, old, 'if False and ' + old[3:]).encode())

    env = dict(os.environ)
    env.update(SDFX_PHOTO_TRACE=str(out.with_suffix('.trace.jsonl')), SDFX_PHOTOMAKER_CFG='6',
               SDFX_PHOTOMAKER_MERGE_EQ='1', SDFX_PHOTOMAKER_DIAGNOSTIC='0', SDFX_PHOTOMAKER_INTERRUPT='1',
               SDFX_PHOTOMAKER_SIZE=str(args.size), SDFX_PHOTOMAKER_STEPS=str(args.steps),
               SDFX_PHOTOMAKER_VRAM_MODE=args.vram_mode, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if args.case == 'signed':
        env.update(SDFX_PHOTOMAKER_NEGPIP='1', SDFX_PHOTOMAKER_NEGPIP_COMPARE='1')
    else:
        env.update(SDFX_PHOTOMAKER_MULTIDIFFUSION='1', SDFX_PHOTOMAKER_TILE=str(args.tile),
                   SDFX_PHOTOMAKER_TILE_OVERLAP=str(args.overlap))
        if args.controls: env['SDFX_PHOTOMAKER_CONTROLS'] = '1'
    required_paths = [root / 'scripts/dev/keyword_photo_trace', root / 'python',
                      Path(os.environ['TEMP']) / 'sdfx-photomaker-v1-source',
                      Path(os.environ['TEMP']) / 'sdfx-photomaker-genuine-target-v2']
    if any(not path.is_dir() for path in required_paths): raise ValueError('PhotoMaker source/dependency directory is missing')
    env['PYTHONPATH'] = os.pathsep.join(str(path) for path in required_paths)
    out.mkdir(parents=True)
    with (out / 'diagnostic.log').open('x', encoding='utf8') as log:
        result = subprocess.run([args.python, 'scripts/dev/gpu_check.py', args.python, str(out),
                                 'model_keyword_photomaker_composable'], stdout=log, stderr=subprocess.STDOUT, env=env)
    print('Diagnostic exit', result.returncode, flush=True)
finally:
    for path, data in original.items():
        path.write_bytes(data)
        print('Restored', path, hashlib.sha256(path.read_bytes()).hexdigest(), path.read_bytes() == data, flush=True)

if result.returncode:
    sys.exit(result.returncode)
