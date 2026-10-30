"""Compare the retained pinned process method with independent native formulas."""
import ast
import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'python'))
from sdfx.conditioning_highres_fix import MODES, resolve_weight

source = root / 'outputs/conditioning-highres-source/scripts/conditioning-highres-fix.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
owner = next(node for node in tree.body if isinstance(node, ast.ClassDef))
method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == 'process')
scope = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), 'exec'), scope)
count = 0
for denoise in (None, 0., .2, .5, .7, 1.):
    for index, mode in enumerate(MODES):
        params = SimpleNamespace(denoising_strength=denoise, inpainting_mask_weight=.37)
        with contextlib.redirect_stdout(io.StringIO()):
            scope['process'](None, params, index)
        weight = resolve_weight(mode, denoise)
        actual = .37 if weight is None else weight
        assert actual == params.inpainting_mask_weight, (mode, denoise, actual, params.inpainting_mask_weight)
        count += 1
print(f'PASS {count} pinned-source mode/denoise cases, including unchanged custom disabled baseline.')
