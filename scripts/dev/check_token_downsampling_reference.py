"""Compare with the pinned MIT wrapper's pure functions, without importing A1111."""
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
from sdfx.token_downsampling import resample

source = Path('outputs/token-downsampling-upstream.py')
assert hashlib.sha256(source.read_bytes()).hexdigest() == '1fdfb2ac5f24f232108a42587a0bbb1d480412e21f6c5e7874e2bae1898b6d0b'
tree = ast.parse(source.read_text(encoding='utf-8'))
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ('up_or_downsample', 'compute_merge')]
assert len(functions) == 2
namespace = dict(torch=torch, F=F, math=math)
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
torch.manual_seed(19)
results = []
for h, w in [(64, 64), (64, 96), (96, 96)]:
    for depth in (1, 2, 4):
        for factor in (1, 1.5, 2, 3, 5):
            ch, cw = h // depth, w // depth
            x = torch.randn(2, ch*cw, 8)
            info = dict(size=(h, w), args=dict(max_depth=depth, downsample_factor={depth: factor}))
            expected = namespace['compute_merge'](x, info)(x)
            actual = resample(x, ch, cw, factor)
            assert torch.equal(expected, actual), (h, w, depth, factor)
            results.append(dict(height=h, width=w, depth=depth, factor=factor, max_error=0))
Path('outputs/todo-reference.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print(f'PASS exact pinned MIT reference comparison: {len(results)} configurations')
