"""Generate small fixtures from the pinned MIT upstream; never import the parent app."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image

root = Path('outputs/inpaint-difference-source')
source = root / 'lib_inpaint_difference/mask_processing.py'
globals_module = types.ModuleType('lib_inpaint_difference.globals')
globals_module.DifferenceGlobals = types.SimpleNamespace(mask_brush_color='#8030ff', show_image_under_mask=True)
package = types.ModuleType('lib_inpaint_difference')
package.__path__ = []
sys.modules['lib_inpaint_difference'] = package
sys.modules['lib_inpaint_difference.globals'] = globals_module
spec = importlib.util.spec_from_file_location('upstream_difference_reference', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

rng = np.random.default_rng(67)
base = rng.integers(0, 256, (9, 11, 3), dtype=np.uint8)
altered = base.copy()
altered[0:5, 1:5] = 255 - altered[0:5, 1:5]
altered[6:9, 7:11] = 0
cases = []
for threshold in (0., .25, .5, 1.):
    for contours, dilate, erode in ((False, 0, 0), (True, 0, 0), (False, 1, 0),
                                    (False, 0, 1), (True, 2, 1)):
        mask, preview = module.compute_mask(Image.fromarray(base), Image.fromarray(altered),
            1, dilate, erode, threshold, contours)
        cases.append(dict(threshold=threshold, contours=contours, dilation=dilate, erosion=erode,
            mask=np.asarray(mask)[:, :, 0].tolist(), preview=np.asarray(preview).tolist()))
edge_cases = []
for name, first, second, blur, show, threshold, contours, dilate, erode in (
    ('resized-base', base[:5, :7], altered, 2, True, .5, False, 0, 0),
    ('empty-mask', base, base, 0, True, 1., False, 0, 0),
    ('full-mask', np.zeros_like(base), np.full_like(base, 255), 0, False, 1., False, 0, 0),
    ('no-underlay', base, altered, 3, False, .5, True, 1, 2),
    ('full-eroded-border', np.zeros_like(base), np.full_like(base, 255), 0, True, 1., False, 0, 3),
    ('full-contour-border', np.zeros_like(base), np.full_like(base, 255), 1, True, 1., True, 0, 0),
):
    globals_module.DifferenceGlobals.show_image_under_mask = show
    mask, preview = module.compute_mask(Image.fromarray(first), Image.fromarray(second),
        blur, dilate, erode, threshold, contours)
    edge_cases.append(dict(name=name, base=first.tolist(), altered=second.tolist(), blur=blur,
        show_under=show, threshold=threshold, contours=contours, dilation=dilate, erosion=erode,
        mask=np.asarray(mask)[:, :, 0].tolist(), preview=np.asarray(preview).tolist()))
target = Path('python/tests/fixtures/inpaint_difference_reference.json')
target.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes((json.dumps(dict(pin='656eed0a214186f4fb12fd6100fef582a01558bf',
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), base=base.tolist(),
    altered=altered.tolist(), cases=cases, edge_cases=edge_cases), ensure_ascii=False, separators=(',', ':')) + '\n').encode())
print(f'{len(cases) + len(edge_cases)} reference cases: {target}')
