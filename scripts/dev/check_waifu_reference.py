"""Compare native inference against downloaded, pinned licensed original models.

References stay in ignored outputs/, never import the unlicensed extension wrapper.
"""
import importlib.util
import json
from pathlib import Path
import sys
import torch
import numpy as np
from PIL import Image
from sdfx.waifu2x import load_model, run_tiles

torch.set_num_threads(2)
def reference(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

reference('Common', 'outputs/waifu2x-licensed-common.py')
original = reference('WaifuReferenceModels', 'outputs/waifu2x-licensed-models.py')
sys.modules['Models'] = original
splitters = reference('WaifuReferenceSplitter', 'outputs/waifu2x-licensed-splitter.py')
torch.manual_seed(813)
pixels = torch.rand(1, 3, 17, 19)
report = []
def tile_comparison(actual, expected):
    errors = []
    for width, height in ((1, 1), (58, 135), (65, 63)):
        data = np.random.default_rng(713).integers(0, 256, (height, width, 3), dtype=np.uint8)
        image = Image.fromarray(data)
        splitter = splitters.ImageSplitter(seg_size=64, scale_factor=2, boarder_pad_size=3)
        patches = splitter.split_img_tensor(image, scale_method=None, img_pad=0)
        reference_pixels = splitter.merge_img_tensor([expected(patch) for patch in patches])[0].permute(1, 2, 0).numpy()
        reference_pixels = np.clip(reference_pixels * 255, 0, 255).astype(np.uint8)
        output = np.asarray(run_tiles(image, actual))
        assert np.array_equal(output, reference_pixels), (width, height, np.abs(output.astype(int)-reference_pixels).max())
        errors.append(dict(width=width, height=height, max_uint8_error=0))
    return errors
with torch.inference_mode():
    for style in ('anime', 'photo'):
        for noise in range(4):
            path = Path(f'outputs/waifu2x-models/{style}-noise{noise}.json')
            expected = original.UpConv_7().eval()
            expected.load_pre_train_weights(str(path))
            actual = load_model(f'{style}-{noise}', path, 'cpu')
            a, b = actual(pixels), expected(pixels)
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)
            report.append(dict(model=f'{style}-{noise}', max_error=float((a-b).abs().max()), shape=list(a.shape), tiles=tile_comparison(actual, expected)))
    path = Path('outputs/waifu2x-models/carn-v2.pt')
    expected = original.network_to_half(original.CARN_V2()).float().eval()
    expected.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    actual = load_model('carn-v2', path, 'cpu')
    a, b = actual(pixels), expected(pixels)
    torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)
    report.append(dict(model='carn-v2', max_error=float((a-b).abs().max()), shape=list(a.shape), tiles=tile_comparison(actual, expected)))
Path('outputs/waifu2x-reference-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('PASS nine native models match pinned licensed original CPU float32 inference', report)
