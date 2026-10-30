"""Real model JSONL difference-mode vs explicit-mask equivalence, local assets only."""
import copy
import json
from pathlib import Path
import sys
import time

from PIL import Image, ImageDraw
from composable_lora_check import Client
from prompt_fusion_check import image_difference, pixel_record


def main():
    executable, model, destination = sys.argv[1:]
    root, output = Path(__file__).resolve().parents[2], Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = Image.new('RGB', (512, 512), '#96b6ce')
    draw = ImageDraw.Draw(base)
    draw.rectangle((0, 310, 511, 511), fill='#9a7958')
    draw.ellipse((145, 90, 365, 350), fill='#e1c19e')
    altered = base.copy()
    ImageDraw.Draw(altered).rectangle((205, 170, 305, 290), fill='#e65050')
    before_path, changed_path = output / 'base.png', output / 'altered.png'
    base.save(before_path); altered.save(changed_path)
    client = Client(executable, root, output, False)
    report = dict(status='running')
    def save():
        (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    try:
        assert client.receive(time.monotonic() + 180)['type'] == 'hello'
        loaded = client.call('load_model', dict(path=str(Path(model).resolve()), dtype='fp16', vram_mode='gpu'))
        assert loaded['type'] == 'result', loaded
        preview = client.call('inpaint_difference_preview', dict(base_path=str(before_path), altered_path=str(changed_path),
            output_dir=str(output / 'preview'), threshold=1., dilation=2, erosion=1, blur=4))
        assert preview['type'] == 'result', preview
        mask = preview['result']['mask_path']
        assert Image.open(mask).getbbox() == (204, 169, 307, 292)
        common = dict(prompt='a portrait of a woman, detailed painting', negative_prompt='low quality', seed=173,
            steps=12, width=512, height=512, cfg_scale=5, sampler_name='DPM++ 2M', scheduler='Automatic',
            img2img=dict(init_image=str(changed_path), denoising_strength=.6, mask_blur=4))
        def run(name, params):
            result = client.call('img2img', dict(params=params, output_dir=str(output / name), preview_every=0))
            assert result['type'] == 'result', result
            images = result['result']['images']
            assert len(images) == 1
            report[name] = pixel_record(images[0]['path'])
            save()
            print(name, 'passed', flush=True)
            return images[0]['path']
        reference = copy.deepcopy(common)
        reference['img2img']['mask'] = mask
        direct = run('explicit-mask', reference)
        difference = copy.deepcopy(common)
        difference['hooks'] = {'inpaint_difference': dict(base_path=str(before_path), threshold=1., dilation=2, erosion=1)}
        actual = run('difference', difference)
        report['comparison'] = image_difference(direct, actual)
        assert report['comparison']['exact'], report['comparison']
        plain = run('ordinary-img2img', common)
        report['ordinary_comparison'] = image_difference(plain, actual)
        assert not report['ordinary_comparison']['exact']
        recovered = run('explicit-mask-recovery', reference)
        report['recovery'] = image_difference(direct, recovered)
        assert report['recovery']['exact']
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', failure=repr(exc))
        raise
    finally:
        save()
        client.close()


if __name__ == '__main__':
    main()
