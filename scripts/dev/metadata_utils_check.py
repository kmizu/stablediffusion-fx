"""Real worker acceptance; run only with coordinated GPU permission.

Usage: python metadata_utils_check.py --python ENGINE_PYTHON --checkpoint PATH
  --lora PATH --output FRESH_DIR [--sdxl] [--engine-dir EXTRACTED/python]
Original models are read only. Staged copies and generated artifacts remain for inspection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import struct
import subprocess
import threading
import time


def digest(path, tensor_only=False):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        if tensor_only:
            size, = struct.unpack('<Q', stream.read(8))
            stream.seek(8 + size)
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


class Worker:
    def __init__(self, python, engine_dir, output):
        self.journal = (output / 'protocol.jsonl').open('w', encoding='utf-8')
        self.stderr = (output / 'stderr.log').open('w', encoding='utf-8')
        self.process = subprocess.Popen([python, '-m', 'sdfx'], cwd=engine_dir, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.stderr, text=True, encoding='utf-8',
            env={**os.environ, 'PYTHONPATH': str(engine_dir)}, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.messages = queue.Queue()
        self.serial = 0
        threading.Thread(target=self._read, daemon=True).start()
        hello = self.messages.get(timeout=120)
        assert hello['type'] == 'hello' and not hello.get('dry_run'), hello

    def _read(self):
        for line in self.process.stdout:
            self.journal.write(line)
            self.journal.flush()
            self.messages.put(json.loads(line))

    def request(self, kind, params):
        self.serial += 1
        rid = str(self.serial)
        message = {'id': rid, 'type': kind, 'params': params}
        self.journal.write(json.dumps({'request': message}) + '\n')
        self.journal.flush()
        self.process.stdin.write(json.dumps(message) + '\n')
        self.process.stdin.flush()
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            response = self.messages.get(timeout=30)
            if response.get('id') != rid:
                continue
            if response['type'] == 'error':
                raise RuntimeError(response)
            if response['type'] == 'result':
                return response['result']
        raise TimeoutError(kind)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=15)
        self.stderr.close()
        self.journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--lora', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sdxl', action='store_true')
    parser.add_argument('--engine-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'python')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    staged = args.output / 'models'
    staged.mkdir()
    checkpoint = staged / 'checkpoint.safetensors'
    lora = staged / 'style.safetensors'
    source_hashes = {str(p): digest(p) for p in (args.checkpoint, args.lora)}
    shutil.copyfile(args.checkpoint, checkpoint)
    shutil.copyfile(args.lora, lora)
    worker = Worker(args.python, args.engine_dir.resolve(), args.output)
    try:
        results = {}
        for label, path in (('checkpoint', checkpoint), ('lora', lora)):
            raw_hash = digest(path, True)
            info = worker.request('metadata_utils', {'operation': 'read', 'path': str(path.resolve())})
            metadata = dict(info['metadata'])
            metadata['NUI metadata test'] = '日本語 copy; original tensors retained'
            result = worker.request('metadata_utils', {'operation': 'write', 'path': str(path.resolve()),
                'metadata_json': json.dumps(metadata, ensure_ascii=False), 'source_token': info['source_token']})
            edited = Path(result['path'])
            assert raw_hash == digest(edited, True) == result['tensor_sha256']
            assert digest(path) != digest(edited)
            results[label] = result
        def generate(model, adapter, label):
            worker.request('load_model', {'path': str(model.resolve()), 'dtype': 'fp16', 'vram_mode': 'gpu'})
            side = 768 if args.sdxl else 512
            response = worker.request('txt2img', {'params': {
                'prompt': 'a woman in a garden, detailed illustration, portrait', 'negative_prompt': 'blurry, low quality',
                'steps': 16, 'seed': 173, 'width': side, 'height': side, 'cfg_scale': 4 if args.sdxl else 6,
                'loras': [{'name': 'style', 'weight': .3}]}, 'lora_paths': {'style': str(adapter.resolve())},
                'output_dir': str((args.output / label).resolve())})
            return Path(response['images'][0]['path'])
        baseline = generate(checkpoint, lora, 'baseline')
        edited = generate(Path(results['checkpoint']['path']), Path(results['lora']['path']), 'edited')
        from PIL import Image
        with Image.open(baseline) as a, Image.open(edited) as b:
            assert a.size == b.size and a.convert('RGB').tobytes() == b.convert('RGB').tobytes(), 'metadata edit changed RGB'
        assert all(digest(Path(path)) == expected for path, expected in source_hashes.items())
        (args.output / 'report.json').write_text(json.dumps({'source_sha256': source_hashes,
            'copies': results, 'baseline': str(baseline), 'edited': str(edited), 'exact_rgb': True}, indent=2), encoding='utf-8')
        print('PASS: source hashes, exact tensor payload, changed metadata and exact real-model RGB')
    finally:
        worker.close()


if __name__ == '__main__':
    main()
