"""Run a protected intermediate-video recipe, prompting for the password at execution."""
import getpass
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from .save_intermediate import Recorder, normalize
from .protocol import EngineError


def export(spec_path, password):
    if not isinstance(password, str) or not password:
        raise ValueError('image password is required')
    spec_path = Path(spec_path).resolve(strict=True)
    if spec_path.stat().st_size > 2_000_000:
        raise ValueError('recipe is too large')
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    name = spec.get('name')
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,120}', name):
        raise ValueError('invalid recipe name')
    settings = normalize(spec.get('settings', {}))
    records = spec.get('frames')
    if not isinstance(records, list) or not 1 <= len(records) <= 10000:
        raise ValueError('invalid recipe frames')
    frames = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('invalid recipe frame')
        filename = record.get('path')
        if not isinstance(filename, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+\.png', filename):
            raise ValueError('invalid recipe frame path')
        path = spec_path.parent / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError('recipe frame missing or linked')
        width, height = record.get('width'), record.get('height')
        if type(width) is not int or type(height) is not int or not (1 <= width <= 16384 and 1 <= height <= 16384):
            raise ValueError('invalid recipe frame size')
        frames.append(dict(path=str(path), width=width, height=height))
    executable = shutil.which('ffmpeg')
    if not executable:
        raise ValueError('ffmpeg is required')
    target = spec_path.parent / (name + '.' + settings['format'])
    if target.exists():
        raise FileExistsError('output video already exists')
    class Context:
        def check_interrupt(self):
            pass
    # Stage on the destination filesystem; publish by an exclusive hard link.
    # A concurrent existing target is never overwritten or removed.
    with tempfile.TemporaryDirectory(prefix='.protected-recipe-', dir=spec_path.parent) as stage:
        recorder = Recorder.__new__(Recorder)
        recorder.c, recorder.ctx, recorder.directory = settings, Context(), Path(stage)
        recorder.image_password, recorder.ffmpeg, recorder.artifacts = password, executable, []
        recorder._video(frames, name)
        os.link(Path(stage) / target.name, target)
    return target


def main():
    import sys
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m sdfx.protected_recipe RECIPE.json')
    print('This exports a readable video and creates temporary readable frames.')
    password = getpass.getpass('Image password: ') if sys.stdin.isatty() else sys.stdin.readline().rstrip('\r\n')
    try:
        path = export(sys.argv[1], password)
    except (ValueError, OSError, EngineError) as error:
        raise SystemExit(f'Could not export video: {error}') from None
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
