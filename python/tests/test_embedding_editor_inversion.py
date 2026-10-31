"""Actual img2img Alternative routing with a scoped editor adapter (CPU only)."""
import types
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('diffusers')
from sdfx.engine import Engine
from sdfx.params import GenerationParams
from sdfx import embedding_editor_generation as editor, extras, extensions

@pytest.mark.parametrize('draft', [False, True])
@pytest.mark.parametrize('override', [{}, {'override_prompt': 'edited'}, {'override_negative': 'edited negative'}])
@pytest.mark.parametrize('neutral', [False, True])
@pytest.mark.parametrize('fail', [False, True])
def test_inversion_override_uses_draft_and_restores_even_on_encode_error(monkeypatch, override, neutral, fail, draft):
    engine = Engine.__new__(Engine)
    engine.pipe = types.SimpleNamespace(scheduler=None)
    engine._editor_preview = ('owned', {'clip_l': [[9.]]}) if draft else None
    expected = 9. if draft else 1.
    row, calls, captured = [1.], [], {}
    original = editor.preview_encoding
    def runner(engine, name, values, operation, *, transaction):
        old = row[0]; row[0] = values['clip_l'][0][0]; calls.append('install')
        try: return operation()
        finally: row[0] = old; calls.append('restore')
    monkeypatch.setattr(editor, 'preview_encoding', lambda engine, operation:
        original(engine, operation, runner=runner, transaction=lambda *_: None))
    def encode(*args):
        captured['encode_value'] = row[0]
        if fail: raise ValueError('encode failed')
        return {'value': row[0]}
    engine._encode = encode
    engine._encode_scheduled = lambda *a, **kw: {'value': expected}
    engine._activate = lambda pipe: None
    engine._scheduler = lambda *args: (types.SimpleNamespace(config={}), None)
    engine._run = lambda *args, **kw: captured.update(sample=kw['embeds'])
    lat = torch.ones(1, 4, 2, 2)
    monkeypatch.setattr(extensions, 'encode_image', lambda *a: lat)
    def inverse(*args, **kw): captured['inverse'] = args[3]; return lat
    monkeypatch.setattr(extras, 'invert_latents', inverse)
    gp = GenerationParams(prompt='base', negative_prompt='negative', hooks={'neutral_prompt': {}} if neutral else {})
    def run():
        engine._img2imgalt(gp, None, types.SimpleNamespace(check_interrupt=lambda: None), dict(override, randomness=0),
                          object(), {'value': expected}, [1], 0, 1, 16, 16, None, None)
    if override and fail:
        with pytest.raises(ValueError, match='encode failed'): run()
    else:
        run()
        assert captured['inverse']['value'] == expected
        assert captured['sample']['value'] == expected
    assert row == [1.]
    assert not getattr(engine, '_editor_preview_active', False)
    if override:
        assert captured['encode_value'] == expected
        assert calls == (['install', 'restore'] if draft else [])
    else: assert calls == []
