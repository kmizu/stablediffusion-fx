"""Run only pinned upstream merge_parser against NumPy-backed fake CLIP rows.

No A1111 imports, model loads or torch. Source is an explicit retained input.
"""
import argparse
import ast
import re
import types

import numpy as np

from sdfx.embedding_merge import parse_expression, evaluate


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)

    @property
    def shape(self):
        return self.data.shape

    def size(self, dim=None):
        return self.data.shape if dim is None else self.data.shape[dim]

    def to(self, **kwargs):
        return self

    def roll(self, amount, dims):
        return Tensor(np.roll(self.data, amount, axis=dims))

    def __getitem__(self, key):
        return Tensor(self.data[key])

    def __setitem__(self, key, value):
        self.data[key] = value.data

    def __add__(self, other):
        return Tensor(self.data + other.data)

    def __sub__(self, other):
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        return Tensor(self.data * other)

    def __truediv__(self, other):
        return Tensor(self.data / other)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    args = parser.parse_args()
    with open(args.source, encoding='utf-8') as source:
        tree = ast.parse(source.read())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'merge_parser')
    torch = types.SimpleNamespace(zeros=lambda *dims: Tensor(np.zeros(dims)),
        cat=lambda values: Tensor(np.concatenate([v.data for v in values])), float32=np.float32)
    cases = ["'a'", "'a'+'b'", "'a'-'b'", "'a'*2/3+'b'", "'a'+'b'=*2-'a'",
        "'a':3:+1", "'a'+'b':-1=:1", "'a','b',0,00,000,0000", "'a'*2,'b'",
        "'it''s'", "'a':0", "'':3", "'a':+100", "'a'+'b'=/3=:4:-1"]
    for widths in [(768,), (768, 1280)]:
        names = ('clip_l', 'clip_g')[:len(widths)]

        def literal(text):
            seeds = {'a': [1, 2], 'b': [10], "it's": [7], '': []}[text]
            return tuple(np.repeat(np.array(seeds, dtype=np.float32)[:, None], width, axis=1)
                         * (i + 1) for i, width in enumerate(widths))

        def token(value):
            value = {'pad': 90, 'end': 91, 'start': 92}.get(value, value)
            return tuple(np.full((1, width), value, dtype=np.float32) for width in widths)

        wrapped_type = type('FrozenCLIPEmbedder', (), {})
        clips = []
        for _ in widths:
            wrapped = wrapped_type()
            wrapped.tokenizer = types.SimpleNamespace(get_vocab=lambda: {str(i): i for i in range(100)})
            clips.append(types.SimpleNamespace(wrapped=wrapped, id_pad=90, id_end=91, id_start=92))
        scope = dict(re=re, torch=torch, devices=types.SimpleNamespace(device='cpu'),
            get_model_clips=lambda: clips, grab_vectors=lambda text: [Tensor(v) for v in literal(text)],
            tokens_to_vectors=lambda values: [Tensor(np.full((1, w), v)) for w, v in zip(widths, values)],
            to_int=lambda v: int(v), to_float=lambda v: float(v),
            reg_clean=re.compile(r'\s+'),
            reg_oper=re.compile(r'(=?)(?:([*/,])([+-]?[0-9]*(?:\.[0-9]*)?(?:L|G)?)|:([+-]?)(-?[0-9]+))'),
            sdxl_sizes={'L': 768, 'G': 1280})
        exec(compile(ast.Module(body=[function], type_ignores=[]), args.source, 'exec'), scope)
        selected = cases + (["'a'*2L/3G", "'a'+'b'=*2G"] if len(widths) == 2 else [])
        for expression in selected:
            expected, error = scope['merge_parser'](expression, False)
            assert error is None, (expression, error)
            actual = evaluate(parse_expression(expression), literal, token, names)
            for left, right in zip(expected, actual):
                np.testing.assert_array_equal(left.data, right, err_msg=f'{widths}: {expression}')
        print(f'{widths}: {len(selected)} source numerical cases match exactly')


if __name__ == '__main__':
    main()
