r"""A1111 prompt editing / alternation: `[from:to:when]`, `[to:when]`, `[from::when]`, `[a|b|c]` (nested).

Port of modules/prompt_parser.get_learned_conditioning_prompt_schedules without the lark dependency: a small
backtracking parser with the same grammar

    start:     (prompt | /[][():]/+)*
    prompt:    (emphasized | scheduled | alternate | plain | WHITESPACE)*
    emphasized: "(" prompt ")" | "(" prompt ":" prompt ")" | "[" prompt "]"
    scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER [WHITESPACE] "]"
    alternate: "[" prompt ("|" [prompt])+ "]"
    plain:     /([^\\\[\]():|]|\\.)+/

`schedule(prompt, steps)` returns [[end_step, text], ...] exactly like the original (doctests in tests/).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Union

_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_WS = re.compile(r"\s+")
_PLAIN = re.compile(r"(?:[^\\\[\]():|]|\\.)+")


@dataclass
class Sched:
    before: Optional[list]
    after: list
    when: str
    step: int = 0


@dataclass
class Alt:
    options: list


@dataclass
class Lit:
    text: str


Node = Union[Lit, Sched, Alt, list]


class _Fail(Exception):
    pass


class _Parser:
    def __init__(self, s: str):
        self.s = s

    def prompt(self, i: int) -> tuple[list, int]:
        out: list = []
        while i < len(self.s):
            c = self.s[i]
            if c in "([":
                try:
                    node, i = self.bracket(i)
                    out.append(node)
                    continue
                except _Fail:
                    break
            m = _PLAIN.match(self.s, i)
            if m:
                out.append(Lit(m.group(0)))
                i = m.end()
                continue
            break  # one of ) ] : | -> handled by the caller
        return out, i

    def expect(self, i: int, ch: str) -> int:
        if i < len(self.s) and self.s[i] == ch:
            return i + 1
        raise _Fail()

    def number_close(self, i: int) -> tuple[str, int]:
        m = _WS.match(self.s, i)
        if m:
            i = m.end()
        m = _NUMBER.match(self.s, i)
        if not m:
            raise _Fail()
        num, i = m.group(0), m.end()
        m = _WS.match(self.s, i)
        if m:
            i = m.end()
        return num, self.expect(i, "]")

    def bracket(self, i: int) -> tuple[Node, int]:
        if self.s[i] == "(":
            p1, j = self.prompt(i + 1)
            if j < len(self.s) and self.s[j] == ")":
                return [Lit("("), p1, Lit(")")], j + 1
            if j < len(self.s) and self.s[j] == ":":
                p2, k = self.prompt(j + 1)
                return [Lit("("), p1, Lit(":"), p2, Lit(")")], self.expect(k, ")")
            raise _Fail()
        # "[" ...
        p1, j = self.prompt(i + 1)
        if j >= len(self.s):
            raise _Fail()
        c = self.s[j]
        if c == "]":
            return [Lit("["), p1, Lit("]")], j + 1
        if c == ":":
            try:  # [to:when]
                num, k = self.number_close(j + 1)
                return Sched(None, p1, num), k
            except _Fail:
                pass
            p2, k = self.prompt(j + 1)
            k = self.expect(k, ":")
            num, k = self.number_close(k)
            return Sched(p1, p2, num), k
        if c == "|":
            opts = [p1]
            while j < len(self.s) and self.s[j] == "|":
                p, j = self.prompt(j + 1)
                opts.append(p)
            return Alt(opts), self.expect(j, "]")
        raise _Fail()

    def start(self) -> list:
        out, i = [], 0
        while i < len(self.s):
            node, j = self.prompt(i)
            out.extend(node)
            if j < len(self.s):
                if self.s[j] in "[]():":
                    out.append(Lit(self.s[j]))
                    j += 1
                else:  # a stray '|' is a parse error in the original grammar
                    raise _Fail()
            i = j
        return out


def _walk(nodes, fn):
    for n in nodes:
        if isinstance(n, list):
            _walk(n, fn)
        else:
            fn(n)
            if isinstance(n, Sched):
                if n.before:
                    _walk(n.before, fn)
                _walk(n.after, fn)
            elif isinstance(n, Alt):
                for o in n.options:
                    _walk(o, fn)


def _text(nodes, step: int) -> str:
    out = []
    for n in nodes:
        if isinstance(n, list):
            out.append(_text(n, step))
        elif isinstance(n, Lit):
            out.append(n.text)
        elif isinstance(n, Sched):
            out.append(_text(n.before or [], step) if step <= n.step else _text(n.after, step))
        elif isinstance(n, Alt):
            opts = n.options
            out.append(_text(opts[(step - 1) % len(opts)], step))
    return "".join(out)


def schedule(prompt: str, steps: int, hires_steps: Optional[int] = None) -> list[list]:
    """[[end_at_step, text], ...] for a prompt sampled with `steps` steps (A1111 semantics)."""
    if hires_steps is None:
        int_offset, flt_offset, total = 0, 0.0, steps
    else:
        int_offset, flt_offset, total = steps, 1.0, hires_steps
    try:
        tree = _Parser(prompt).start()
    except _Fail:
        return [[total, prompt]]
    res = {total}

    def collect(n):
        if isinstance(n, Sched):
            v = float(n.when)
            v = (v - flt_offset) * total if "." in n.when else (v - int_offset)
            n.step = min(total, int(v))
            if n.step >= 1:
                res.add(n.step)
        elif isinstance(n, Alt):
            res.update(range(1, total + 1))
    _walk(tree, collect)
    return [[t, _text(tree, t)] for t in sorted(res)]


def has_schedule(prompt: str) -> bool:
    return "[" in prompt and len(schedule(prompt, 10)) > 1


def text_at(sched: list[list], step: int) -> str:
    """Prompt text used for 1-based step `step`."""
    for end, text in sched:
        if step <= end:
            return text
    return sched[-1][1]
