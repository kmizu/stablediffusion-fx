"""Weight Gradient syntax, reimplemented from DingoBite/weight_gradient's README and numeric examples.

No upstream code is used. Exponential/circular curves are standard ease-in/out functions. Gradients restart
in each sampling pass; A1111 editing inside/outside the tags retains its normal hires-offset semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Optional

MODES = ("a", "e", "ei", "eo", "c", "ci", "co")
_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_WEIGHTS = re.compile(rf"\s*({_NUM})\s*-\s*({_NUM})(?:\s*-\s*({_NUM}))?\s*")
_INTERVAL = re.compile(rf"\s*({_NUM}%?)\s*-\s*({_NUM}%?)\s*")


def curve(mode: str, x: float) -> float:
    x = max(0.0, min(1.0, x))
    if x in (0.0, 1.0):
        return x
    if mode == "a":
        return x
    if mode == "ei":
        return 2 ** (10 * x - 10)
    if mode == "eo":
        return 1 - 2 ** (-10 * x)
    if mode == "e":
        return 2 ** (20 * x - 10) / 2 if x < 0.5 else 1 - 2 ** (-20 * x + 10) / 2
    if mode == "ci":
        return 1 - math.sqrt(1 - x * x)
    if mode == "co":
        return math.sqrt(1 - (x - 1) ** 2)
    if mode == "c":
        return (1 - math.sqrt(1 - (2 * x) ** 2)) / 2 if x < 0.5 else (1 + math.sqrt(1 - (2 - 2 * x) ** 2)) / 2
    raise ValueError(f"unknown weight gradient mode: {mode}")


def _fields(body: str) -> Optional[list[str]]:
    fields, stack, start, i = [], [], 0, 0
    while i < len(body):
        c = body[i]
        if c == "\\":
            i += 2
            continue
        if c in "([":
            stack.append(c)
        elif c in ")]":
            if not stack or stack.pop() != ("(" if c == ")" else "["):
                return None
        elif c in "{}":
            return None
        elif c == ":" and not stack:
            fields.append(body[start:i].strip())
            start = i + 1
        i += 1
    return fields + [body[start:].strip()] if not stack else None


def _point(raw: str) -> tuple[float, bool]:
    if raw.endswith("%"):
        return float(raw[:-1]) / 100, True
    value = float(raw)
    fractional = 0 <= value <= 1 and any(c in raw.lower() for c in ".e")
    return value, fractional


@dataclass(frozen=True)
class Gradient:
    tokens: str
    interval: Optional[tuple]
    weights: tuple[float, ...]
    mode: str

    def weight(self, step: int, total: int) -> Optional[float]:
        start, end = 1, total
        if self.interval:
            start, end = (max(1, math.floor(value * total if fractional else value))
                          for value, fractional in self.interval)
        if end < start:
            return None
        x = 1.0 if start == end and step >= end else (step - start) / max(1, end - start)
        x = max(0.0, min(1.0, x))
        a, b = self.weights[:2]
        if len(self.weights) == 3:
            if x <= 0.5:
                x *= 2
            else:
                a, b = b, self.weights[2]
                x = (x - 0.5) * 2
        return a + (b - a) * curve(self.mode, x)


def _parse(body: str) -> Optional[Gradient]:
    fields = _fields(body)
    if not fields or len(fields) not in (3, 4) or not fields[0]:
        return None
    mode = (fields[3] if len(fields) == 4 else "a") or "a"
    weights = _WEIGHTS.fullmatch(fields[2])
    if mode not in MODES or weights is None:
        return None
    values = tuple(float(x) for x in weights.groups() if x is not None)
    if not all(math.isfinite(x) for x in values):
        return None
    interval = None
    if fields[1]:
        match = _INTERVAL.fullmatch(fields[1])
        if match is None:
            return None
        interval = tuple(_point(x) for x in match.groups())
        if not all(math.isfinite(v) for v, _ in interval):
            return None
        if interval[0][1] == interval[1][1] and interval[0][0] > interval[1][0]:
            return None
    return Gradient(fields[0], interval, values, mode)


def _tags(text: str):
    i = 0
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] != "{":
            i += 1
            continue
        end, depth = i + 1, 1
        while end < len(text) and depth:
            if text[end] == "\\":
                end += 2
                continue
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        if depth:
            break
        gradient = _parse(text[i + 1:end - 1])
        if gradient is not None:
            yield i, end, gradient
        i = end


def has_gradient(text: str) -> bool:
    return next(_tags(text), None) is not None


def render(text: str, step: int, total: int) -> str:
    parts, at = [], 0
    for start, end, gradient in _tags(text):
        parts.append(text[at:start])
        weight = gradient.weight(step, total)
        if weight is None:
            parts.append(text[start:end])
        else:
            value = f"{weight:.4f}".rstrip("0").rstrip(".")
            parts.append(f"({gradient.tokens}:{'0' if value == '-0' else value})")
        at = end
    parts.append(text[at:])
    return "".join(parts)


def schedule(text: str, steps: int, hires_steps: Optional[int] = None, gradient_steps: Optional[int] = None) -> list[list]:
    from . import prompt_schedule
    total = gradient_steps if gradient_steps is not None else hires_steps if hires_steps is not None else steps
    if not has_gradient(text):
        return prompt_schedule.schedule(text, steps, hires_steps)
    result = []
    for step in range(1, total + 1):
        expanded = render(text, step, total)
        value = prompt_schedule.text_at(prompt_schedule.schedule(expanded, steps, hires_steps), step)
        if result and result[-1][1] == value:
            result[-1][0] = step
        else:
            result.append([step, value])
    return result
