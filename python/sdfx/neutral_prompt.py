"""Neutral Prompt expressions and denoised-x0 math.

Adapted from ljleb/sd-webui-neutral-prompt, Copyright (c) 2023 ljleb, MIT licence;
see licenses/ljleb__sd-webui-neutral-prompt.txt and THIRD_PARTY_NOTICES.md.
Parsing follows the upstream tree/weight semantics. Math uses an explicit leading batch dimension,
computes each image independently in float32 and guards zero norms/constant tensors.
Torch is imported locally so parsing works in the torch-free dry-run environment.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

KINDS = ("AND_PERP", "AND_SALT", "AND_TOPK", "AND")
_TOKENS = re.compile(r"(\\.|\[|\]|:|\b(?:AND_PERP|AND_SALT|AND_TOPK|AND)\b)")


@dataclass(frozen=True)
class Node:
    kind: str = "AND"
    weight: float = 1.0
    text: str | None = None
    children: tuple[Node, ...] = ()


def _number(text):
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value):
        raise ValueError("Neutral Prompt weights must be finite")
    return value


def parse(text: str, *, max_depth: int = 64) -> Node:
    """Parse AND-family operators; retain ordinary attention/editing square brackets as leaf text."""
    tokens = [t for t in _TOKENS.split(text) if t.strip()]

    def sequence(index, nested=False, depth=0):
        if depth > max_depth:
            raise ValueError(f"Neutral Prompt nesting exceeds {max_depth}")
        nodes = []
        while True:
            kind = "AND"
            if nodes and index < len(tokens) and tokens[index] in KINDS:
                kind, index = tokens[index], index + 1
            start = index
            if index < len(tokens) and tokens[index] == "[":
                children, after = sequence(index + 1, True, depth + 1)
                if after < len(tokens) and tokens[after] == "]":
                    after += 1
                if len(children) > 1:
                    weight = 1.0
                    if after + 1 < len(tokens) and tokens[after] == ":":
                        value = _number(tokens[after + 1].strip())
                        if value is not None:
                            weight, after = value, after + 2
                    nodes.append(Node(kind, weight, children=tuple(children)))
                    index = after
                    if index >= len(tokens) or nested and tokens[index] == "]":
                        return nodes, index
                    continue
            index, bracket_depth, weight = start, 0, 1.0
            parts = []
            while index < len(tokens):
                token = tokens[index]
                if token == "]":
                    if bracket_depth == 0 and nested:
                        break
                    bracket_depth = max(bracket_depth - 1, 0)
                elif token == "[":
                    bracket_depth += 1
                elif token == ":" and index + 1 < len(tokens):
                    # A colon inside editing/attention is text unless its numeric value terminates the leaf.
                    terminates = (index + 2 == len(tokens) or tokens[index + 2] in KINDS or
                                  tokens[index + 2] == "]" and bracket_depth == 0)
                    if terminates:
                        value = _number(tokens[index + 1].strip())
                        if value is not None:
                            weight, index = value, index + 2
                            break
                elif token in KINDS:
                    break
                parts.append(token)
                index += 1
            nodes.append(Node(kind, weight, text="".join(parts)))
            if index >= len(tokens) or nested and tokens[index] == "]":
                return nodes, index
            # An operator at the beginning creates an empty ordinary base, just like upstream.

    nodes, _ = sequence(0)
    return Node(children=tuple(nodes))


def leaves(root: Node) -> list[Node]:
    if root.text is not None:
        return [root]
    return [leaf for child in root.children for leaf in leaves(child)]


def _flat(vector):
    if vector.ndim < 2:
        raise ValueError("Neutral Prompt tensors need a leading batch dimension and at least one feature dimension")
    return vector.float().flatten(1)


def perpendicular(normal, vector):
    """Remove vector's component parallel to normal independently for every image."""
    import torch
    n, v = _flat(normal), _flat(vector)
    scale = n.abs().amax(1, keepdim=True)
    n = n / torch.where(scale > 0, scale, 1)
    norm2 = (n * n).sum(1, keepdim=True)
    projection = (n * v).sum(1, keepdim=True) / torch.where(norm2 > 0, norm2, 1)
    return (v - n * projection).reshape_as(vector)


def _zero_noise_winner(maps):
    """Compare softmax Taylor terms; equal first terms need the centered distribution's higher moments."""
    import torch
    centered = [v.double() - v.double().mean(1, keepdim=True) for v in maps]
    winner = torch.stack(centered).argmax(0)
    for candidate in range(1, len(centered)):
        for previous in range(candidate):
            ties = (winner == previous) & (centered[candidate] == centered[previous])
            for batch in ties.any(1).nonzero().flatten().tolist():
                left, right = centered[previous][batch], centered[candidate][batch]
                if torch.equal(left.sort().values, right.sort().values):
                    continue  # equal centered multisets give identical normalizers at every noise level
                scale = torch.maximum(left.abs().max(), right.abs().max())
                left, right = left / scale, right / scale
                # For two finite N-point distributions, moments through 2N-1 distinguish unequal multisets.
                # Float64-indistinguishable moments retain the earlier branch, just like ordinary argmax ties.
                for order in range(2, 2 * left.numel()):
                    lm, rm = left.pow(order).mean(), right.pow(order).mean()
                    if lm != rm:
                        if rm < lm:
                            winner[batch][ties[batch]] = candidate
                        break
    return winner


def salient_blend(normal, vectors, *, zero_noise_limit=False):
    """Return the auxiliary delta: the winning vector replaces normal, weighted after map selection."""
    import torch
    all_vectors = [normal] + [v for v, _ in vectors]
    maps = [_flat(v.abs()) for v in all_vectors]
    winner = (_zero_noise_winner(maps) if zero_noise_limit else
              torch.stack([torch.softmax(v, dim=1) for v in maps]).argmax(0)).reshape_as(normal)
    result = torch.zeros_like(normal, dtype=torch.float32)
    for index, (vector, weight) in enumerate(vectors, 1):
        result += weight * (winner == index) * (vector.float() - normal.float())
    return result


def topk(vector, ratio=0.05):
    """Upstream kthvalue threshold (inclusive ties); at least one feature for tiny inputs."""
    if not 0 < ratio <= 1:
        raise ValueError("top-k ratio must be in (0, 1]")
    v = _flat(vector)
    k = max(1, int(v.shape[1] * (1 - ratio)))
    threshold = v.abs().kthvalue(k, dim=1, keepdim=True).values
    return (v * (v.abs() >= threshold)).reshape_as(vector)


def rescale(guided, conditional, phi):
    """Interpolate both mean and std toward the unscaled conditional result (per-image, upstream formula)."""
    if not math.isfinite(phi) or not 0 <= phi <= 1:
        raise ValueError("CFG rescale must be between 0 and 1")
    if phi == 0:
        return guided
    import torch
    g, c = _flat(guided), _flat(conditional)
    gm, cm = g.mean(1, keepdim=True), c.mean(1, keepdim=True)
    correction = 1 if g.shape[1] > 1 else 0
    gs, cs = g.std(1, correction=correction, keepdim=True), c.std(1, correction=correction, keepdim=True)
    standardized = (g - gm) / torch.where(gs > 0, gs, 1)
    # Normalize first: cs / gs can overflow even when the final normalized tensor is well behaved.
    return ((1 - phi) * g + phi * (cm + standardized * cs)).reshape_as(guided)


def compose(root: Node, predictions: list, unconditional, cfg: float, *, phi: float = 0):
    """Compose denoised x0 predictions. Each list entry corresponds to one leaf, in depth-first order.

    Leaf weights apply in the parent, once. At a group, auxiliary siblings see the normal sibling sum before
    any auxiliary is added. Nested groups resolve before their parent's weight/operator is applied.
    """
    if not math.isfinite(cfg):
        raise ValueError("CFG must be finite")
    if len(predictions) != len(leaves(root)):
        raise ValueError("Neutral Prompt prediction count does not match leaf count")
    if any(v.shape != unconditional.shape for v in predictions):
        raise ValueError("Neutral Prompt predictions must have the unconditional tensor's shape")
    _flat(unconditional)
    u = unconditional.float()
    total = combine_deltas(root, [v.float() - u for v in predictions])
    return rescale(u + cfg * total, u + total, phi)


def combine_deltas(root: Node, values: list, *, zero_noise_limit=False):
    """Resolve the tree from x0 deltas directly, avoiding cancellation against a shared large x0 baseline."""
    import torch
    if len(values) != len(leaves(root)) or not values:
        raise ValueError("Neutral Prompt delta count does not match leaf count")
    u = torch.zeros_like(values[0], dtype=torch.float32)
    if any(v.shape != u.shape for v in values):
        raise ValueError("Neutral Prompt deltas must have equal shapes")
    predictions = iter(values)

    def delta(node):
        if node.text is not None:
            return next(predictions).float()
        children = [(child, delta(child)) for child in node.children]
        normal = torch.zeros_like(u)
        for child, value in children:
            if child.kind == "AND":
                normal += child.weight * value
        auxiliary, salient = torch.zeros_like(u), []
        for child, value in children:
            if child.kind == "AND_PERP":
                auxiliary += child.weight * perpendicular(normal, value)
            elif child.kind == "AND_TOPK":
                auxiliary += child.weight * topk(value)
            elif child.kind == "AND_SALT":
                salient.append((value, child.weight))
        return normal + auxiliary + salient_blend(normal, salient, zero_noise_limit=zero_noise_limit)

    return delta(root)
