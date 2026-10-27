"""Pure per-image interrogation prompt composition.

Behavior is informed by Img2img Batch Interrogator at commit
ed91cec663bcddd7ca5514264a61483c7d80f8d2, MIT, Santiago Alvarez.
The source-derived emoticon exception list retains the upstream notice in
licenses/Alvi-alvarez__sd-Img2img-batch-interrogator.txt.
No model is loaded here. Each call builds from immutable submitted prompts.
Unsafe control characters are rejected and replacement strings are literal;
unlike upstream regex substitution, backslashes are not replacement escapes.
The >_< emoticon is preserved in punctuation mode as in the pinned source.
"""
from dataclasses import dataclass
import math
import re
import uuid

MAX_TEXT = 131_072
MAX_RESULTS = 256
MAX_PAIRS = 1024
MAX_REPLACEMENT_WORK = 1_000_000

# Text-emoticon exceptions from the pinned source (MIT); preserve order so
# overlapping emoticons are treated consistently with that source.
_EMOTICONS = (
    "'s", "...", ":-)", ":)", ":-]", ":]", ":->", ":>", "8-)", "8)", ":-}", ":}", ":^)", "=]", "=)", ":-D", ":D", "8-D", "8D", "=D", "=3", "B^D",
    "c:", "C:", "x-D", "X-D", ":-))", ":))", ":-(", ":(", ":-c", ":c", ":-<", ":<", ":-[", ":[", ":-||", ":{", ":@", ";(", ":'-(", ":'(", ":=(", ":'-)",
    ":')", ">:(", ">:[", "D-':", "D:<", "D:", "D;", "D=", ":-O", ":O", ":-o", ":o", ":-0", ":0", "8-0", ">:O", "=O", "=o", "=0", ":-3", ":3", "=3", ">:3",
    ":-*", ":*", ":x", ";-)", ";)", "*-)", "*)", ";-]", ";]", ";^)", ";>", ":-,", ";D", ";3", ":-P", ":P", "X-P", "x-p", ":-p", ":p", ":-Þ", ":Þ", ":-þ",
    ":þ", ":-b", ":b", "d:", "=p", ">:P", ":-/", ":/", ":-.", ">:/", "=/", ":L", "=L", ":S", ":-|", ":|", ":$", "://)", "://3", ":-X", ":X", ":-#", ":#",
    ":-&", ":&", "O:-)", "O:)", "0:-3", "0:3", "0:-)", "0:)", "0;^)", ">:-)", ">:)", "}:-)", "}:)", "3:-)", "3:)", ">;-)", ">;)", ">:3", ">;3", "|;-)", "|-O",
    "B-)", ":-J", "#-)", "%-)", "%)", ":-###..", ":###..", "<:-|", "',:-|", "',:-l", ":E", "8-X", "8=X", "x-3", "x=3", "~:>", "@};-", "@}->--", "@}-;-'---",
    "@>-->--", "8====D", "8===D", "8=D", "3=D", "8=>", "8===D~~~", "*<|:-)", "</3", "<\\3", "<3", "><>", "<><", "<*)))-{", "><(((*>", "\\o/", "*\\0/*", "o7",
    "v.v", "._.", "._.;", "X_X", "x_x", "+_+", "X_x", "x_X", "<_<", ">_>", "<.<", ">.>", "O_O", "o_o", "O-O", "o-o", "O_o", "o_O", ">.<", "^5", "o/\\o",
    ">_>^ ^<_<", "V.v.V", ">_<",
)


@dataclass(frozen=True)
class BatchInterrogateOptions:
    position: str = "prepend"
    reverse: bool = False
    exaggeration: bool = False
    filter_positive: bool = False
    filter_negative: bool = False
    use_custom_filter: bool = False
    custom_filter: str = ""
    replacements: tuple[tuple[str, str], ...] = ()
    remove_punctuation: bool = False
    weight: float | None = None


@dataclass(frozen=True)
class ComposedPrompts:
    positive: str
    negative: str
    interrogation: str


def _text(value):
    if not isinstance(value, str) or len(value) > MAX_TEXT:
        raise ValueError("prompt text must be a bounded string")
    if any((ord(c) < 32 and c not in "\t\n\r") or ord(c) == 127 for c in value):
        raise ValueError("control characters are not supported")
    return value


def _comparison(text):
    # Preserve escaped parentheses while ignoring attention parentheses and
    # numeric colon weights in the original's filter comparison.
    text = re.sub(r":\d+(\.\d+)?", "", text)
    text = re.sub(r"(?<!\\)[()]", "", text)
    return text.strip()


def _filter_tags(tags, text):
    excluded = {_comparison(item.strip()) for item in text.split(",")}
    return [tag for tag in tags if _comparison(tag.strip()) not in excluded]


def _placeholder_prefix(text):
    """Select a marker in a fixed number of bounded scans; fail on collision."""
    for _ in range(3):
        prefix = "SDFXEMOTICON" + uuid.uuid4().hex + "TOKEN"
        if prefix not in text:
            return prefix
    raise ValueError("unable to select a unique emoticon marker")

def _punctuation(text):
    prefix = _placeholder_prefix(text)
    preserved = []
    for emoticon in _EMOTICONS:
        if emoticon in text:
            marker = f"{prefix}{len(preserved)}TOKEN"
            text = text.replace(emoticon, marker)
            preserved.append((marker, emoticon))
    text = re.sub(r"[^\w\s,]", "", text)
    text = ", ".join(part.strip() for part in text.split(",") if part.strip())
    for marker, emoticon in preserved:
        text = text.replace(marker, emoticon)
    return text


def compose(original_positive, original_negative, ordered_results, options):
    """Compose results without changing either original prompt or result list."""
    positive, negative = _text(original_positive), _text(original_negative)
    if not isinstance(options, BatchInterrogateOptions):
        raise ValueError("invalid batch interrogation options")
    if options.position not in ("prepend", "append"):
        raise ValueError("position must be prepend or append")
    for flag in (options.reverse, options.exaggeration, options.filter_positive,
                 options.filter_negative, options.use_custom_filter, options.remove_punctuation):
        if type(flag) is not bool:
            raise ValueError("batch interrogation flags must be boolean")
    _text(options.custom_filter)
    if options.weight is not None:
        try:
            valid_weight = (not isinstance(options.weight, bool) and isinstance(options.weight, (int, float))
                            and math.isfinite(options.weight) and 0 <= options.weight <= 1)
        except OverflowError:
            valid_weight = False
        if not valid_weight:
            raise ValueError("weight must be finite and between 0 and 1")
    if not isinstance(ordered_results, (list, tuple)) or len(ordered_results) > MAX_RESULTS:
        raise ValueError("too many interrogation results")
    if not isinstance(options.replacements, tuple) or len(options.replacements) > MAX_PAIRS:
        raise ValueError("invalid replacement pairs")
    replacement_size = 0
    for pair in options.replacements:
        if not isinstance(pair, tuple) or len(pair) != 2 or not _text(pair[0]) or not isinstance(pair[1], str):
            raise ValueError("invalid replacement pair")
        _text(pair[1])
        replacement_size += len(pair[0]) + len(pair[1])
        if replacement_size > MAX_TEXT:
            raise ValueError("replacement input exceeds limit")
    texts = [_text(item) for item in ordered_results]
    if sum(map(len, texts)) + max(0, len(texts) - 1) > MAX_TEXT:
        raise ValueError("combined interrogation exceeds limit")
    tags = [item.strip() for item in ", ".join(texts).split(",") if item.strip()]
    if not options.exaggeration:
        tags = list(dict.fromkeys(tags))
    text = ", ".join(tags)
    if len(text) * len(options.replacements) > MAX_REPLACEMENT_WORK:
        raise ValueError("replacement work exceeds budget")
    work = 0
    for old, new in options.replacements:
        work += len(text)
        if work > MAX_REPLACEMENT_WORK:
            raise ValueError("replacement work exceeds budget")
        pattern = re.compile(r"\b" + re.escape(old) + r"\b")
        matches = sum(1 for _ in pattern.finditer(text))
        if len(text) + matches * max(0, len(new) - len(old)) > MAX_TEXT:
            raise ValueError("replacement output exceeds limit")
        text = pattern.sub(lambda _: new, text)
    tags = [item.strip() for item in text.split(",") if item.strip()]
    if options.filter_positive:
        tags = _filter_tags(tags, positive)
    if options.filter_negative:
        tags = _filter_tags(tags, negative)
    if options.use_custom_filter:
        tags = _filter_tags(tags, options.custom_filter)
    text = ", ".join(tags)
    if options.remove_punctuation:
        text = _punctuation(text)
    if text and options.weight is not None:
        text = f"({text}:{options.weight})"
    original = negative if options.reverse else positive
    if not text:
        result = original
    elif not original:
        result = text
    elif options.position == "append":
        result = f"{original.rstrip(', ')}, {text}"
    else:
        result = f"{text}, {original}"
    if len(result) > MAX_TEXT:
        raise ValueError("composed prompt exceeds limit")
    return ComposedPrompts(positive if options.reverse else result,
                           result if options.reverse else negative, text)






