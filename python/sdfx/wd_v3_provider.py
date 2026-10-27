"""Internal offline WD v3 provider; not public WD EXT parity or capability.

Thresholds and underscore exceptions reflect the MIT-licensed batch extension
at ed91cec663bcddd7ca5514264a61483c7d80f8d2; its notice is archived under
licenses/Alvi-alvarez__sd-Img2img-batch-interrogator.txt. Tag ordering follows
the CSV until the WD EXT implementation is independently verified.
"""
from dataclasses import dataclass
import math

from .batch_interrogate import MAX_TEXT
from .wd_v3_assets import AssetPin, MAX_TAG_ROWS
from .wd_v3_runtime import ScoredTag, run_internal

_EMOTICONS = frozenset(("0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<",
                        "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"))


@dataclass(frozen=True)
class WdV3Options:
    threshold: float = .35
    remove_underscores: bool = True
    append_ratings: bool = False
    rating_threshold: float = .5


class WdV3Provider:
    input_mode = "RGBA"
    model_id = "SmilingWolf/wd-vit-tagger-v3@7f6b584d0bd3f55c4531f14ba3d4761b2bccdc0f"

    def __init__(self, model, tags, temp_parent, options=WdV3Options()):
        if not isinstance(model, AssetPin) or not isinstance(tags, AssetPin) or not isinstance(options, WdV3Options):
            raise ValueError("explicit local WD assets and options required")
        for threshold in (options.threshold, options.rating_threshold):
            if type(threshold) not in (int, float) or not 0 <= threshold <= 1 or not math.isfinite(threshold):
                raise ValueError("WD thresholds must be finite values in 0..1")
        if type(options.remove_underscores) is not bool or type(options.append_ratings) is not bool:
            raise ValueError("invalid WD boolean option")
        self._model, self._tags, self._temp_parent, self._options = model, tags, temp_parent, options

    def provenance(self):
        return {"model_id": self.model_id, "model_sha256": self._model.sha256,
                "tags_sha256": self._tags.sha256, "threshold": self._options.threshold,
                "remove_underscores": self._options.remove_underscores,
                "append_ratings": self._options.append_ratings,
                "rating_threshold": self._options.rating_threshold,
                "preprocessing": "exif-white-alpha-square-pillow-bicubic-bgr-f32-v1"}

    def interrogate(self, image, ctx):
        scores = run_internal(self._model, self._tags, self._temp_parent, image, ctx)
        if not isinstance(scores, tuple) or len(scores) > MAX_TAG_ROWS:
            raise ValueError("invalid WD scores")
        selected = []
        ratings = []
        total = 0
        for row in scores:
            if (not isinstance(row, ScoredTag) or not isinstance(row.name, str) or not row.name
                    or len(row.name) > 512 or row.category not in (0, 4, 9)
                    or type(row.score) not in (int, float) or not math.isfinite(row.score)
                    or not 0 <= row.score <= 1):
                raise ValueError("invalid WD score")
            if row.category in (0, 4) and row.score > self._options.threshold:
                name = row.name.replace("_", " ") if self._options.remove_underscores and row.name not in _EMOTICONS else row.name
                selected.append(name)
                total += len(name) + 2
            elif row.category == 9 and self._options.append_ratings and row.score >= self._options.rating_threshold:
                ratings.append(row.name)
                total += len(row.name) + 2
            if total > MAX_TEXT:
                raise ValueError("WD output exceeds text limit")
        return ", ".join(selected + ratings)

    def unload(self):
        # run_internal owns and releases each CPU session inside the call;
        # this provider does not retain model/session state between calls.
        return None
