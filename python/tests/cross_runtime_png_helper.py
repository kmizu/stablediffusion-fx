"""File-based bridge used by PixelScrambleSuite; called in a separate Python process."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sdfx.encrypt_image import encode_png, decode_png

action, source, target = sys.argv[1:]
data = Path(source).read_bytes()
if action == "encode":
    result = encode_png(data, "cross-runtime-secret")
elif action == "decode":
    result, verified = decode_png(data, "cross-runtime-secret")
    if verified is not True:
        raise ValueError("password verifier missing")
else:
    raise ValueError("unknown action")
Path(target).write_bytes(result)
