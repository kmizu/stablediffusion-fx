"""Small native extra-network adapter for the AGPL controller grammar."""
import re
from types import SimpleNamespace

TAG = re.compile(r'<(lora|lyco):([^<>]+)>')


def parse_prompt(text):
    found = {}
    for match in TAG.finditer(text):
        found.setdefault(match[1], []).append(SimpleNamespace(items=match[2].split(':')))
    return TAG.sub('', text), found
