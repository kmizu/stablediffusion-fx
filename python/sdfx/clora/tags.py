"""Small native extra-network adapter for the AGPL controller grammar."""
import re
from types import SimpleNamespace

TAG = re.compile(r'<(lora|lyco):([^<>]+)>')


def component_prompt(text, unet=False):
    """Select native positional/named multipliers using LoraSyntax precedence."""
    def rewrite(match):
        parts = [part.strip() for part in match[2].split(':')]
        named, positional = {}, []
        for part in parts[1:]:
            if '=' in part:
                key, value = part.split('=', 1)
                named[key.strip().lower()] = value.strip()
            else:
                positional.append(part)
        def number(value):
            try:
                return float(value)
            except (ValueError, TypeError):
                return None
        def choose(key, index, default):
            value = number(named.get(key))
            if value is None and len(positional) > index:
                value = number(positional[index])
            return default if value is None else value
        te = choose('te', 0, 1.)
        value = choose('unet', 1, te) if unet else te
        return f'<{match[1]}:{parts[0]}:{value}>'
    return TAG.sub(rewrite, text)


def parse_prompt(text):
    found = {}
    for match in TAG.finditer(text):
        found.setdefault(match[1], []).append(SimpleNamespace(items=match[2].split(':')))
    return TAG.sub('', text), found
