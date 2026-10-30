"""Produce test fixtures from pinned AGPL Hires Fix Tweaks pure prompt functions."""
import ast
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

pin = '28902f742a795afc31d86641ad85466afdd280e6'
path = Path('outputs/hires-fix-tweaks-source/hires_fix_tweaks/hr_modules/hr_prompt_mode.py')
assert hashlib.sha256(path.read_bytes()).hexdigest() == '31903bb40f960ae7a543997e917a3f3afe99379aa1e88e7512b9725e98bb7729'
names = {'setup_regex', 'hires_prompt_mode_default', 'hires_prompt_mode_append',
         'hires_prompt_mode_prepend', 'hires_prompt_mode_search_replace'}
nodes = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names]
options = SimpleNamespace(hires_fix_tweaks_marker_char='@', hires_fix_tweaks_append_separator='{newline}',
                          hires_fix_tweaks_prepend_separator='{newline}')
environment = dict(re=re, shared=SimpleNamespace(opts=options),
    remove_extra_networks=lambda text: re.sub(r'<(\w+):([^>]+)>', '', text),
    one_leading_and_trailing_newline_pattern=re.compile(r'^\r?\n?([\W\w]*)\r?\n?$'))
exec(compile(ast.Module(nodes, []), str(path), 'exec'), environment)
functions = dict(Default='default', Append='append', Prepend='prepend', **{'Prompt S/R': 'search_replace'})
cases = []

def add(base, hires, mode, remove=False, separator='{newline}', marker='@'):
    options.hires_fix_tweaks_marker_char = marker
    options.hires_fix_tweaks_append_separator = options.hires_fix_tweaks_prepend_separator = separator
    environment['setup_regex']()
    result = environment['hires_prompt_mode_' + functions[mode]](base, hires, remove)
    cases.append(dict(base=base, hires=hires, mode=mode, remove=remove, separator=separator,
                      marker=marker, expected=list(result)))

for mode in ('Default', 'Append', 'Prepend'):
    for base, hires in [('cat', ''), ('cat <lora:x:1> embedding', 'dog'), ('cat <hypernet:x:1>', '  '),
                        ('cat\n犬', 'red\r\nblue'), ('a', '{braces}')]:
        for remove in (False, True):
            add(base, hires, mode, remove)
for marker in ('@', '#', '*', '\\', '★', '😀'):
    add('cat '+marker+'slot'+marker+' red', marker+'slot'+marker+'blue\n'+marker+'red'+marker+'green', 'Prompt S/R', marker=marker)
    add('a'+marker+'b', marker+'a'+marker*2+'b'+marker+'x'+marker*2+'y', 'Prompt S/R', marker=marker)
for base, instructions in [('a\nb', '@a\nb@multi\nline\n'), ('red red', '@red@blue\n@blue@'),
                           ('cat @slot@', 'ignored prefix\n@slot@\r\nnew\r\n'), ('cat', 'no instruction')]:
    add(base, instructions, 'Prompt S/R')
add('a', 'b', 'Append', separator='{{literal}}{newline}')
target = Path('core/src/test/resources/hires-prompt-reference.json')
target.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes((json.dumps(dict(pin=pin, cases=cases), ensure_ascii=False, indent=2)+'\n').encode('utf-8'))
print(f'Generated {len(cases)} pinned reference cases')
