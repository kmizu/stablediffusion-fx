"""Pure model-keyword compatibility primitives; independent of the worker and UI."""
import csv
import difflib
import hashlib
import io
import math
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


UPSTREAM_LORA_URL = ('https://raw.githubusercontent.com/mix1009/model-keyword/'
                     'c3d0833a9434bb41ecd7ea715b5abd5e6b43a5af/lora-keyword.txt')
UPSTREAM_LORA_SHA256 = 'b20edcfd8f121c6a5ce9012e4be7ae587a4a66b85f2cd0328768523df78db5ec'
UPSTREAM_LORA_BYTES = 31395702


def install_upstream_lora_catalog(directory, *, confirmed, open_url=None,
                                  expected_sha256=UPSTREAM_LORA_SHA256, expected_size=UPSTREAM_LORA_BYTES):
    """Install the pinned upstream catalog on explicit user request; never replace a file."""
    if confirmed is not True:
        raise ValueError('explicit confirmation is required to download the catalog')
    if open_url is None:
        from urllib.request import urlopen
        open_url = lambda url: urlopen(url, timeout=30)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / 'lora-keyword.txt'
    if destination.exists():
        raise FileExistsError('LoRA catalog already exists; it was not replaced')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.model-keyword-download-', dir=directory, delete=False) as output:
            temporary = Path(output.name)
            digest, count = hashlib.sha256(), 0
            with open_url(UPSTREAM_LORA_URL) as source:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > expected_size:
                        raise ValueError('download exceeds the pinned catalog size')
                    digest.update(chunk)
                    output.write(chunk)
            if count != expected_size or digest.hexdigest() != expected_sha256:
                raise ValueError('download does not match the pinned catalog hash and size')
            output.flush(); os.fsync(output.fileno())
        os.link(temporary, destination)
        return str(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class Entry:
    model_hash: str
    keywords: str
    filename: str
    source: int


class IterationState:
    def __init__(self):
        self._lock = threading.RLock()
        self._model = self._lora = 0

    @contextmanager
    def transaction(self):
        with self._lock:
            initial = self._model, self._lora
            try:
                yield
            except BaseException:
                self._model, self._lora = initial
                raise

    def reserve(self, count, model_keywords, model_choice, lora_keywords, lora_choice):
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError('count must be a nonnegative integer')
        with self._lock:
            indices = (self._model, self._lora)
            if model_choice == 'iterate' and model_keywords is not None and '|' in model_keywords:
                self._model += count
            if lora_choice == '< iterate >' and lora_keywords is not None and '|' in lora_keywords:
                self._lora += count
            return indices


class MappingIndex:
    def __init__(self, default_path, user_path, allow_missing_default=False):
        self.paths = (Path(default_path), Path(user_path))
        self.allow_missing_default = allow_missing_default
        self._signature = None
        self._entries = {}
        self._lock = threading.RLock()

    def lookup(self, model_hash, filename):
        with self._lock:
            self.paths[1].parent.mkdir(parents=True, exist_ok=True)
            with _process_lock(self.paths[1]):
                return self._lookup_locked(model_hash, filename)

    def _lookup_locked(self, model_hash, filename):
        def signature(path):
            try:
                stat = path.stat()
                return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns
            except FileNotFoundError:
                return None
        current = tuple(signature(p) for p in self.paths)
        if current != self._signature:
            entries = {}
            for source, path in enumerate(self.paths):
                if path.exists():
                    for entry in parse_rows(path.read_text(encoding='utf8'), source):
                        entries.setdefault(entry.model_hash, []).append(entry)
                elif source == 0 and not self.allow_missing_default:
                    raise FileNotFoundError(path)
            self._entries = entries
            self._signature = current
        return find_match(self._entries.get(model_hash.lower(), ()), model_hash, filename)


class KeywordPlanner:
    def __init__(self, checkpoint_index, lora_index, iteration):
        self.checkpoint_index, self.lora_index, self.iteration = checkpoint_index, lora_index, iteration

    def plan(self, preview, prompts, checkpoint, lora, options, rng):
        if not isinstance(preview, str) or not isinstance(prompts, (list, tuple)) or len(prompts) > 1600 or any(
                not isinstance(prompt, str) for prompt in prompts):
            raise ValueError('invalid model keyword prompts')
        if options.get('placement', 'keyword prompt') not in ('keyword prompt', 'prompt keyword', 'keyword, prompt', 'prompt, keyword'):
            raise ValueError('invalid model keyword placement')
        if options.get('order', 'textual inversion first') not in ('textual inversion first', 'model keyword first'):
            raise ValueError('invalid model keyword order')
        multiplier = options.get('multiplier', options.get('lora_weight', 0.7))
        if isinstance(multiplier, bool) or not isinstance(multiplier, (float, int)) or not math.isfinite(multiplier) or not 0 <= multiplier <= 2:
            raise ValueError('invalid LoRA multiplier')
        checkpoint = Path(checkpoint)
        checkpoint_hash = legacy_hash(checkpoint)
        model_entry = self.checkpoint_index.lookup(checkpoint_hash, checkpoint.name)
        lora_name = (options.get('lora_name') or Path(lora).name) if lora else None
        if lora and self.lora_index is None:
            raise ValueError('selected LoRA requires a mapping index')
        lora_hash = legacy_hash(lora) if lora else None
        lora_entry = self.lora_index.lookup(lora_hash, lora_name) if lora else None
        model_words = model_entry.keywords if model_entry else None
        lora_words = lora_entry.keywords if lora_entry else None
        model_choice = options.get('model_choice', 'keyword1, keyword2')
        lora_choice = options.get('lora_choice', 'None')
        if lora and lora_choice not in keyword_choices('lora',
                [word.strip() for word in lora_words.split('|')] if lora_words else []):
            lora_choice = 'None'
        indices = self.iteration.reserve(len(prompts), model_words, model_choice, lora_words, lora_choice)
        embedding = options.get('embedding')
        if embedding in ('', 'None'):
            embedding = None
        common = (model_words, model_choice, lora_words, lora_choice, embedding,
                  options.get('order', 'textual inversion first'), options.get('placement', 'keyword prompt'),
                  lora_name, multiplier, *indices, rng)
        rendered, generated = plan_with_preview(preview, prompts, *common)
        def source(entry, default):
            return ('custom' if entry.source else default) if entry else 'no match'
        model_info = dict(filename=checkpoint.name, legacy_hash=checkpoint_hash, mapping_source=source(model_entry, 'default'))
        lora_info = dict(filename=lora_name, legacy_hash=lora_hash, mapping_source=source(lora_entry, 'user catalog')) if lora else None
        return dict(preview=rendered, prompts=generated, checkpoint=model_info, lora=lora_info)


def _file_signature(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def legacy_hash(path):
    path = Path(path)
    with path.open('rb') as stream:
        initial = _file_signature(os.fstat(stream.fileno()))
        stream.seek(0x100000)
        result = hashlib.sha256(stream.read(0x10000)).hexdigest()[:8]
        if _file_signature(os.fstat(stream.fileno())) != initial or _file_signature(path.stat()) != initial:
            raise ValueError('model changed while hashing')
        return result


def inspect_model(path, index, match_name=None):
    path = Path(path)
    model_hash = legacy_hash(path)
    name = match_name if match_name else path.name
    entry = index.lookup(model_hash, name)
    return dict(filename=name, hash=model_hash,
                keywords=[word.strip() for word in entry.keywords.split('|')] if entry else [],
                source=('custom' if entry.source else 'default') if entry else 'no match')


def mapping_action(action, path, index, keywords=None, match_name=None):
    if action not in ('check', 'save', 'delete'):
        raise ValueError('unknown model keyword action')
    path = Path(path)
    info = inspect_model(path, index, match_name)
    if action == 'save':
        if not isinstance(keywords, str):
            raise ValueError('keywords are required')
        save_custom(index.paths[1], info['hash'], keywords, info['filename'])
        return inspect_model(path, index, match_name)
    if action == 'delete':
        deleted = delete_custom(index.paths[1], info['hash'], info['filename'])
        return dict(inspect_model(path, index, match_name), deleted=deleted)
    return info


def parse_rows(text, source):
    entries = []
    for row in csv.reader(io.StringIO(text, newline=''), skipinitialspace=True):
        if len(row) < 2 or not row[0].strip() or row[0].lstrip().startswith('#'):
            continue
        entries.append(Entry(row[0].strip().lower(), row[1].strip(),
                             row[2].strip() if len(row) > 2 else 'default', source))
    return entries


def find_match(rows, model_hash, name):
    found, best = None, -1.0
    for row in rows:
        if row.model_hash == model_hash.lower():
            score = difflib.SequenceMatcher(None, row.filename, name).ratio()
            if score >= best:
                found, best = row, score
    return found


def choose(keywords, mode, index, rng):
    if keywords is None:
        return None
    values = [item.strip(' ') for item in keywords.split('|')]
    if len(values) == 1:
        return values[0]
    if mode == 'keyword1, keyword2' or mode == ', '.join(values):
        return ', '.join(values)
    if mode in ('random', '< random >'):
        return rng.choice(values)
    if mode in ('iterate', '< iterate >'):
        return values[index % len(values)]
    if mode == 'keyword1':
        return values[0]
    if mode == 'keyword2':
        return values[1]
    return mode if mode in values else values[0]


def render_prompt(prompt, keyword, embedding, lora_keyword, order, placement, lora_model, multiplier):
    parts = ([keyword, lora_keyword, embedding] if order == 'model keyword first'
             else [embedding, lora_keyword, keyword])
    parts = [part for part in parts if part is not None]
    if placement.startswith('keyword'):
        parts.append(prompt)
    else:
        parts.insert(0, prompt)
    if lora_model:
        name = lora_model.replace('\\', '/').rsplit('/', 1)[-1].rsplit('.', 1)[0]
        parts.insert(0, f'<lora:{name}:{multiplier}>')
    return (', ' if ',' in placement else ' ').join(parts)


_mapping_lock = threading.RLock()


@contextmanager
def _process_lock(path):
    """Coordinate multiple app processes editing the same custom mapping file."""
    lock_path = path.with_name(path.name + '.lock')
    with lock_path.open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            deadline = time.monotonic() + 30
            while True:
                try:
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('timed out waiting for mapping file lock')
                    time.sleep(.01)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def _edit_custom(path, model_hash, filename, keywords):
    path = Path(path)
    if not model_hash.strip() or not filename.strip():
        raise ValueError('model hash and filename are required')
    if keywords is not None and not keywords.strip():
        raise ValueError('keywords are required')
    with _mapping_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _process_lock(path):
            return _edit_locked(path, model_hash, filename, keywords)


def _edit_locked(path, model_hash, filename, keywords):
    original = path.read_bytes() if path.exists() else b''
    text = original.decode('utf8')
    lines = text.splitlines(keepends=True)
    reader = csv.reader(io.StringIO(text, newline=''), skipinitialspace=True, strict=True)
    records, previous = [], 0
    try:
        for row in reader:
            records.append((row, ''.join(lines[previous:reader.line_num])))
            previous = reader.line_num
    except csv.Error as error:
        raise ValueError(f'cannot edit malformed mapping CSV: {error}') from error
    def matches(row):
        return (len(row) >= 3 and not row[0].lstrip().startswith('#') and
                row[0].strip().lower() == model_hash.strip().lower() and row[2].strip() == filename)
    found = any(matches(row) for row, _ in records)
    retained = ''.join(raw for row, raw in records if not matches(row))
    if keywords is not None:
        addition = io.StringIO(newline='')
        csv.writer(addition, lineterminator='\n').writerow([model_hash.strip().lower(), keywords.strip(), filename])
        if retained and not retained.endswith(('\r', '\n')):
            retained += '\n'
        retained += addition.getvalue()
    elif not found:
        return False
    if path.exists():
        backup_fd, backup_name = tempfile.mkstemp(prefix=path.name + '.backup-', dir=path.parent)
        try:
            with os.fdopen(backup_fd, 'wb') as backup:
                backup.write(original)
        except BaseException:
            os.unlink(backup_name)
            raise
    fd, name = tempfile.mkstemp(prefix='.model-keyword-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf8', newline='') as stream:
            stream.write(retained)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return found


def save_custom(path, model_hash, keywords, filename):
    _edit_custom(path, model_hash, filename, keywords)


def delete_custom(path, model_hash, filename):
    return _edit_custom(path, model_hash, filename, None)


def plan_prompts(prompts, model_keywords, model_choice, lora_keywords, lora_choice, embedding,
                 order, placement, lora_model, multiplier, model_index, lora_index, rng):
    return [render_prompt(prompt, choose(model_keywords, model_choice, model_index + i, rng), embedding,
                          choose(lora_keywords, lora_choice, lora_index + i, rng)
                          if lora_model and lora_choice != 'None' else None, order, placement,
                          lora_model, multiplier) for i, prompt in enumerate(prompts)]


def plan_for_model(prompts, checkpoint, checkpoint_index, lora, lora_index, options, rng):
    checkpoint = Path(checkpoint)
    model = checkpoint_index.lookup(legacy_hash(checkpoint), checkpoint.name)
    lora_name = (options.get('lora_name') or Path(lora).name) if lora else None
    lora_entry = (lora_index.lookup(legacy_hash(lora), lora_name or Path(lora).name)
                  if lora else None)
    return plan_prompts(prompts, model.keywords if model else None, options.get('model_choice', 'keyword1, keyword2'),
        lora_entry.keywords if lora_entry else None, options.get('lora_choice', 'None'), options.get('embedding'),
        options.get('order', 'textual inversion first'), options.get('placement', 'keyword prompt'),
        lora_name, options.get('multiplier', 0.7), options.get('model_index', 0), options.get('lora_index', 0), rng)


def plan_with_preview(preview, prompts, model_keywords, model_choice, lora_keywords, lora_choice, embedding,
                      order, placement, lora_model, multiplier, model_index, lora_index, rng):
    def render(items):
        return plan_prompts(items, model_keywords, model_choice, lora_keywords, lora_choice, embedding,
                            order, placement, lora_model, multiplier, model_index, lora_index, rng)
    return render([preview])[0], render(prompts)


def compose_per_item(trees, branches, leaf_count, combine, concatenate):
    if not trees or not branches or any(branch.shape[0] != len(trees) for branch in branches):
        raise ValueError('branch batches must match the number of image trees')
    result = []
    for index, tree in enumerate(trees):
        count = leaf_count(tree)
        if count < 1 or count > len(branches):
            raise ValueError('image tree has invalid branch count')
        result.append(combine(tree, [branch[index:index + 1] for branch in branches[:count]]))
    return concatenate(result)


def transpose_branches(items):
    if not items or any(not branches for branches in items):
        raise ValueError('each image must have at least one branch')
    return [[branches[index] if index < len(branches) else branches[0] for branches in items]
            for index in range(max(map(len, items)))]


def assemble_neutral(items, fallback_trees, collate):
    if not items or len(items) != len(fallback_trees):
        raise ValueError('one fallback tree is required for each image')
    trees, branches, phi = [], [], None
    for item, fallback in zip(items, fallback_trees):
        plan = item.get('_neutral')
        if plan is None:
            trees.append(fallback)
            branches.append([item])
        else:
            if phi is not None and phi != plan.phi:
                raise ValueError('per-image Neutral rescale values differ')
            phi = plan.phi
            trees.append(plan.tree)
            branches.append(plan.branches)
    return trees, [collate(group) for group in transpose_branches(branches)], phi if phi is not None else 0


def civitai_keywords(response, sha256):
    if not isinstance(sha256, str) or not re.fullmatch('[0-9a-fA-F]{64}', sha256):
        raise ValueError('a full SHA-256 is required')
    if not isinstance(response, dict) or not isinstance(response.get('files'), list):
        raise ValueError('invalid model version response')
    matches = any(isinstance(file, dict) and isinstance(file.get('hashes'), dict) and
                  isinstance(file['hashes'].get('SHA256'), str) and
                  file['hashes']['SHA256'].lower() == sha256.lower() for file in response['files'])
    if not matches:
        raise ValueError('remote model version does not contain the local SHA-256')
    words = response.get('trainedWords', [])
    if not isinstance(words, list) or any(not isinstance(word, str) for word in words):
        raise ValueError('invalid trained words in model version')
    return [word.strip() for word in words if word.strip()]


def keyword_choices(kind, words):
    if kind == 'checkpoint':
        return ['keyword1, keyword2', 'random', 'iterate'] + (list(words) if words else ['keyword1', 'keyword2'])
    if kind == 'lora':
        return ['None'] + ([', '.join(words)] if len(words) > 1 else []) + list(words) + (
            ['< iterate >', '< random >'] if len(words) > 1 else [])
    raise ValueError('unknown model keyword kind')


def batch_extra_networks(prompts):
    """reForge removes every image's tags but activates only the first prompt's networks."""
    expression = re.compile(r'<(\w+):([^>]+)>')
    cleaned, first = [], None
    for prompt in prompts:
        networks = {}
        def found(match):
            networks.setdefault(match.group(1), []).append(match.group(2).split(':'))
            return ''
        cleaned.append(expression.sub(found, prompt))
        if first is None:
            first = networks
    return cleaned, first


def replay_prompts_for_batch(prompts):
    if not prompts:
        raise ValueError('batch must contain at least one prompt')
    clean, _ = batch_extra_networks(prompts)
    tags = re.findall(r'<\w+:[^>]+>', prompts[0])
    # Tags are removed verbatim on replay; separators added here would change
    # the actual text sent to the encoder when a batch activates several.
    prefix = ''.join(tags)
    return [(prefix + value) if prefix else value for value in clean]


def postprocess_keyword_items(images, prompts, seeds, process):
    if len(images) != len(prompts) or len(images) != len(seeds):
        raise ValueError('one prompt and seed are required per image')
    return [process(image, prompt, seed, index)
            for index, (image, prompt, seed) in enumerate(zip(images, prompts, seeds))]


def validate_batch_plan(plan, batch_size, n_iter):
    if (not isinstance(plan, dict) or isinstance(batch_size, bool) or not isinstance(batch_size, int) or
            isinstance(n_iter, bool) or not isinstance(n_iter, int) or not 1 <= batch_size <= 16 or not 1 <= n_iter <= 100):
        raise ValueError('invalid batch plan dimensions')
    prompts, batches = plan.get('prompts'), plan.get('batches')
    if (not isinstance(plan.get('preview'), str) or not isinstance(prompts, list) or
            len(prompts) != batch_size * n_iter or any(not isinstance(prompt, str) for prompt in prompts) or
            not isinstance(batches, list) or len(batches) != n_iter):
        raise ValueError('batch plan count or prompt types do not match the request')
    for index, batch in enumerate(batches):
        clean, networks = batch_extra_networks(prompts[index * batch_size:(index + 1) * batch_size])
        unsupported = sorted(set(networks) - {'lora', 'lyco'})
        if unsupported:
            raise ValueError('unsupported extra network types: ' + ', '.join(unsupported))
        adapters = [dict(kind=kind, **parse_lora_activation(args))
                    for kind, entries in networks.items() if kind in ('lora', 'lyco') for args in entries]
        if (not isinstance(batch, dict) or batch.get('clean_prompts') != clean or batch.get('networks') != networks or
                batch.get('adapters') != adapters):
            raise ValueError(f'batch plan {index} does not match its effective prompts')


def parse_lora_activation(arguments):
    if not isinstance(arguments, list) or any(not isinstance(item, str) for item in arguments):
        raise ValueError('LoRA activation arguments must be strings')
    positional, named = [], {}
    for argument in arguments:
        parts = argument.strip().split('=', 1)
        if len(parts) == 2:
            named[parts[0].strip().lower()] = parts[1].strip()
        else:
            positional.append(argument.strip())
    if not positional or not positional[0]:
        raise ValueError('LoRA activation requires a name')
    te = float(named.get('te', positional[1] if len(positional) > 1 else 1.0))
    unet = float(named.get('unet', positional[2] if len(positional) > 2 else te))
    dyn = named.get('dyn', positional[3] if len(positional) > 3 else None)
    if not math.isfinite(te) or not math.isfinite(unet):
        raise ValueError('LoRA weights must be finite')
    result = dict(name=positional[0], te=te, unet=unet, dyn=int(dyn) if dyn is not None else None)
    if named.get('lbw'):
        result['lbw'] = named['lbw']
    def number(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    step = named.get('step', '').split('-')
    step_start, step_stop = (number(step[0]), number(step[1])) if len(step) == 2 else (None, None)
    start, stop = number(named.get('start')), number(named.get('stop'))
    start, stop = (step_start if start is None else start), (step_stop if stop is None else stop)
    if start is not None:
        result['start'] = start
    if stop is not None:
        result['stop'] = stop
    return result


def generation_batch_loras(batch, manual, paths):
    """Convert first-prompt extra networks while retaining independent manual adapters."""
    from .params import LoraRef
    result = list(manual)
    for adapter in batch['adapters']:
        name = adapter['name']
        if name not in paths:
            matches = [path for alias, path in paths.items() if alias.casefold() == name.casefold()]
            if matches:
                paths[name] = matches[0]
        if name not in paths or not Path(paths[name]).is_file():
            raise ValueError(f'LoRA file is not available: {name}')
        result.append(LoraRef(name=name, weight=adapter['te'], unet=adapter['unet'], dyn=adapter['dyn'],
                              lbw=adapter.get('lbw'), start=adapter.get('start'), stop=adapter.get('stop')))
    return result


def generation_composable_loras(prompts, manual, paths):
    refs = {(ref.name, ref.dyn): ref for ref in manual}
    for prompt in prompts:
        _, networks = batch_extra_networks([prompt])
        unsupported = sorted(set(networks) - {'lora', 'lyco'})
        if unsupported:
            raise ValueError('unsupported extra network types: ' + ', '.join(unsupported))
        adapters = [dict(kind=kind, **parse_lora_activation(args)) for kind, entries in networks.items()
                    for args in entries]
        for ref in generation_batch_loras({'adapters': adapters}, [], paths):
            refs.setdefault((ref.name, ref.dyn), ref)
    return list(refs.values())


def lookup_civitai(path, consent, fetch, check_interrupt=lambda: None):
    if consent is not True:
        raise ValueError('online lookup requires explicit consent')
    path = Path(path)
    with path.open('rb') as stream:
        initial = _file_signature(os.fstat(stream.fileno()))
        digest = hashlib.sha256()
        while True:
            check_interrupt()
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
        if _file_signature(os.fstat(stream.fileno())) != initial or _file_signature(path.stat()) != initial:
            raise ValueError('model changed while hashing')
    sha = digest.hexdigest()
    check_interrupt()
    response = fetch(sha)
    check_interrupt()
    if _file_signature(path.stat()) != initial:
        raise ValueError('model changed during lookup')
    return dict(keywords=civitai_keywords(response, sha), source='Civitai live', sha256=sha)


def merge_schedules(schedules, combine):
    """Merge inclusive end-step timelines without changing any image's schedule."""
    if not schedules:
        raise ValueError('at least one schedule is required')
    for schedule in schedules:
        ends = [end for end, _ in schedule]
        if not ends or any(not isinstance(end, int) or end <= 0 for end in ends) or any(
                a >= b for a, b in zip(ends, ends[1:])):
            raise ValueError('schedule endpoints must be positive and strictly increasing')
    ends = sorted({end for schedule in schedules for end, _ in schedule})
    return [(end, combine([next((value for stop, value in schedule if end <= stop), schedule[-1][1])
                           for schedule in schedules])) for end in ends]
