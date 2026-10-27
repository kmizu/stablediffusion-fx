"""Internal per-image Img2img orchestration; not a public worker capability.

Callers must supply reviewed backend, input verifier, and operation-owned
transaction. A verifier must validate exact identity/contents on each call;
the transaction owns its staging directory, validates generated paths and
atomically publishes or rolls back with a cancellation-aware commit. These
requirements are not established by this orchestration seam alone.
"""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import stat
from dataclasses import dataclass

from .batch_interrogate import BatchInterrogateOptions, compose
from .protocol import EngineError, INTERRUPTED, OOM
from .batch_face_manifest import FileSnapshot, MAX_FILE_BYTES
from .temporal_lab_io import OutputTransaction, _absolute, _reject_links

MAX_ITEMS = 1000
MAX_MODELS = 32
MAX_OUTPUT_FILES = 10_000
MAX_OUTPUT_BYTES = 8 * 1024 * 1024 * 1024
MAX_TOTAL_INPUT_BYTES = 2 * 1024 * 1024 * 1024


class TransactionFailure(Exception):
    """An ownership or cleanup failure must abort the entire operation."""


class TransactionEngineFailure(EngineError):
    """Fatal transaction failure retaining the original engine error code."""


@dataclass(frozen=True)
class BatchInput:
    path: str
    identity: object


def run_batch(engine, inputs, generation, models, options, backend, verify_input, transaction_factory, ctx):
    """Run ordered independent inputs using an explicitly supplied owned transaction.

    No capability is advertised; public callers must not inject these objects.
    Model loading, source identity enforcement, output safety and cancellation
    at publication remain the responsibility of the reviewed integration.
    """
    if not isinstance(inputs, (tuple, list)) or not 1 <= len(inputs) <= MAX_ITEMS:
        raise ValueError("batch inputs must be a bounded nonempty sequence")
    if any(not isinstance(item, BatchInput) or not isinstance(item.path, str) or not item.path
           or item.identity is None for item in inputs):
        raise ValueError("invalid batch input identity")
    if not isinstance(models, (tuple, list)) or not 1 <= len(models) <= MAX_MODELS or any(
            not isinstance(model, str) or not model or len(model) > 256 for model in models):
        raise ValueError("interrogator models must be a bounded nonempty sequence")
    if not isinstance(options, BatchInterrogateOptions) or not isinstance(generation, dict):
        raise ValueError("invalid batch options or generation request")
    if not callable(getattr(engine, "img2img", None)) or not callable(getattr(backend, "interrogate", None)) \
            or not callable(verify_input) or not callable(transaction_factory) or not callable(getattr(ctx, "check_interrupt", None)):
        raise ValueError("reviewed backend, verifier and owned transaction are required")
    frozen = deepcopy(generation)
    original = frozen.get("params")
    if not isinstance(original, dict):
        raise ValueError("generation params must be an object")
    if frozen.get("vae_recovery_dir"):
        raise ValueError("external recovery output is unsupported")
    if original.get("hooks"):
        raise ValueError("generation hooks are unsupported until output effects are reviewed")
    positive = original.get("prompt", "")
    negative = original.get("negative_prompt", "")
    # Validate options and original text before any backend calls.
    compose(positive, negative, (), options)
    selected = tuple(models)
    items = tuple(inputs)
    results = []
    for index, item in enumerate(items):
        ctx.check_interrupt()
        try:
            verify_input(item)
            ctx.check_interrupt()
            with transaction_factory(index) as transaction:
                bound = transaction.bind_input(item, ctx)
                raw = backend.interrogate(str(bound), selected, ctx)
                ctx.check_interrupt()
                prompts = compose(positive, negative, raw, options)
                verify_input(item)
                ctx.check_interrupt()
                request = deepcopy(frozen)
                parameters = request["params"]
                current = parameters.get("img2img")
                if current is None:
                    current = {}
                if not isinstance(current, dict):
                    raise ValueError("img2img parameters must be an object")
                current["init_image"] = str(bound)
                parameters["img2img"] = current
                parameters["prompt"] = prompts.positive
                parameters["negative_prompt"] = prompts.negative
                request["output_dir"] = str(transaction.directory)
                generated = engine.img2img(request, ctx)
                ctx.check_interrupt()
                published = transaction.commit(generated, ctx)
            results.append({"status": "complete", "input": item.path,
                            "positive": prompts.positive, "negative": prompts.negative,
                            "result": published})
        except TransactionEngineFailure:
            raise
        except EngineError as exc:
            if exc.code in (INTERRUPTED, OOM) or getattr(exc, "_sdfx_cleanup_failed", False):
                raise
            results.append({"status": "error", "input": item.path, "error": exc.message})
        except MemoryError:
            raise
        except TransactionFailure:
            raise
        except Exception as exc:
            if getattr(exc, "_sdfx_cleanup_failed", False) or type(exc).__name__ in ("OutOfMemoryError", "CudaOutOfMemoryError"):
                raise
            results.append({"status": "error", "input": item.path, "error": str(exc)})
    return results


class OwnedBatchTransaction:
    """Stage and atomically publish one batch in a private operation directory.

    As with OutputTransaction, concurrent hostile writers to the output parent
    are outside the path-based ownership guarantee.
    """
    def __init__(self, parent, name, items, ctx):
        self.items = tuple(items)
        if not 1 <= len(self.items) <= MAX_ITEMS or any(
                not isinstance(item, BatchInput) or not isinstance(item.identity, FileSnapshot)
                or type(item.identity.size) is not int or not 0 < item.identity.size <= MAX_FILE_BYTES
                for item in self.items):
            raise ValueError("verified bounded input snapshots required")
        if sum(item.identity.size for item in self.items) > MAX_TOTAL_INPUT_BYTES:
            raise ValueError("aggregate input size exceeds limit")
        self.ctx = ctx
        self.transaction = OutputTransaction(parent, name, [item.path for item in items])
        self.bound = {}
        self.directories = {}

    def __enter__(self):
        self.transaction.__enter__()
        try:
            self.inputs_dir = self.transaction.staging / ".inputs"
            self.inputs_dir.mkdir()
            return self
        except BaseException as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise

    def __exit__(self, kind, exc, tb):
        try:
            return self.transaction.__exit__(kind, exc, tb)
        except Exception as cleanup:
            stage = self.transaction.staging
            message = f"cleanup failed; staging retained at {stage}: {cleanup}"
            if isinstance(exc, EngineError):
                raise TransactionEngineFailure(exc.code, f"{exc.code}: {exc.message}; {message}") from exc
            if exc is not None:
                raise TransactionFailure(f"{type(exc).__name__}: {exc}; {message}") from exc
            raise TransactionFailure(message) from cleanup

    def bind_input(self, item, index):
        if not 0 <= index < len(self.items) or self.items[index] != item or index in self.bound:
            raise ValueError("invalid input binding")
        snap = item.identity
        if not isinstance(snap, FileSnapshot) or _absolute(item.path) != _absolute(snap.path):
            raise ValueError("verified input snapshot required")
        if not 0 < snap.size <= MAX_FILE_BYTES:
            raise ValueError("input size exceeds limit")
        path = _absolute(item.path)
        _reject_links(path)
        signature = (snap.device, snap.inode, snap.size, snap.mtime_ns)
        info = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or not info.st_ino or (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != signature:
            raise ValueError("input identity changed")
        target = self.inputs_dir / f"{index:04d}{path.suffix}"
        self.bound[index] = target
        digest = hashlib.sha256()
        total = 0
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(path, flags), "rb") as source, target.open("xb") as destination:
            opened = os.fstat(source.fileno())
            if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != signature:
                raise ValueError("input identity changed")
            while True:
                self.ctx.check_interrupt()
                chunk = source.read(min(1024 * 1024, snap.size - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > snap.size:
                    raise ValueError("input grew during snapshot")
                digest.update(chunk)
                destination.write(chunk)
            after = os.fstat(source.fileno())
            if total != snap.size or digest.hexdigest() != snap.sha256 or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != signature:
                raise ValueError("input changed during snapshot")
        self.ctx.check_interrupt()
        return target

    def item_directory(self, index):
        if not 0 <= index < len(self.items):
            raise ValueError("invalid item index")
        if index not in self.directories:
            directory = self.transaction.staging / f"item-{index:04d}"
            directory.mkdir()
            self.directories[index] = directory
        return self.directories[index]

    def discard_item(self, index):
        try:
            directory = self.directories.pop(index, None)
            if directory is not None:
                _reject_links(directory)
                for root, dirs, files in os.walk(directory, followlinks=False):
                    for name in (*dirs, *files):
                        _reject_links(Path(root) / name)
                shutil.rmtree(directory)
        except Exception as exc:
            raise TransactionFailure(f"could not clean item {index}: {exc}") from exc

    def check_output_budget(self):
        count = size = 0
        for directory in self.directories.values():
            _reject_links(directory)
            for path in directory.iterdir():
                _reject_links(path)
                info = path.stat(follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise TransactionFailure("unexpected output entry")
                count += 1
                size += info.st_size
                if count > MAX_OUTPUT_FILES or size > MAX_OUTPUT_BYTES:
                    raise TransactionFailure("generated outputs exceed limit")

    def publish(self, results):
        result = deepcopy(results)
        expected = set()
        count = size = 0
        for index, row in enumerate(result):
            if row.get("status") != "complete":
                continue
            directory = self.item_directory(index)
            images = row.get("result", {}).get("images")
            if not isinstance(images, list) or not images:
                raise ValueError("generated image list is empty")
            for image in images:
                path = _absolute(image["path"])
                if path.parent != directory or path in expected:
                    raise ValueError("generated image escaped its owned directory")
                _reject_links(path)
                info = path.stat(follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError("generated image is not an owned regular file")
                count += 1; size += info.st_size
                if count > MAX_OUTPUT_FILES or size > MAX_OUTPUT_BYTES:
                    raise ValueError("generated outputs exceed limit")
                expected.add(path)
                image["path"] = str(self.transaction.destination / directory.name / path.name)
        for directory in self.directories.values():
            _reject_links(directory)
            for path in directory.iterdir():
                if path not in expected:
                    raise ValueError(f"unexpected generated file: {path}")
        allowed = {self.inputs_dir, *self.directories.values()}
        if set(self.transaction.staging.iterdir()) != allowed:
            raise ValueError("unexpected entry in batch staging directory")
        for path in self.bound.values():
            if os.path.lexists(path):
                _reject_links(path)
                path.unlink()
        self.inputs_dir.rmdir()
        self.ctx.check_interrupt()
        self.ctx.publish(self.transaction.commit)
        return result


class _OwnedItemTransaction:
    def __init__(self, owner, index):
        self.owner, self.index = owner, index
    def __enter__(self):
        self.directory = self.owner.item_directory(self.index)
        return self
    def bind_input(self, item, ctx):
        return self.owner.bind_input(item, self.index)
    def commit(self, result, ctx):
        ctx.check_interrupt()
        self.owner.check_output_budget()
        return result
    def __exit__(self, kind, exc, tb):
        if exc is not None:
            try:
                self.owner.discard_item(self.index)
            except Exception as cleanup:
                message = f"item cleanup failed at {self.directory}: {cleanup}"
                if isinstance(exc, EngineError):
                    raise TransactionEngineFailure(exc.code, f"{exc.code}: {exc.message}; {message}") from exc
                raise TransactionFailure(f"{exc}; {message}") from exc
        return False


def run_owned_batch(engine, inputs, generation, models, options, backend, output_parent, output_name, ctx, *, _before_publish=None, _provenance=None, _terminal_check=None):
    """Internal entry point binding verified inputs and grouped publication."""
    if not isinstance(inputs, (tuple, list)) or not isinstance(generation, dict) or not isinstance(generation.get("params"), dict):
        raise ValueError("invalid batch request")
    params = generation["params"]
    batch = params.get("batch_size") or 1
    iterations = params.get("n_iter") or 1
    width = params.get("width") or 512
    height = params.get("height") or 512
    if any(type(value) is not int or value <= 0 for value in (batch, iterations, width, height)):
        raise ValueError("invalid output dimensions or count")
    if width > 4096 or height > 4096 or batch > 16 or iterations > 100:
        raise ValueError("output dimensions or count exceed limit")
    hires = params.get("hires") or {}
    if not isinstance(hires, dict) or hires.get("enabled"):
        raise ValueError("hires output is unsupported until output bounds are reviewed")
    count = len(inputs) * batch * iterations
    # Normal engine PNGs are produced from bounded RGB/RGBA pixels; reserve
    # space for PNG framing and metadata in addition to uncompressed pixels.
    if count > MAX_OUTPUT_FILES or count * (width * height * 4 + 1024 * 1024) > MAX_OUTPUT_BYTES:
        raise ValueError("estimated outputs exceed limit")
    with OwnedBatchTransaction(output_parent, output_name, inputs, ctx) as owner:
        rows = run_batch(engine, inputs, generation, models, options, backend,
                         lambda item: None, lambda index: _OwnedItemTransaction(owner, index), ctx)
        if _before_publish is not None:
            _before_publish()
        if _provenance is not None:
            for row in rows:
                if row["status"] == "complete":
                    row["interrogators"] = deepcopy(_provenance)
        if any(row["status"] == "complete" for row in rows):
            if _terminal_check is not None:
                _terminal_check()
            return owner.publish(rows)
    # Empty batches still claim the same cancellation/terminal ownership. The
    # owned staging has already been cleaned before claiming that boundary.
    if _terminal_check is not None:
        _terminal_check()
    return ctx.publish(lambda: rows)


def run_registered_batch(engine, inputs, generation, models, options, registry, output_parent, output_name, ctx, *, _before_publish=None):
    """Internal registry integration; release models before publishing outputs."""
    from .batch_interrogate_backends import BackendRegistry
    if not isinstance(registry, BackendRegistry):
        raise ValueError("internal backend registry required")
    if not isinstance(models, (tuple, list)):
        raise ValueError("backend selection must be a sequence")
    models = tuple(models)
    scope = registry.owned(models)
    scope.__enter__()
    closed = False

    def close():
        nonlocal closed
        closed = True
        scope.__exit__(None, None, None)

    try:
        provenance = registry.provenance(models)
        def finish():
            if _before_publish is not None:
                _before_publish()
            if registry.provenance(models) != provenance:
                raise ValueError("backend provenance changed during batch")
            close()
            if _before_publish is not None:
                _before_publish()
        return run_owned_batch(engine, inputs, generation, models, options, registry,
                               output_parent, output_name, ctx, _before_publish=finish, _provenance=provenance,
                               _terminal_check=_before_publish)
    except BaseException as exc:
        if not closed:
            closed = True
            scope.__exit__(type(exc), exc, exc.__traceback__)
        raise
    finally:
        if not closed:
            close()

