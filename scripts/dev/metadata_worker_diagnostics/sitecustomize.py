"""Opt-in test startup instrumentation; never included in packaged Python modules."""
import faulthandler
import functools
import os
import sys
import time

from sdfx import metadata_utils

diagnostic_file = open(os.environ['SDFX_METADATA_DIAGNOSTIC_FILE'], 'a', encoding='utf-8', buffering=1)


def log(message):
    print(f'{time.monotonic():.3f} metadata diagnostic: {message}', file=diagnostic_file, flush=True)


def trace(name, operation):
    @functools.wraps(operation)
    def wrapped(*args, **kwargs):
        log(name + ' begin')
        try:
            result = operation(*args, **kwargs)
        except BaseException as error:
            log(f'{name} failed: {type(error).__name__}: {error}')
            raise
        log(name + ' end')
        return result
    return wrapped


metadata_utils.os.fsync = trace('fsync', metadata_utils.os.fsync)
metadata_utils.os.link = trace('publication link', metadata_utils.os.link)
metadata_utils.read_metadata = trace('read', metadata_utils.read_metadata)
metadata_utils.write_metadata = trace('write', metadata_utils.write_metadata)
faulthandler.dump_traceback_later(30, repeat=True, file=diagnostic_file)
log(f'loaded {metadata_utils.__file__}')
