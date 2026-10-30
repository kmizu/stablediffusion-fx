"""Diagnostic only: time packaged metadata read/copy/fsync/publication, without torch."""
import argparse
import faulthandler
import time
from pathlib import Path

from sdfx import metadata_utils as metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output_stem')
    args = parser.parse_args()
    started = time.monotonic()

    def log(message):
        print(f'{time.monotonic() - started:.3f}s {message}', flush=True)

    original_fsync = metadata.os.fsync

    def fsync(fd):
        log('fsync begin')
        result = original_fsync(fd)
        log('fsync end')
        return result

    def progress(done, total):
        if done == total or done % (256 * metadata.CHUNK) == 0:
            log(f'progress {done}/{total}')

    def publish(operation):
        log('publish begin')
        result = operation()
        log('publish end')
        return result

    metadata.os.fsync = fsync
    faulthandler.dump_traceback_later(30, repeat=True)
    try:
        log(f'module {metadata.__file__}')
        log('read begin')
        loaded = metadata.read_metadata(args.source, progress=progress)
        log('read end; write begin')
        result = metadata.write_metadata(args.source, loaded['editable'], args.output_stem,
            source_token=loaded['source_token'], progress=progress, publish=publish)
        log(f'write end {result}')
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
