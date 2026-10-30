"""Acceptance-only fault after genuine PhotoMaker has injected its adapter."""
import contextlib
import os
import sys

if os.environ.get('SDFX_TEST_PHOTOMAKER_LOAD_FAILURE') == '1':
    with contextlib.redirect_stdout(sys.stderr):
        from photomaker import PhotoMakerStableDiffusionXLPipeline
    original = PhotoMakerStableDiffusionXLPipeline.load_photomaker_adapter
    injected = False

    def fail_once_after_real_load(self, *args, **kwargs):
        global injected
        result = original(self, *args, **kwargs)
        if not injected:
            injected = True
            raise RuntimeError('acceptance fault after genuine PhotoMaker adapter injection')
        return result

    PhotoMakerStableDiffusionXLPipeline.load_photomaker_adapter = fail_once_after_real_load
