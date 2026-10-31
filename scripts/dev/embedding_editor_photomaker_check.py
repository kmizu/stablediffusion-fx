"""Real #45 editor / PhotoMaker / collision / cancellation integration fixture.

Run only with a coordinated GPU slot. This is a synthetic vector fixture; preview
requests bypass the diagnostic journal so its draft is never written there.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

from composable_lora_check import Client
from prompt_fusion_check import pixel_record, image_difference


def main():
    p = argparse.ArgumentParser()
    for name in ('python', 'model', 'adapter', 'reference', 'output'):
        p.add_argument(name)
    p.add_argument('--python-path', action='append', required=True)
    p.add_argument('--engine-root')
    p.add_argument('--neutral-runtime', action='store_true', help='Exercise #85 runtime hook rebinding during edited PhotoMaker encoding')
    p.add_argument('--merge-composed', action='store_true', help='Resolve a #47 expression from the edited embedding')
    p.add_argument('--vram-mode', choices=['gpu', 'model_offload', 'sequential_offload'], default='model_offload')
    a = p.parse_args()
    root = Path(a.engine_root).resolve() if a.engine_root else Path(__file__).resolve().parents[2]
    output = Path(a.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    client = Client(a.python, root, output, False, python_paths=a.python_path)
    # Editor open replies contain vectors; keep all responses out of the journal.
    def receive_private(deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('engine response deadline exceeded')
        message = client.messages.get(timeout=remaining)
        if message['type'] in ('eof', 'reader-error'):
            raise RuntimeError(message['type'])
        return message
    client.receive = receive_private
    report = {'status': 'running', 'mode': a.vram_mode}
    def call(kind, params):
        result = client.call(kind, params)
        assert result['type'] == 'result', result
        return result['result']
    def private_send(rid, kind, params):
        client.process.stdin.write(json.dumps({'id':rid, 'type':kind, 'params':params}) + '\n')
        client.process.stdin.flush()
    def private_result(rid):
        deadline = time.monotonic() + 300
        while True:
            message = client.receive(deadline)
            if message.get('id') == rid and message['type'] in ('result', 'error'):
                return message
    try:
        assert client.receive(time.monotonic() + 180)['type'] == 'hello'
        loaded = call('load_model', {'path':str(Path(a.model).resolve()), 'dtype':'fp16', 'vram_mode':a.vram_mode})
        assert loaded['kind'] == 'sdxl', loaded
        saved = call('embedding_merge', {'operation':'save', 'text':"'red hair' * 0.7 + 'blue hair' * 0.3", 'name':'editorfixture45', 'output_dir':str(output/'embeddings')})
        assert saved['active'], saved
        source = Path(saved['paths'][0]); original = source.read_bytes(); original_hash = hashlib.sha256(original).hexdigest()
        draft = call('embedding_editor', {'operation':'open', 'name':'editorfixture45', 'path':str(source)})
        collision = output/'collision'/source.name; collision.parent.mkdir(); shutil.copyfile(source, collision)
        rejected = client.call('embedding_editor', {'operation':'open', 'name':'editorfixture45', 'path':str(collision)})
        assert rejected['type'] == 'error', rejected
        assert 'already loaded' in rejected.get('message', '').lower(), rejected
        assert source.read_bytes() == original and collision.read_bytes() == original
        # Independent base and suffix owners must remain intact on save collision.
        owner_base = output/'embeddings'/'ownerprobe.safetensors'
        owner_suffix = output/'embeddings'/'ownerprobe_1.safetensors'
        # The edited expression has two rows and already owns a _1 token.
        # Use a single-token embedding for two genuinely independent owners.
        owner_saved = call('embedding_merge', {'operation':'save', 'text':"'cat'", 'name':'ownerprobe', 'output_dir':str(output/'embeddings')})
        assert owner_saved['active'], owner_saved
        shutil.copyfile(owner_base, owner_suffix)
        owner_draft = call('embedding_editor', {'operation':'open', 'name':'ownerprobe', 'path':str(owner_base)})
        suffix_draft = call('embedding_editor', {'operation':'open', 'name':'ownerprobe_1', 'path':str(owner_suffix)})
        owner_original, suffix_original = owner_base.read_bytes(), owner_suffix.read_bytes()
        rejected_draft = copy.deepcopy(owner_draft)
        rejected_draft['vectors']['clip_l'][0][0] += 0.5
        private_send('owner-collision-save', 'embedding_editor', {'operation':'save', 'name':'ownerprobe', 'draft':rejected_draft, 'overwrite':True})
        failure = private_result('owner-collision-save')
        assert failure['type'] == 'error' and 'collision' in failure.get('message', '').lower(), failure
        assert owner_base.read_bytes() == owner_original and owner_suffix.read_bytes() == suffix_original
        assert not list(owner_base.parent.glob(owner_base.name + '.before-edit-*.backup'))
        assert call('embedding_editor', {'operation':'open','name':'ownerprobe'})['vectors'] == owner_draft['vectors']
        assert call('embedding_editor', {'operation':'open','name':'ownerprobe_1'})['vectors'] == suffix_draft['vectors']
        for rows in draft['vectors'].values():
            for row in rows:
                for i in range(min(32, len(row))): row[i] = 3.0 if i % 2 else -3.0
        pm = {'model':str(Path(a.adapter).resolve()), 'images':[str(Path(a.reference).resolve())], 'trigger_word':'img', 'start_merge_step':2}
        token = "<'editorfixture45' * 1>" if a.merge_composed else 'editorfixture45'
        base = {'prompt':f'a woman img with {token} in a garden, detailed portrait', 'negative_prompt':'blurry', 'seed':173, 'steps':8, 'width':768, 'height':768, 'cfg_scale':5, 'sampler_name':'DPM++ 2M', 'scheduler':'Automatic', 'hooks':{'photomaker':pm}}
        if a.neutral_runtime:
            base['prompt'] += f' AND_PERP a woman img with {token} and red flowers'
            base['hooks']['neutral_prompt'] = {'cfg_rescale': 0}
        ordinary = dict(base, prompt='a woman with editorfixture45 in a garden, detailed portrait', hooks={})
        def request(name, params, edited=False):
            q={'params':params, 'embedding_paths':[str(source)], 'output_dir':str(output/name), 'preview_every':0}
            if edited: q['embedding_editor_preview']=draft
            return q
        def generate(name, edited=False, params=None):
            rid='photo-'+name
            private_send(rid, 'txt2img', request(name, base if params is None else params, edited))
            result=private_result(rid); assert result['type']=='result', result
            return pixel_record(result['result']['images'][0]['path'])
        ordinary_before=generate('ordinary-before', params=ordinary)
        before=generate('before'); changed=generate('preview', True); after=generate('after')
        change=image_difference(before['path'], changed['path']); recovery=image_difference(before['path'], after['path'])
        assert not change['exact'], change
        assert recovery['exact'], recovery
        private_send('editor-cancel', 'txt2img', request('cancelled', dict(base, steps=40), True))
        deadline=time.monotonic()+300; sent=False; final=None; decision=None
        while final is None or decision is None:
            msg=client.receive(deadline)
            if msg.get('id')=='editor-cancel' and msg.get('event')=='progress' and not sent:
                private_send('editor-interrupt','interrupt',{'target_id':'editor-cancel'}); sent=True
            if msg.get('id')=='editor-cancel' and msg['type'] in ('result','error'): final=msg
            if msg.get('id')=='editor-interrupt' and msg['type'] in ('result','error'): decision=msg
        assert sent and final['type']=='error' and final['code']=='interrupted', final
        assert decision['type']=='result' and decision['result']['accepted'], decision
        recovered=generate('after-cancel'); cancel_diff=image_difference(before['path'], recovered['path'])
        assert cancel_diff['exact'], cancel_diff
        ordinary_after=generate('ordinary-after', params=ordinary)
        ordinary_diff=image_difference(ordinary_before['path'], ordinary_after['path'])
        assert ordinary_diff['exact'], ordinary_diff
        assert hashlib.sha256(source.read_bytes()).hexdigest()==original_hash
        report.update(status='passed-pending-visual-inspection', merge_composed=a.merge_composed, neutral_runtime=a.neutral_runtime, cases={'ordinary_before':ordinary_before,'before':before,'preview':changed,'after':after,'after_cancel':recovered,'ordinary_after':ordinary_after}, comparisons={'preview':change,'recovery':recovery,'cancel_recovery':cancel_diff,'ordinary_recovery':ordinary_diff}, collision_rejected=True, independent_owner_collision_rejected_before_write=True, cancellation_accepted=True, source_sha256=original_hash)
    except BaseException as e:
        report.update(status='failed', failure=repr(e)); raise
    finally:
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        client.close()

if __name__ == '__main__': main()
