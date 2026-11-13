"""Small synthetic visual Advisor receipts; no provider, simulator or real database."""
import json
import sqlite3
import base64
import io

import pytest
from PIL import Image

from attention_memory_service.core.store import StateConflictError
from attention_memory_service.dev_v2 import digest, encoded, view_words
from attention_memory_service.memory_service import MemoryService
from attention_memory_service.memory_v2 import MemoryV2Manager
from tests.dev_v2_fixtures import Fixture


def visual_source(tmp_path, defect=None):
    f = Fixture(tmp_path)
    failed = f.make('failed', visual=True)
    f.add_advice(failed)
    directory = f.root / 'failed'
    answer = 'Move directly toward the target.'
    frames = []
    for view in ('overview', 'side', 'top', 'left_wrist', 'right_wrist'):
        name = 'recording_frames/frame.jpg' if view == 'overview' else f'recording_views/{view}/frame.jpg'
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        source = io.BytesIO()
        Image.new('RGB', (2, 2), (20, 40, 60)).save(source, format='JPEG', quality=90)
        path.write_bytes(source.getvalue())
        ref = {'uri': str(path), 'sha256': digest(path.read_bytes())}
        projected = io.BytesIO()
        Image.open(io.BytesIO(path.read_bytes())).convert('RGB').save(projected, format='JPEG', quality=80)
        frames.append({'view': view, 'filename': name, 'sha256': ref['sha256'],
                       'data_url': 'data:image/jpeg;base64,' + base64.b64encode(projected.getvalue()).decode()})
    review = {'passed': False, 'reason': 'The gripper missed.', 'reference_suggestion': 'Move closer.'}
    view = {'schema_version': 'attentionbench.request-view.v1',
            'task': {'task_id': 'pick', 'task_prompt': failed['config']['task_prompt']},
            'question': 'What repair should be tried?', 'reasoning': '', 'request_kind': 'generic',
            'frames': frames, 'steps': [{'index': 0, 'label': 'Moved arm', 'status': 'failed',
                                        'count': 1, 'stop_reason': 'Stopped: timeout'}],
            'diff': 'changed move', 'eval_review': review, 'shown_video_seconds': 2.0}
    if defect == 'missing_frames':
        view['frames'] = []
    if defect == 'bad_frame_hash':
        view['frames'][0]['sha256'] = '0' * 64
    if defect == 'bad_image':
        view['frames'][0]['data_url'] = 'data:image/jpeg;base64,AA=='
    if defect == 'oracle':
        view['native_success'] = False
    view_ref = f.write(directory / 'advisor-input.json', view)
    words = sum(view_words(x) for x in [view['question'], view['reasoning'], view['diff'],
                *(s['stop_reason'] for s in view['steps']), str(review['passed']),
                review['reason'], review['reference_suggestion']])
    seconds = 2.0 + words / 4.0 + view_words(answer) / 0.7
    usage = {'total_tokens': 7, 'prompt_tokens': 4, 'completion_tokens': 3}
    hint = {'schema_version': 'attentionbench.advisor-proxy.v3', 'kind': 'hint',
            'request_type': 'hint', 'status': 'completed', 'requested': True,
            'feedback': answer, 'model': 'controlled_test_visual_advisor', 'mode': 'vision',
            'cache_hit': False, 'request_view': view_ref, 'frames': [
                {k: frame[k] for k in ('filename', 'sha256', 'view')} for frame in frames],
            'views': sorted({frame['view'] for frame in frames}), 'question': view['question'],
            'selected_moments': [], 'localization': {'status': 'generic_fallback'},
            'shown_video_seconds': 2.0, 'shown_words': words, 'answer_words': view_words(answer),
            'read_wps': 4.0, 'write_wps': 0.7, 'hint_human_seconds': seconds,
            'logical_human_seconds': seconds, 'usage': usage}
    if defect == 'bad_view_hash':
        hint['request_view'] = {**view_ref, 'sha256': '0' * 64}
    if defect == 'bad_cost':
        hint['answer_words'] += 1
    hint_ref = f.write(directory / 'advisor-hint.json', hint)
    help_ref = f.write(directory / 'help.json', {'provider': 'other' if defect == 'unknown_provider' else 'human_proxy', 'content': answer,
                        'mode': 'vision', 'scope': 'public repair hint', 'advisor_proxy': hint_ref,
                        'selected_moments': [], 'localization': {'status': 'generic_fallback'}})
    model_request = json.loads((directory / 'model-input.json').read_text())
    packet = json.loads(model_request['messages'][-1]['content'])
    packet['attention_control']['help']['provider'] = 'human_proxy'
    packet['attention_control']['help']['content'] = answer
    if defect == 'not_adopted':
        packet['attention_control']['help']['content'] = 'Different guidance.'
    model_request['messages'][-1]['content'] = encoded(packet)
    model_ref = f.write(directory / 'model-input.json', model_request)
    record = json.loads((directory / 'version2.json').read_text())
    record.update(help=help_ref, model_input=model_ref)
    record_ref = f.write(directory / 'version2.json', record)
    ledger = directory / 'ledger.sqlite'
    with sqlite3.connect(ledger) as db:
        db.execute('UPDATE v2_versions SET payload=? WHERE version=2', (encoded(record),))
        for kind in ('advisor', 'model'):
            job = json.loads(db.execute('SELECT payload FROM v2_jobs WHERE kind=?', (kind,)).fetchone()[0])
            if kind == 'advisor':
                job['receipt'] = {'advisor_answer': hint_ref, 'request_view': view_ref,
                                  'cache_hit': False, **{k: hint[k] for k in
                                  ('shown_video_seconds', 'shown_words', 'answer_words', 'read_wps', 'write_wps')}}
                job['actual']['hint_human_seconds'] = seconds
                if defect == 'late':
                    job['settled_at'] = 5.5
            else:
                job['receipt']['request'] = model_ref
            db.execute('UPDATE v2_jobs SET payload=? WHERE kind=?', (encoded(job), kind))
    source = json.loads((directory / 'source.json').read_text())
    source['version_record'] = record_ref
    source.pop('request_id')
    source['ledger']['sha256'] = digest(ledger.read_bytes())
    f.write(directory / 'source.json', source)
    execution = json.loads((directory / 'execution.json').read_text())
    execution['ledger']['sha256'] = source['ledger']['sha256']
    f.write(directory / 'execution.json', execution)
    service = MemoryService(tmp_path / 'authority.sqlite', dev_v2_catalogs=f.seal())
    service.import_dev_v2_execution(import_id='failed', catalog_id='failed', descriptor='failed/execution.json')
    return service


def test_visual_source_requires_sealed_frames_and_fails_closed(tmp_path):
    service = visual_source(tmp_path)
    assert service.dev_v2.catalog('failed')['excluded_frame_images'] == 5
    with pytest.raises(StateConflictError, match='visual Advisor frame not sealed: per-frame images are not registered evidence'):
        service.import_dev_v2_source(import_id='visual', catalog_id='failed', descriptor='failed/source.json')


@pytest.mark.parametrize(('defect', 'reason'), [
    ('late', 'settled after Dev model'),
    ('not_adopted', 'visual Advisor frame not sealed'),
    ('bad_view_hash', 'request_view/ledger reference mismatch|preregistered'),
    ('bad_frame_hash', 'visual Advisor frame not sealed'),
    ('bad_image', 'visual Advisor frame not sealed'),
    ('missing_frames', 'request_view frames required'),
    ('oracle', 'oracle field'),
    ('bad_cost', 'visual Advisor frame not sealed'),
])
def test_visual_source_rejects_invalid_receipts(tmp_path, defect, reason):
    service = visual_source(tmp_path, defect)
    with pytest.raises(StateConflictError, match=reason):
        service.import_dev_v2_source(import_id='visual', catalog_id='failed', descriptor='failed/source.json')


def test_unknown_provider_rejected(tmp_path):
    service = visual_source(tmp_path, 'unknown_provider')
    with pytest.raises(StateConflictError, match='unsupported Advisor provider'):
        service.import_dev_v2_source(import_id='visual', catalog_id='failed', descriptor='failed/source.json')
