"""C10c audience-specific feedback from sealed synthetic reviews."""
import json
from unittest.mock import patch

import pytest

from attention_memory_service.core.store import StateConflictError
from attention_memory_service.feedback_evidence import verify_feedback
from tests.dev_v2_fixtures import Fixture
from tests import test_feedback_projection_c10b as c10b


def require(value, reason):
    if not value:
        raise StateConflictError(reason)


def check(packet, **kwargs):
    return verify_feedback(packet, {}, version={}, cell={}, manifest={},
                           read=lambda ref: dict(c10b.REVIEW),
                           eval_job=lambda: {'state': 'settled', 'receipt': {'eval_review': 'sealed'}},
                           require=require, **kwargs)


def test_audience_is_required():
    with pytest.raises(TypeError, match='audience'):
        check({})


@pytest.mark.parametrize('audience', ['', 'Advisor', 'model', None])
def test_unknown_audience_refused_even_for_exact_packet(audience):
    with pytest.raises(StateConflictError, match='audience'):
        check({}, audience=audience)


@pytest.mark.parametrize('audience', ['dev', 'advisor'])
@pytest.mark.parametrize('include_suggestion', [False, True])
def test_review_matches_only_its_audience(audience, include_suggestion):
    review = {'passed': False, 'reason': c10b.REVIEW['reason']}
    if include_suggestion:
        review['reference_suggestion'] = c10b.REVIEW['reference_suggestion']
    if include_suggestion == (audience == 'advisor'):
        assert check({'eval_review': review}, audience=audience) == 'with_eval_review'
    else:
        with pytest.raises(StateConflictError, match='failed trace mismatch'):
            check({'eval_review': review}, audience=audience)


def audience_writer(write, fault):
    """Alter the projected packet before its digest and planner binding are sealed."""
    def wrapped(path, value):
        if path.name in ('advisor-input.json', 'model-input.json'):
            packet = json.loads(value['messages'][-1]['content'])
            review = packet['public_feedback']['eval_review']
            if path.name == 'advisor-input.json':
                if fault == 'advisor_missing':
                    review.pop('reference_suggestion')
                elif fault == 'advisor_changed':
                    review['reference_suggestion'] = 'Not the sealed suggestion'
                elif fault == 'advisor_extra':
                    review['extra'] = 'Not in the projection'
            elif fault == 'dev_suggestion':
                review['reference_suggestion'] = c10b.REVIEW['reference_suggestion']
            value['messages'][-1]['content'] = json.dumps(packet)
        return write(path, value)
    return wrapped


@pytest.mark.parametrize('planner', [False, True])
@pytest.mark.parametrize('suggestion', ['', c10b.REVIEW['reference_suggestion']])
@pytest.mark.parametrize('fault', [None, 'advisor_missing', 'advisor_changed', 'advisor_extra', 'dev_suggestion'])
def test_sealed_source_uses_packet_audiences(tmp_path, planner, suggestion, fault):
    review = dict(c10b.REVIEW, reference_suggestion=suggestion)
    with patch.object(c10b, 'REVIEW', review), \
         patch.object(Fixture, 'write', staticmethod(audience_writer(Fixture.write, fault))):
        _, service = c10b.enriched_source(tmp_path, planner=planner)
    if fault is not None:
        with pytest.raises(StateConflictError, match='failed trace mismatch'):
            service.import_dev_v2_source(import_id='source', catalog_id='failed', descriptor='failed/source.json')
    else:
        result = service.import_dev_v2_source(import_id='source', catalog_id='failed', descriptor='failed/source.json')
        assert result['feedback_projection'] == 'with_eval_review'
        assert service.dev_v2.source('source') == result
        assert service.create_dev_v2_candidate(memory_id='m1', source_import_id='source', created_at=1)
