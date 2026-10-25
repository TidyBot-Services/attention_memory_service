"""Layered note applicability (plan 0d, fourth revision 2026-10-23).

``same_scene`` is the exact tuple and the only level at which a trusted note is
trusted. ``same_task`` lets other scenes of the same task read a note as a
labelled reference that is always excluded from results. ``same_family`` is
an explicit opt-in read of another task in the configured family, never trusted.

Execution profiles must match at every level. Omission on both sides is the
legacy profile; no profile may read or validate a note from another profile.

SDK matching uses the interface identity (contract plus Dev SDK prompt).
Compatibility rule for notes stored before the field existed: a note whose
applicability has no ``sdk_interface_sha256`` keeps the old exact match on the
implementation hash ``sdk_sha256``. New notes are matched on the interface
identity only; the implementation hash stays stored for audit.
"""
from __future__ import annotations

import hashlib
import json
import re

SAME_SCENE = 'same_scene'
SAME_TASK = 'same_task'
SAME_FAMILY = 'same_family'
LEVELS = (SAME_SCENE, SAME_TASK, SAME_FAMILY)
LAYER = {SAME_SCENE: 'scene', SAME_TASK: 'task', SAME_FAMILY: 'family'}

PROFILE_KEY = 'execution_profile'
_PROFILE = re.compile(r'[a-z0-9][a-z0-9-]{0,47}\Z')
INTERFACE_KEY = 'sdk_interface_sha256'
SDK_KEYS = frozenset(('sdk_sha256', INTERFACE_KEY))
# Seed, scene contract, scene/object/camera/variant axes may differ.
TASK_KEYS = ('benchmark', 'suite', 'task_id', 'robot', 'controller', 'perception_mode',
             'namespace', 'task_prompt')
_SHA = re.compile(r'[a-f0-9]{64}\Z')


def layer_of(level):
    return LAYER.get(level, 'scene')

def same_family(applicability, context):
    from .families import family_of
    keys = ('benchmark', 'suite', 'robot', 'controller', 'perception_mode', 'namespace')
    family = family_of(applicability.get('task_id'))
    return (profile_matches(applicability, context)
            and sdk_matches(applicability, context)
            and all(isinstance(context.get(k), str) and context[k]
                    and applicability.get(k) == context[k] for k in keys)
            and applicability.get('task_id') != context.get('task_id')
            and family is not None and family == family_of(context.get('task_id')))


def interface_digest(sdk_identity):
    """SHA-256 of ``{contract, prompt}`` encoded as the main repo encodes it."""
    value = {'contract': sdk_identity['contract'], 'prompt': sdk_identity['prompt']}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def valid_interface(context):
    value = context.get(INTERFACE_KEY)
    return value is None or isinstance(value, str) and _SHA.fullmatch(value) is not None


def validate_profile(value):
    """Only omission represents legacy; explicit null/legacy are invalid."""
    # Existing validators own container-shape errors; keep legacy errors intact.
    if isinstance(value, dict) and PROFILE_KEY in value:
        profile = value[PROFILE_KEY]
        if not isinstance(profile, str) or profile == 'legacy' or not _PROFILE.fullmatch(profile):
            raise ValueError('execution_profile must match ^[a-z0-9][a-z0-9-]{0,47}$ and cannot be legacy')


def profile_matches(applicability, context):
    """A note can cross no execution-profile boundary, including legacy."""
    return applicability.get(PROFILE_KEY) == context.get(PROFILE_KEY)


def require_profile_match(applicability, context):
    from .core.store import StateConflictError
    validate_profile(applicability)
    validate_profile(context)
    if not profile_matches(applicability, context):
        raise StateConflictError(
            f"execution profile mismatch: note {applicability.get(PROFILE_KEY, 'legacy')}, "
            f"context {context.get(PROFILE_KEY, 'legacy')}")


def config_profile(config):
    settings = config.get('execution_settings', {})
    if not isinstance(settings, dict):
        raise ValueError('execution_settings must be an object')
    value = {PROFILE_KEY: settings['profile']} if 'profile' in settings else {}
    validate_profile(value)
    return value


def profile_fields(applicability):
    """Omission labels legacy without changing existing response bytes."""
    return {PROFILE_KEY: applicability[PROFILE_KEY]} if PROFILE_KEY in applicability else {}


def sdk_matches(applicability, context):
    stored = applicability.get(INTERFACE_KEY)
    if stored is not None:
        return context.get(INTERFACE_KEY) == stored
    return context.get('sdk_sha256') == applicability.get('sdk_sha256')


def same_scene(applicability, context, keys):
    """Exact tuple on ``keys`` with the SDK compared by the rule above."""
    return profile_matches(applicability, context) and sdk_matches(applicability, context) and all(
        applicability.get(k) == context.get(k) for k in keys if k not in SDK_KEYS)


def same_task(applicability, context, task_prompts):
    return (profile_matches(applicability, context)
            and sdk_matches(applicability, context)
            and all(isinstance(context.get(k), str) and context[k] for k in TASK_KEYS)
            and all(applicability.get(k) == context.get(k) for k in TASK_KEYS if k != 'task_prompt')
            and context.get('task_prompt') in task_prompts)


def marker(level, trusted_here):
    value = {'match_level': level, 'layer': layer_of(level), 'trusted_here': trusted_here}
    if not trusted_here:
        value['excluded_from_results'] = True
    return value
