"""Layered note applicability (plan 0d, second revision 2026-10-22).

``same_scene`` is the exact tuple and the only level at which a trusted note is
trusted. ``same_task`` lets other scenes of the same task read a note as a
labelled reference that is always excluded from results.

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
LEVELS = (SAME_SCENE, SAME_TASK)
INTERFACE_KEY = 'sdk_interface_sha256'
SDK_KEYS = frozenset(('sdk_sha256', INTERFACE_KEY))
# Seed, scene contract, scene/object/camera/variant axes may differ.
TASK_KEYS = ('benchmark', 'suite', 'task_id', 'robot', 'controller', 'perception_mode',
             'namespace', 'task_prompt')
_SHA = re.compile(r'[a-f0-9]{64}\Z')


def interface_digest(sdk_identity):
    """SHA-256 of ``{contract, prompt}`` encoded as the main repo encodes it."""
    value = {'contract': sdk_identity['contract'], 'prompt': sdk_identity['prompt']}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def valid_interface(context):
    value = context.get(INTERFACE_KEY)
    return value is None or isinstance(value, str) and _SHA.fullmatch(value) is not None


def sdk_matches(applicability, context):
    stored = applicability.get(INTERFACE_KEY)
    if stored is not None:
        return context.get(INTERFACE_KEY) == stored
    return context.get('sdk_sha256') == applicability.get('sdk_sha256')


def same_scene(applicability, context, keys):
    """Exact tuple on ``keys`` with the SDK compared by the rule above."""
    return sdk_matches(applicability, context) and all(
        applicability.get(k) == context.get(k) for k in keys if k not in SDK_KEYS)


def same_task(applicability, context, task_prompts):
    return (sdk_matches(applicability, context)
            and all(isinstance(context.get(k), str) and context[k] for k in TASK_KEYS)
            and all(applicability.get(k) == context.get(k) for k in TASK_KEYS if k != 'task_prompt')
            and context.get('task_prompt') in task_prompts)


def marker(level, trusted_here):
    value = {'match_level': level, 'trusted_here': trusted_here}
    if not trusted_here:
        value['excluded_from_results'] = True
    return value
