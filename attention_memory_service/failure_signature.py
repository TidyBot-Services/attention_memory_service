"""Canonical public-feedback failure signature; shared with the worker."""
import re
_IDENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]*\Z')
_OPERATION = re.compile(r'[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\Z')
REASON_CODE = re.compile(r'[a-z][a-z0-9_]{0,119}\Z')
_PLANNER_REASON = re.compile(r'cuRoboV2 planning rejected: ([a-z][a-z0-9_]{0,119})(?=[; ]|\Z)')


def planner_reason_code(error):
    """Read only the bounded code in the planner's public rejection format."""
    match = _PLANNER_REASON.search(error) if isinstance(error, str) else None
    return match.group(1) if match else None


def public_error_class(error):
    tokens = []
    for part in error.split(': '):
        if not _IDENT.fullmatch(part):
            break
        tokens.append(part)
    return ': '.join(tokens)


def matches_frozen_derivation(feedback, signature):
    """Recheck creation bindings using the frozen signature's original shape."""
    derived = derive_signature(feedback)
    if 'reason_code' not in signature:
        derived.pop('reason_code', None)
    return derived == signature

def derive_signature(feedback):
    """Public class/operation, plus a planner reason code when available.

    The class is the longest prefix of ``': '``-separated identifier tokens of
    ``public_error``; the operation is the last failed trace event.
    """
    error = feedback.get('public_error') or ''
    if not isinstance(error, str) or not error.strip():
        raise ValueError('no_public_error')
    failed = [e for e in feedback.get('action_trace', []) if isinstance(e, dict) and e.get('status') == 'failed']
    if not failed:
        raise ValueError('no_failed_event')
    operation = '.'.join(x for x in (failed[-1].get('source'), failed[-1].get('operation')) if x)
    error_class = public_error_class(error)
    if not error_class or len(error_class) > 120 or not _OPERATION.fullmatch(operation):
        raise ValueError('signature_not_derivable')
    signature = {'public_error_class': error_class, 'operation': operation}
    reason = planner_reason_code(error)
    if reason is not None:
        signature['reason_code'] = reason
    return signature
