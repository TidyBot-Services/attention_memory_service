"""Canonical public-feedback failure signature; shared with the worker."""
import re
_IDENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]*\Z')
_OPERATION = re.compile(r'[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\Z')

def derive_signature(feedback):
    """``{'public_error_class', 'operation'}`` in the Service's trusted-path format.

    The class is the longest prefix of ``': '``-separated identifier tokens of
    ``public_error``; the operation is the last failed trace event.
    """
    error = feedback.get('public_error') or ''
    if not isinstance(error, str) or not error.strip():
        raise ValueError('no_public_error')
    tokens = []
    for part in error.split(': '):
        if not _IDENT.fullmatch(part):
            break
        tokens.append(part)
    failed = [e for e in feedback.get('action_trace', []) if isinstance(e, dict) and e.get('status') == 'failed']
    if not failed:
        raise ValueError('no_failed_event')
    operation = '.'.join(x for x in (failed[-1].get('source'), failed[-1].get('operation')) if x)
    error_class = ': '.join(tokens)
    if not tokens or len(error_class) > 120 or not _OPERATION.fullmatch(operation):
        raise ValueError('signature_not_derivable')
    return {'public_error_class': error_class, 'operation': operation}

