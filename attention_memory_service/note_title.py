"""Validation for short, public display text on Dev v2 notes."""

import re
import unicodedata


_ORACLE = re.compile(
    r'oracle|reward|score|native|simulator|evaluator|evaluation|'
    r'_check_success|private|hidden|diagnos|eval|原生评估|评估器|神谕|奖励|得分', re.I)
_URL = re.compile(r'\b(?:[a-z][a-z0-9+.-]*://|www\.|[\w.-]+\.(?:com|org|net|io|ai|dev|edu|gov)\b)', re.I)
_SECRET = re.compile(
    r'\b(?:sk|pk|ghp|gho|github_pat)[-_][A-Za-z0-9_-]{12,}\b|'
    r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|'
    r'\b(?:api[_ -]?key|secret|password|bearer|token)\s*[:=]\s*\S+|'
    r'\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b|'
    r'\b(?:[a-f0-9]{32,}|[A-Za-z0-9+/]{40,}={0,2})\b', re.I)


def _validate_display_text(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f'{field} must be a string')
    if '\n' in value or '\r' in value:
        raise ValueError(f'{field} must not contain a newline')
    if any(unicodedata.category(char).startswith('C') for char in value):
        raise ValueError(f'{field} must not contain control characters')
    normalized = value.strip()
    if not 1 <= len(normalized) <= 80:
        raise ValueError(f'{field} must contain 1–80 characters after trimming')
    if '<' in normalized or '>' in normalized:
        raise ValueError(f'{field} must be plain text')
    if _URL.search(normalized):
        raise ValueError(f'{field} must not contain a URL')
    if '/' in normalized or '\\' in normalized:
        raise ValueError(f'{field} must not contain a path')
    if _SECRET.search(normalized):
        raise ValueError(f'{field} must not contain key-like text')
    if _ORACLE.search(normalized):
        raise ValueError(f'{field} must not contain native evaluator or oracle terms')
    return normalized


def validate_title(value: str) -> str:
    return _validate_display_text(value, 'title')


def validate_summary(value: str) -> str:
    return _validate_display_text(value, 'summary')


def validate_group(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError('group must be a string')
    if '\n' in value or '\r' in value:
        raise ValueError('group must not contain a newline')
    if any(unicodedata.category(char).startswith('C') for char in value):
        raise ValueError('group must not contain control characters')
    group = ' '.join(part for part in value.strip().split(' ') if part)
    if not 1 <= len(group) <= 24:
        raise ValueError('group must contain 1–24 characters after trimming')
    if any(char != ' ' and char != '-' and not (char.isalpha() or char.isdecimal())
           for char in group):
        raise ValueError('group must contain only letters, digits, spaces and hyphens')
    return group
