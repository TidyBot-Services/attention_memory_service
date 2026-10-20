"""Validation for the short, public display title of a Dev v2 note."""

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


def validate_title(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError('title must be a string')
    if '\n' in value or '\r' in value:
        raise ValueError('title must not contain a newline')
    if any(unicodedata.category(char).startswith('C') for char in value):
        raise ValueError('title must not contain control characters')
    title = value.strip()
    if not 1 <= len(title) <= 80:
        raise ValueError('title must contain 1–80 characters after trimming')
    if '<' in title or '>' in title:
        raise ValueError('title must be plain text')
    if _URL.search(title):
        raise ValueError('title must not contain a URL')
    if '/' in title or '\\' in title:
        raise ValueError('title must not contain a path')
    if _SECRET.search(title):
        raise ValueError('title must not contain key-like text')
    if _ORACLE.search(title):
        raise ValueError('title must not contain native evaluator or oracle terms')
    return title
