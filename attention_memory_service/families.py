"""Task families for reference-only Memory reads."""
from fnmatch import fnmatchcase

# The only place task families are defined (plan 0d fourth revision).
FAMILIES = {'pick_and_place': ('pick_place_*',)}


def family_of(task_id) -> str | None:
    if not isinstance(task_id, str):
        return None
    return next((family for family, patterns in FAMILIES.items()
                 if any(fnmatchcase(task_id, pattern) for pattern in patterns)), None)
