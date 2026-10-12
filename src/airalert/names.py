"""Нормалізація назв адмінодиниць для зведення джерел до кодів (AC-2.1)."""

import re

# Різні знаки, якими в джерелах пишуть апостроф, зводимо до одного
_APOSTROPHES = str.maketrans({"'": "’", "ʼ": "’", "`": "’", "‘": "’"})
_SPACES = re.compile(r"\s+")


def normalize(name: str) -> str:
    """Уніфікує апострофи й пробіли та переводить назву в нижній регістр."""
    return _SPACES.sub(" ", name.translate(_APOSTROPHES)).strip().lower()
