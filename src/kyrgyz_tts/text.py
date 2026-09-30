"""Kyrgyz text → token ids for the kyrgyz-tts Matcha voices.

The voices were trained on lowercased Cyrillic characters (no phonemizer). Characters outside
the symbol set, such as digits and Latin letters, are dropped.
"""

import re

_pad = "_"
_punctuation = "!'(),.:;?!¡¿—…\"«»“” "
_special = "-"
_letters = "абвгдеёжзийклмнңоөпрстуүфхцчшщьыъэюя"

# Order matters: the ids are baked into the checkpoints.
SYMBOLS = [_pad] + list(_punctuation) + list(_letters) + list(_special)

_symbol_to_id = {s: i for i, s in enumerate(SYMBOLS)}
_whitespace_re = re.compile(r"\s+")


def clean(text: str) -> str:
    """Lowercase and collapse whitespace (Matcha's `basic_cleaners`)."""
    return _whitespace_re.sub(" ", text.lower())


def text_to_sequence(text: str) -> list[int]:
    return [_symbol_to_id[s] for s in clean(text) if s in _symbol_to_id and s != _pad]


def intersperse(lst: list, item) -> list:
    """Put `item` between and around every element: [a, b] -> [item, a, item, b, item]."""
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def has_letters(text: str) -> bool:
    return any(c in _letters for c in clean(text))


def dropped_characters(text: str) -> str:
    """Characters of `text` the model cannot pronounce (they are skipped)."""
    return "".join(dict.fromkeys(c for c in clean(text) if c not in _symbol_to_id))
