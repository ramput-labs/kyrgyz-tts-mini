"""Kyrgyz text → token ids. Voices use raw lowercase Cyrillic (no phonemizer); other characters are dropped."""

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
    return _whitespace_re.sub(" ", text.lower())


def text_to_sequence(text: str) -> list[int]:
    return [_symbol_to_id[s] for s in clean(text) if s in _symbol_to_id and s != _pad]


def intersperse(lst: list, item) -> list:
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def has_letters(text: str) -> bool:
    return any(c in _letters for c in clean(text))


def dropped_characters(text: str) -> str:
    return "".join(dict.fromkeys(c for c in clean(text) if c not in _symbol_to_id))
