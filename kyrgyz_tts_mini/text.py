"""Kyrgyz text → token ids. The voices read raw lowercase Cyrillic (no phonemizer); other characters are dropped."""

import re

PAD = "_"
PUNCTUATION = "!'(),.:;?!¡¿—…\"«»“” "
LETTERS = "абвгдеёжзийклмнңоөпрстуүфхцчшщьыъэюя"

# The ids are baked into the checkpoints: never reorder or dedupe (the second "!" is intentional).
SYMBOLS = [PAD, *PUNCTUATION, *LETTERS, "-"]

_ids = {s: i for i, s in enumerate(SYMBOLS)}
_whitespace = re.compile(r"\s+")


def clean(text: str) -> str:
    return _whitespace.sub(" ", text.lower())


def to_ids(text: str) -> list[int]:
    """Token ids with the pad id between every symbol and at both ends, as the voices were trained."""
    ids = [_ids[s] for s in clean(text) if s in _ids and s != PAD]
    result = [_ids[PAD]] * (2 * len(ids) + 1)
    result[1::2] = ids
    return result


def has_letters(text: str) -> bool:
    return any(c in LETTERS for c in clean(text))


def dropped_characters(text: str) -> str:
    """The distinct characters of `text` that the voices cannot pronounce, in order of appearance."""
    return "".join(dict.fromkeys(c for c in clean(text) if c not in _ids))
