from kyrgyz_tts_mini.text import SYMBOLS, clean, dropped_characters, has_letters, intersperse, text_to_sequence


def test_symbol_table_matches_checkpoints():
    # Token ids are baked into the checkpoints; this order must never change.
    assert len(SYMBOLS) == 58
    assert SYMBOLS[0] == "_" and SYMBOLS[-1] == "-"
    assert SYMBOLS.index("а") == 21 and SYMBOLS.index("ң") == 36


def test_clean_lowercases_and_collapses_whitespace():
    assert clean("Салам   Дүйнө\n!") == "салам дүйнө !"


def test_text_to_sequence_skips_unknown_characters():
    assert text_to_sequence("Аa1") == text_to_sequence("а")
    assert dropped_characters("Салам, world 42") == "world42"


def test_has_letters():
    assert has_letters("Өмүр")
    assert not has_letters("hello 123 !")


def test_intersperse():
    assert intersperse([1, 2], 0) == [0, 1, 0, 2, 0]
