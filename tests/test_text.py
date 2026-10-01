from kyrgyz_tts_mini.text import SYMBOLS, clean, dropped_characters, has_letters, to_ids


def test_symbol_table_matches_checkpoints():
    assert len(SYMBOLS) == 58
    assert SYMBOLS[0] == "_" and SYMBOLS[-1] == "-"
    assert SYMBOLS.index("а") == 21 and SYMBOLS.index("ң") == 36


def test_clean_lowercases_and_collapses_whitespace():
    assert clean("Салам   Дүйнө\n!") == "салам дүйнө !"


def test_to_ids_intersperses_pad_and_skips_unknown_characters():
    a = SYMBOLS.index("а")
    assert to_ids("Аa1") == to_ids("а") == [0, a, 0]
    assert dropped_characters("Салам, world 42") == "world42"


def test_has_letters():
    assert has_letters("Өмүр")
    assert not has_letters("hello 123 !")
