import pytest

from src.adapters.normalizers.text_cleanup import clean_text


def test_nfc_normalization():
    decomposed = "café"  # e + combining acute
    assert clean_text(decomposed) == "café"


@pytest.mark.parametrize("char", ["​", "⁠", "﻿", "­"])
def test_invisible_characters_removed(char):
    assert clean_text(f"pri{char}ce") == "price"


def test_zwj_and_zwnj_are_kept():
    persian = "می‌خواهم"
    family_emoji = "👨‍👩‍👧"
    assert clean_text(persian) == persian
    assert clean_text(family_emoji) == family_emoji


@pytest.mark.parametrize("space", [" ", " ", " ", " ", "　"])
def test_exotic_spaces_become_regular(space):
    assert clean_text(f"10{space}kg") == "10 kg"


def test_trailing_whitespace_stripped_but_indentation_kept():
    text = "- item   \n  - nested\t\n    code"
    assert clean_text(text) == "- item\n  - nested\n    code"


def test_blank_line_runs_collapsed():
    assert clean_text("a\n\n\n\n\nb\n\nc") == "a\n\nb\n\nc"


def test_windows_and_old_mac_newlines():
    assert clean_text("a\r\nb\rc") == "a\nb\nc"


def test_outer_blank_lines_stripped():
    assert clean_text("\n  \n\nhello  \n\n  ") == "hello"


def test_first_line_indentation_kept():
    assert clean_text("\n\n    return x\nend") == "    return x\nend"


@pytest.mark.parametrize("text", ["", "   ", "\n\n​\n"])
def test_empty_inputs(text):
    assert clean_text(text) == ""


def test_meaning_preserved():
    text = "Price: $9.99 (EXCL. VAT) — 50% off!"
    assert clean_text(text) == text
