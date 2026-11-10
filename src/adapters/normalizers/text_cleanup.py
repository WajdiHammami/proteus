import re
import unicodedata

# Invisible characters that carry no meaning in extracted text.
# ZWJ/ZWNJ (U+200D/U+200C) are deliberately kept: they matter in Persian,
# Indic scripts and emoji sequences.
_INVISIBLE = dict.fromkeys(map(ord, "​⁠﻿­"))

# Non-breaking and other exotic spaces -> regular space.
_SPACES = re.compile(r"[  -   　]")

_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_LEADING_BLANK_LINES = re.compile(r"\A(?:[ \t]*\n)+")


def clean_text(text: str) -> str:
    """
    Light, meaning-preserving cleanup shared by normalizers:
      - Unicode NFC
      - drop zero-width / soft-hyphen characters
      - exotic spaces -> regular spaces
      - strip trailing whitespace per line (leading kept: markdown nesting depends on it)
      - collapse 3+ consecutive newlines into one blank line
      - drop leading/trailing blank lines (the first line's indentation is kept: it may be code)
    """
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_INVISIBLE)
    text = _SPACES.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_WS.sub("", text)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return _LEADING_BLANK_LINES.sub("", text).rstrip()
