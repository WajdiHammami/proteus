"""
Finds code regions in plain text (PDF output, .txt files), where nothing marks code.

Each line is labelled CODE, PROSE, NEUTRAL or BLANK by weighing C/C++ signals
(`;`/`{`/`}` endings, keywords, preprocessor, operators) against sentence signals
(mostly plain words, final period, capital start). A region starts at a CODE line and
grows over following CODE lines, tolerating up to `max_gap` BLANK/NEUTRAL lines in
between (wrapped lines, blank lines inside a function); a PROSE line ends it.
Regions are then cleaned (margin line numbers, common indentation) and handed to the
same tree-sitter check as fenced blocks.
"""

import re
from dataclasses import dataclass
from enum import Enum

from src.adapters.extractors.candidate_block import CandidateBlock


class LineKind(Enum):
    CODE = "code"
    PROSE = "prose"
    NEUTRAL = "neutral"
    BLANK = "blank"


# --------------------------------------------------------------------------- #
# Line classification
# --------------------------------------------------------------------------- #

_LINE_NUMBER = re.compile(r"^\s*(\d{1,4})(?:\s+|$)")
_LINE_NUMBER_PREFIX = re.compile(r"^\s*\d{1,4}(?: |\t|$)")   # number + one separator; keeps the code's indent
_PREPROCESSOR = re.compile(r"^#\s*(?:include|define|undef|ifn?def|if|elif|else|endif|pragma|error)\b")
_COMMENT = re.compile(r"^(?://|/\*|\*/|\*\s|\*$)")
_BRACES_ONLY = re.compile(r"^[{}()\[\];,\s]+$")
_LABEL = re.compile(r"^(?:(?:public|private|protected)\s*:|(?:case\s.+|default)\s*:)$")
_KEYWORD_START = re.compile(
    r"^(?:template\s*<|(?:class|struct|union)\s+\w|enum\s+(?:class\s+)?\w|namespace\b|using\s|typedef\s|"
    r"return\b|if\s*\(|else\b|for\s*\(|while\s*\(|do\s*\{|switch\s*\(|try\s*\{|catch\s*\(|throw\b|"
    r"co_(?:return|await|yield)\b|static_assert\s*\(|"
    r"(?:const|constexpr|consteval|constinit|static|virtual|inline|explicit|extern|friend|mutable|volatile|"
    r"unsigned|signed|typename)\s|"
    r"(?:void|int|char|bool|double|float|long|short|auto|size_t|std::\w+)\b[\s*&<:(])"
)
_CODE_END = re.compile(r"(?:;|\{|\}|\};)\s*$")
_CALL_END = re.compile(r"\)\s*(?:const|override|noexcept|final|=\s*(?:default|delete|0))?\s*$")
_INITIALIZER = re.compile(r"^[:,]\s*\w+\s*[({]")
_OPERATORS = re.compile(r"::|->|&&|\|\||<<|>>|==|!=|\+=|-=|\+\+|--|\w\(|\[\]|\w\[|\s=\s")
_TRAILING_COMMENT = re.compile(r"\s//.*$")
_WORD = re.compile(r"^[A-Za-z][a-z'’]*[.,;:!?)]?$|^[A-Z]{1,2}[.,]?$")


def classify_line(line: str) -> LineKind:
    text = line.strip()
    if not text:
        return LineKind.BLANK
    numbered = _LINE_NUMBER.match(text)
    if numbered and numbered.end() < len(text):
        text = text[numbered.end():]  # judge "12 int x;" by its code

    if _PREPROCESSOR.match(text) or _COMMENT.match(text) or _BRACES_ONLY.match(text) or _LABEL.match(text):
        return LineKind.CODE

    code_part = _TRAILING_COMMENT.sub("", text).rstrip()
    code = 0
    if _KEYWORD_START.match(code_part):
        code += 2
    if _CODE_END.search(code_part):
        code += 2
    elif _CALL_END.search(code_part):
        code += 1
    if _INITIALIZER.match(code_part):
        code += 1
    code += min(2, len(_OPERATORS.findall(code_part)))

    tokens = text.split()
    words = sum(1 for token in tokens if _WORD.match(token))
    prose = 0
    if words >= 5 and words / len(tokens) >= 0.7:
        prose += 2
    if text[-1] in ".?!" or (text[-1] == ":" and words >= 2):
        prose += 1
    if text[0].isupper() and words >= 3:
        prose += 1

    if code >= 2 and code > prose:
        return LineKind.CODE
    if prose >= 2:
        return LineKind.PROSE
    return LineKind.NEUTRAL


# --------------------------------------------------------------------------- #
# Headings, captions, context
# --------------------------------------------------------------------------- #

# "3.2 Move Semantics", "Chapter 4: Templates", "Section 2 Basics"
_HEADING = re.compile(r"^(?:(?:Chapter|Section|Part)\s+\d+\b.*|\d+(?:\.\d+){0,3}\.?\s+[A-Z][^.!?;{}]*)$")
# "Listing 3.4 A movable buffer", "Example 2-1. Hello", "Code 5"
_CAPTION = re.compile(r"^(?:Listing|Example|Code|Snippet|Program)\s+[\dA-Z]+(?:[.\-]\d+)*\b", re.IGNORECASE)
_MAX_HEADING_WORDS = 12


def _is_heading(text: str) -> bool:
    return len(text) <= 80 and len(text.split()) <= _MAX_HEADING_WORDS and bool(_HEADING.match(text))


@dataclass
class _Region:
    start: int
    end: int    # inclusive


def find_code_blocks(text: str, max_gap: int = 3) -> list[CandidateBlock]:
    lines = text.split("\n")
    kinds = [classify_line(line) for line in lines]

    blocks: list[CandidateBlock] = []
    heading: str | None = None
    caption: str | None = None
    paragraph: list[str] = []
    last_paragraph = ""

    def finish_paragraph():
        nonlocal paragraph, last_paragraph
        if paragraph:
            last_paragraph = " ".join(paragraph)
            paragraph = []

    i = 0
    while i < len(lines):
        kind, stripped = kinds[i], lines[i].strip()
        if kind == LineKind.CODE:
            region = _grow_region(kinds, i, max_gap)
            finish_paragraph()
            context = " ".join(part for part in (last_paragraph, caption) if part)
            blocks.append(CandidateBlock(_clean_region(lines[region.start:region.end + 1]), None, heading, context))
            last_paragraph, caption = "", None
            i = region.end + 1
            continue

        if kind == LineKind.BLANK:
            finish_paragraph()
        elif _CAPTION.match(stripped):
            finish_paragraph()
            caption = stripped
        elif _is_heading(stripped):
            finish_paragraph()
            heading, last_paragraph, caption = stripped, "", None
        elif kind == LineKind.NEUTRAL and lines[i][:1].isspace():
            # Indented non-prose (program output, a formula): not part of the explanation.
            finish_paragraph()
            last_paragraph = ""
        else:
            paragraph.append(stripped)
        i += 1
    return blocks


def _grow_region(kinds: list[LineKind], start: int, max_gap: int) -> _Region:
    end, gap = start, 0
    for j in range(start + 1, len(kinds)):
        if kinds[j] == LineKind.CODE:
            end, gap = j, 0
        elif kinds[j] == LineKind.PROSE:
            break
        else:
            gap += 1
            if gap > max_gap:
                break
    return _Region(start, end)


def _clean_region(lines: list[str]) -> str:
    lines = _strip_line_numbers(lines)
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    margin = min(indents, default=0)
    return "\n".join(line[margin:].rstrip() for line in lines).strip("\n")


def _strip_line_numbers(lines: list[str]) -> list[str]:
    """Removes margin line numbers when (nearly) every line has one and they increase."""
    content = [line for line in lines if line.strip()]
    numbers = [_LINE_NUMBER.match(line) for line in content]
    if len(content) < 2 or sum(1 for m in numbers if m) < 0.8 * len(content):
        return lines
    values = [int(m.group(1)) for m in numbers if m]
    if any(b <= a for a, b in zip(values, values[1:])):
        return lines
    return [_LINE_NUMBER_PREFIX.sub("", line, count=1) for line in lines]
