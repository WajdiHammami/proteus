import copy
import logging
import re
import uuid
from typing import Any

from src.adapters.models.normalized_document import NormalizedDocument, TextFormat
from src.adapters.models.raw_document import RawDocument
from src.adapters.normalizers.text_cleanup import clean_text
from src.ports.normalization_port import NormalizationPort, UnsupportedDocumentError


logger = logging.getLogger(__name__)

PDF_MIME_TYPE = "application/pdf"
DEFAULT_PAGE_SEPARATOR = "\f"

# TODO(tables): pypdf flattens tables into space-separated runs, so column structure is
#   lost before it reaches this normalizer (and internal space runs are collapsed below).
#   If tables matter, switch the PDF scraper to a layout-aware reader (e.g. pdfplumber)
#   that emits tables explicitly, then render them as markdown here.
# TODO(reading-order): multi-column layouts and RTL scripts can come out of pypdf in the
#   wrong order. This can't be repaired from plain text; it also needs a layout-aware reader.

_LIGATURES = str.maketrans({
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
    "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
})

_LEADING_WS = re.compile(r"^[ \t]*")
_INNER_WS_RUN = re.compile(r"(?<=\S)[ \t]{2,}(?=\S)")
# "infor-\nmation" -> "information". The next line must start directly with a lowercase
# letter (no indentation), which keeps indented code continuation lines untouched.
_HYPHENATED_BREAK = re.compile(r"(?<=[A-Za-z])-\n(?=[a-z])")

# Explicit page-number lines, removed when found in a header/footer zone.
_PAGE_LABEL = re.compile(
    r"^(?:page|p\.|pg\.?)\s*\d+(?:\s*(?:of|/)\s*\d+)?$"   # Page 3, Page 3 of 10, p. 3
    r"|^\d+\s*(?:of|/)\s*\d+$"                             # 3 of 10, 3/10
    r"|^[-–—]\s*\d+\s*[-–—]$",                             # - 3 -
    re.IGNORECASE,
)
_BARE_NUMBER = re.compile(r"^\d{1,5}$")
_DIGITS = re.compile(r"\d+")
_LETTER = re.compile(r"[^\W\d_]")
# Language names that contain code punctuation, neutralized before the code check.
_LANGUAGE_NAMES = re.compile(r"(?<!\w)(?:[Cc]\+\+|[CcFf]#)(?:\d+)?(?!\w)")
_CODE_CHARS = re.compile(r"[=+*/<>(){}\[\];`\\$#@%^~&|!\"']|::|->|\w:$")
_CODE_START = re.compile(
    r"^(?:return|def|class|import|from|if|elif|else|for|while|try|except|finally|with|yield|"
    r"break|continue|pass|raise|end|fi|done|esac|then|do|var|let|const|function|func|fn|"
    r"public|private|protected|static|package|namespace|using|echo|print|async|await|"
    # C++ (case-sensitive on purpose: a header like "New features" must not match "new")
    r"case|default|switch|goto|catch|throw|template|typename|struct|enum|union|typedef|"
    r"virtual|inline|explicit|friend|operator|auto|void|int|char|bool|double|float|long|"
    r"short|unsigned|signed|size_t|std|constexpr|consteval|constinit|noexcept|override|final|"
    r"delete|new|sizeof|alignof|decltype|static_assert|extern|mutable|volatile|concept|"
    r"requires|co_await|co_return|co_yield|nullptr|this|true|false)\b"
)


class PdfNormalizer(NormalizationPort):
    """
    Compacts raw pypdf text (as produced by PdfScraperAdapter) into token-efficient
    plain text while preserving code: line breaks and relative indentation are kept.

    Per page:
      - text of pages flagged for OCR (scanned / garbled) is dropped
      - ligatures expanded, common left margin removed, inner space runs collapsed
      - lines indented beyond `max_indent` (centered titles etc.) are left-stripped
    Across pages:
      - repeated headers/footers and page numbers are removed
      - hyphenated line breaks are rejoined ("infor-\\nmation" -> "information")
      - pages are joined with a single newline; char offsets stored in metadata["page_spans"]
    """

    def __init__(
        self,
        margin_lines: int = 2,
        min_repeat_ratio: float = 0.5,
        min_pages_for_repeats: int = 3,
        max_indent: int = 24,
    ):
        """
        Args:
            margin_lines: Number of non-empty lines at the top/bottom of each page
                considered as header/footer candidates.
            min_repeat_ratio: Share of pages a line must appear on to be treated as a header/footer.
            min_pages_for_repeats: Minimum page count before repeated-line removal is attempted.
            max_indent: Lines indented more than this (after margin removal) are left-stripped.
        """
        self.margin_lines = margin_lines
        self.min_repeat_ratio = min_repeat_ratio
        self.min_pages_for_repeats = min_pages_for_repeats
        self.max_indent = max_indent

    def supports(self, raw: RawDocument) -> bool:
        return (raw.mime_type or "").split(";", 1)[0].strip().lower() == PDF_MIME_TYPE

    def normalize(self, raw: RawDocument) -> NormalizedDocument:
        if not self.supports(raw):
            raise UnsupportedDocumentError(
                f"{type(self).__name__} does not support mime type '{raw.mime_type}'."
            )

        metadata = copy.deepcopy(raw.metadata)
        pages = self._split_pages(raw, metadata)

        dropped = set(metadata.get("text_layer", {}).get("ocr_pages", []))
        page_lines = [
            [] if index in dropped else self._compact_page(text)
            for index, text in enumerate(pages)
        ]

        boilerplate = self._remove_boilerplate(page_lines)
        content, page_spans = self._join_pages(["\n".join(lines) for lines in page_lines])

        metadata.update({
            "page_spans": page_spans,
            "dropped_pages": sorted(i for i in dropped if i < len(pages)),
            "removed_boilerplate": boilerplate,
            "page_metadata": _page_metadata(metadata.get("pdf_metadata", {})),
        })

        return NormalizedDocument(
            id=str(uuid.uuid4()),
            raw_document_id=raw.id,
            source_id=raw.source_id,
            content=content,
            format=TextFormat.PLAIN,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # Pages
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_pages(raw: RawDocument, metadata: dict[str, Any]) -> list[str]:
        if not raw.content:
            return []
        separator = metadata.get("page_separator") or DEFAULT_PAGE_SEPARATOR
        pages = raw.content.split(separator)
        expected = metadata.get("page_count")
        if expected is not None and len(pages) != expected:
            logger.warning(
                "PDF page count mismatch (%d split vs %d expected), treating as a single page: %s",
                len(pages), expected, raw.id,
            )
            return [raw.content]
        return pages

    def _compact_page(self, text: str) -> list[str]:
        text = clean_text(text.translate(_LIGATURES))
        lines = text.split("\n")

        indents = [len(_LEADING_WS.match(line).group()) for line in lines if line.strip()]
        margin = min(indents, default=0)

        compacted = []
        for line in lines:
            if not line.strip():
                compacted.append("")
                continue
            line = line[margin:]
            indent = _LEADING_WS.match(line).group()
            body = _INNER_WS_RUN.sub(" ", line[len(indent):])
            compacted.append(body if len(indent) > self.max_indent else indent + body)

        return _collapse_blank_lines(compacted)

    @staticmethod
    def _join_pages(pages: list[str]) -> tuple[str, list[list[int]]]:
        """Joins pages with '\\n', rejoining a word hyphenated across a page break."""
        parts: list[str] = []
        spans: list[list[int]] = []
        cursor = 0
        previous_non_empty: int | None = None

        for page in pages:
            page = _HYPHENATED_BREAK.sub("", page)
            separator = "\n" if parts and page else ""

            if (
                page and previous_non_empty is not None
                and re.search(r"[A-Za-z]-$", parts[previous_non_empty])
                and page[0].islower()
            ):
                # Drop the trailing hyphen of the previous page and glue the words.
                parts[previous_non_empty] = parts[previous_non_empty][:-1]
                cursor -= 1
                spans[previous_non_empty][1] -= 1
                separator = ""

            start = cursor + len(separator)
            parts.append(separator + page if page else "")
            cursor = start + len(page) if page else cursor
            spans.append([start if page else cursor, cursor])
            if page:
                previous_non_empty = len(parts) - 1

        return "".join(parts), spans

    # ------------------------------------------------------------------ #
    # Headers / footers
    # ------------------------------------------------------------------ #

    def _remove_boilerplate(self, page_lines: list[list[str]]) -> list[str]:
        """Removes repeated header/footer lines and page numbers in place. Returns removed patterns."""
        zones = [self._zone_indices(lines) for lines in page_lines]
        to_remove: list[set[int]] = [set() for _ in page_lines]

        # Explicit page labels ("Page 3 of 10", "- 3 -") inside a zone.
        for page_index, (lines, zone) in enumerate(zip(page_lines, zones)):
            for i in zone:
                if _PAGE_LABEL.match(lines[i].strip()):
                    to_remove[page_index].add(i)

        # Bare numbers: only when they track the page index (number - index constant),
        # so a stray "42" at the end of a code listing is left alone.
        offsets: dict[int, list[tuple[int, int]]] = {}
        for page_index, (lines, zone) in enumerate(zip(page_lines, zones)):
            for i in zone:
                value = lines[i].strip()
                if _BARE_NUMBER.match(value):
                    offsets.setdefault(int(value) - page_index, []).append((page_index, i))
        for hits in offsets.values():
            if len({page for page, _ in hits}) >= 2:
                for page_index, i in hits:
                    to_remove[page_index].add(i)

        # Repeated text lines at the same position (e.g. always the 1st line, or always the
        # last line). Digits are ignored, so "Chapter 3" matches "Chapter 4".
        removed_patterns: set[str] = set()
        pages_with_text = sum(1 for lines in page_lines if any(lines))
        if pages_with_text >= self.min_pages_for_repeats:
            locations: dict[tuple[str, str], list[tuple[int, int]]] = {}
            for page_index, lines in enumerate(page_lines):
                for slot, i in self._zone_slots(lines):
                    key = _repeat_key(lines[i])
                    if key:
                        locations.setdefault((slot, key), []).append((page_index, i))

            threshold = max(2, self.min_repeat_ratio * pages_with_text)
            for (_, key), hits in locations.items():
                if len(hits) >= threshold:
                    removed_patterns.add(key)
                    for page_index, i in hits:
                        to_remove[page_index].add(i)

        for page_index, indices in enumerate(to_remove):
            if indices:
                kept = [line for i, line in enumerate(page_lines[page_index]) if i not in indices]
                page_lines[page_index][:] = _collapse_blank_lines(kept)

        return sorted(removed_patterns)

    def _zone_indices(self, lines: list[str]) -> list[int]:
        return sorted({i for _, i in self._zone_slots(lines)})

    def _zone_slots(self, lines: list[str]) -> list[tuple[str, int]]:
        """(slot, line index) pairs: 'top0' = first non-empty line, 'bottom0' = last, etc."""
        non_empty = [i for i, line in enumerate(lines) if line.strip()]
        n = self.margin_lines
        if n <= 0:
            return []
        top = [(f"top{k}", i) for k, i in enumerate(non_empty[:n])]
        bottom = [(f"bottom{k}", i) for k, i in enumerate(reversed(non_empty[-n:]))]
        return top + bottom


def _repeat_key(line: str) -> str | None:
    """Comparison key for header/footer detection; None for lines that can't be boilerplate."""
    stripped = line.strip()
    # Require some letters: lines like "}" or ");" repeat at page ends in code listings.
    if len(_LETTER.findall(stripped)) < 3:
        return None
    # Code punctuation never appears in running headers/footers but is common in code.
    # "Modern C++ Patterns" is a header, not code: ignore language names for this check.
    probe = _LANGUAGE_NAMES.sub("Lang", stripped)
    if _CODE_CHARS.search(probe) or _CODE_START.match(probe):
        return None
    return _DIGITS.sub("#", " ".join(stripped.lower().split()))


def _collapse_blank_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        if not line and (not out or not out[-1]):
            continue
        out.append(line)
    while out and not out[-1]:
        out.pop()
    return out


# ---------------------------------------------------------------------- #
# Metadata
# ---------------------------------------------------------------------- #

_PDF_DATE = re.compile(r"^D:(\d{4})(\d{2})?(\d{2})?")


def _page_metadata(pdf_meta: dict[str, str]) -> dict[str, Any]:
    """Maps PDF info fields onto the same keys the HTML normalizer uses."""
    meta: dict[str, Any] = {}
    for source_key, target_key in (("Title", "title"), ("Author", "author"), ("Subject", "description")):
        value = (pdf_meta.get(source_key) or "").strip()
        if value:
            meta[target_key] = value

    keywords = [k.strip() for k in re.split(r"[,;]", pdf_meta.get("Keywords") or "") if k.strip()]
    if keywords:
        meta["tags"] = keywords

    date = _parse_pdf_date(pdf_meta.get("CreationDate") or pdf_meta.get("ModDate") or "")
    if date:
        meta["date"] = date
    return meta


def _parse_pdf_date(value: str) -> str | None:
    """'D:20260301120000+01'00'' -> '2026-03-01' (missing month/day default to 01)."""
    match = _PDF_DATE.match(value.strip())
    if not match:
        return None
    year, month, day = match.group(1), match.group(2) or "01", match.group(3) or "01"
    if not ("01" <= month <= "12" and "01" <= day <= "31"):
        return None
    return f"{year}-{month}-{day}"
