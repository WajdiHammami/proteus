import copy
import datetime
import re
import uuid
from pathlib import PurePath
from typing import Any

import yaml

from src.adapters.models.normalized_document import NormalizedDocument, TextFormat
from src.adapters.models.raw_document import RawDocument
from src.adapters.normalizers.code_fence import fence_code
from src.adapters.normalizers.text_cleanup import clean_text
from src.ports.normalization_port import NormalizationPort, UnsupportedDocumentError


PLAIN_MIME_TYPES = frozenset({"text/plain"})
MARKDOWN_MIME_TYPES = frozenset({"text/markdown", "text/x-markdown"})
# mime type -> code fence language
CODE_MIME_TYPES = {
    "text/x-csrc": "c",
    "text/x-chdr": "c",
    "text/x-c++src": "cpp",
    "text/x-c++hdr": "cpp",
}

_FRONT_MATTER = re.compile(r"\A---[ \t]*\n(.*?)\n(?:---|\.\.\.)[ \t]*(?:\n|\Z)", re.DOTALL)
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_INDENTED_CODE = re.compile(r"^(?: {4}|\t)")
_HTML_COMMENT = re.compile(r"[ \t]*<!--.*?-->", re.DOTALL)
_IMAGE = re.compile(r"[ \t]*!\[[^\]]*\](?:\([^)]*\)|\[[^\]]*\])")
_HTML_IMAGE = re.compile(r"[ \t]*<img\b[^>]*>", re.IGNORECASE)
_EMPTY_LINK = re.compile(r"[ \t]*\[\s*\]\([^)]*\)")  # left behind by "[![badge](img)](link)"
_ATX_H1 = re.compile(r"^ {0,3}#[ \t]+(.+?)[ \t#]*$")
_SETEXT_H1_UNDERLINE = re.compile(r"^ {0,3}=+[ \t]*$")


class TextNormalizer(NormalizationPort):
    """
    Normalizes local text files read by the scraper:

      - text/plain:     shared cleanup only (no reflow: the text may be code)
      - text/markdown:  front matter -> metadata, HTML comments and images removed,
                        first H1 as title fallback; fenced/indented code is never touched
      - C/C++ sources:  wrapped in a single fenced block (```cpp / ```c) preceded by the
                        file name, so extraction knows for sure it is code
    """

    def supports(self, raw: RawDocument) -> bool:
        mime = _base_mime(raw.mime_type)
        return mime in PLAIN_MIME_TYPES or mime in MARKDOWN_MIME_TYPES or mime in CODE_MIME_TYPES

    def normalize(self, raw: RawDocument) -> NormalizedDocument:
        if not self.supports(raw):
            raise UnsupportedDocumentError(
                f"{type(self).__name__} does not support mime type '{raw.mime_type}'."
            )

        mime = _base_mime(raw.mime_type)
        metadata = copy.deepcopy(raw.metadata)

        if mime in CODE_MIME_TYPES:
            language = CODE_MIME_TYPES[mime]
            content = _fence_code(clean_text(raw.content), language, _file_name(metadata))
            metadata.update({"language": language, "page_metadata": {}})
            text_format = TextFormat.MARKDOWN
        elif mime in MARKDOWN_MIME_TYPES:
            content, page_metadata, front_matter = _normalize_markdown(raw.content)
            metadata.update({"page_metadata": page_metadata, "front_matter": front_matter})
            text_format = TextFormat.MARKDOWN
        else:
            content = clean_text(raw.content)
            metadata["page_metadata"] = {}
            text_format = TextFormat.PLAIN

        return NormalizedDocument(
            id=str(uuid.uuid4()),
            raw_document_id=raw.id,
            source_id=raw.source_id,
            content=content,
            format=text_format,
            metadata=metadata,
        )


# ---------------------------------------------------------------------- #
# Code files
# ---------------------------------------------------------------------- #

def _fence_code(code: str, language: str, file_name: str | None) -> str:
    block = fence_code(code, language)
    return f"{file_name}\n{block}" if file_name and block else block


def _file_name(metadata: dict[str, Any]) -> str | None:
    path = metadata.get("path") or metadata.get("url")
    return PurePath(path).name if path else None


# ---------------------------------------------------------------------- #
# Markdown
# ---------------------------------------------------------------------- #

def _normalize_markdown(text: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    text = clean_text(text)
    front_matter, text = _split_front_matter(text)

    segments = _split_code(text)
    cleaned = []
    for is_code, chunk in segments:
        if not is_code:
            chunk = _HTML_COMMENT.sub("", chunk)
            chunk = _IMAGE.sub("", chunk)
            chunk = _HTML_IMAGE.sub("", chunk)
            chunk = _EMPTY_LINK.sub("", chunk)
        cleaned.append(chunk)
    content = clean_text("".join(cleaned))

    page_metadata = _front_matter_metadata(front_matter)
    if "title" not in page_metadata:
        title = _first_h1(_split_code(content))
        if title:
            page_metadata["title"] = title
    return content, page_metadata, front_matter


def _split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return {}, text
    if not isinstance(data, dict):
        return {}, text
    return _json_safe(data), text[match.end():]


def _split_code(text: str) -> list[tuple[bool, str]]:
    """
    Splits markdown into (is_code, chunk) segments, keeping line endings so that
    "".join(chunks) == text. Code = fenced blocks (``` / ~~~, unclosed runs to EOF)
    and indented blocks (4+ spaces or a tab, starting after a blank line).
    """
    segments: list[tuple[bool, str]] = []

    def push(is_code: bool, line: str):
        if segments and segments[-1][0] == is_code:
            segments[-1] = (is_code, segments[-1][1] + line)
        else:
            segments.append((is_code, line))

    fence: str | None = None
    previous_blank = True      # start of document counts as "after a blank line"
    in_indented_code = False
    for line in text.splitlines(keepends=True):
        if fence:
            push(True, line)
            stripped = line.strip()
            if stripped.startswith(fence) and set(stripped) == {fence[0]}:
                fence = None
            previous_blank = False
            continue

        blank = not line.strip()
        opening = _FENCE_OPEN.match(line)
        if opening:
            fence = opening.group(1)
            in_indented_code = False
            push(True, line)
        elif not blank and _INDENTED_CODE.match(line) and (previous_blank or in_indented_code):
            # As in CommonMark, an indented line inside a paragraph is a continuation,
            # not code; indented code only starts after a blank line.
            in_indented_code = True
            push(True, line)
        else:
            if not blank:
                in_indented_code = False
            push(False, line)
        previous_blank = blank
    return segments


def _first_h1(segments: list[tuple[bool, str]]) -> str | None:
    for is_code, chunk in segments:
        if is_code:
            continue
        lines = chunk.split("\n")
        for i, line in enumerate(lines):
            atx = _ATX_H1.match(line)
            if atx:
                return atx.group(1).strip()
            if line.strip() and i + 1 < len(lines) and _SETEXT_H1_UNDERLINE.match(lines[i + 1]):
                return line.strip()
    return None


def _front_matter_metadata(front_matter: dict[str, Any]) -> dict[str, Any]:
    """Maps common front matter fields onto the shared page_metadata keys."""
    meta: dict[str, Any] = {}

    def first(*keys):
        for key in keys:
            value = front_matter.get(key)
            if value not in (None, "", []):
                return value
        return None

    title = first("title")
    if title is not None:
        meta["title"] = str(title).strip()

    author = first("author", "authors")
    if isinstance(author, list):
        author = ", ".join(str(a) for a in author)
    if author is not None:
        meta["author"] = str(author).strip()

    description = first("description", "summary", "excerpt")
    if description is not None:
        meta["description"] = str(description).strip()

    date = first("date", "published", "created")
    if date is not None:
        meta["date"] = str(date)[:10] if re.match(r"\d{4}-\d{2}-\d{2}", str(date)) else str(date)

    tags = first("tags", "keywords", "categories")
    if isinstance(tags, str):
        tags = [t.strip() for t in re.split(r"[,;]", tags)]
    if isinstance(tags, list):
        tags = [str(t).strip() for t in tags if str(t).strip()]
        if tags:
            meta["tags"] = tags

    return {k: v for k, v in meta.items() if v}


def _json_safe(value: Any) -> Any:
    """YAML can produce dates/datetimes; keep metadata JSON-serializable."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    return value


def _base_mime(mime_type: str) -> str:
    return (mime_type or "").split(";", 1)[0].strip().lower()
