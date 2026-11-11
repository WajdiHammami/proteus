import copy
import json
import logging
import re
import uuid
from typing import Any

import lxml.html
import trafilatura
from lxml.html import HtmlElement
from trafilatura.utils import load_html

from src.adapters.models.normalized_document import NormalizedDocument, TextFormat
from src.adapters.models.raw_document import RawDocument
from src.adapters.normalizers.code_fence import fence_code
from src.adapters.normalizers.text_cleanup import clean_text
from src.ports.normalization_port import NormalizationPort, UnsupportedDocumentError


logger = logging.getLogger(__name__)

HTML_MIME_TYPES = frozenset({"text/html", "application/xhtml+xml"})

# OpenGraph and its object namespaces, plus Twitter cards.
_META_PREFIXES = ("og:", "article:", "product:", "book:", "profile:", "music:", "video:", "twitter:")

# Wrappers sometimes left around JSON-LD payloads by CMSs.
_JSON_LD_WRAPPERS = re.compile(r"^\s*(?://\s*)?(?:<!\[CDATA\[|<!--)|(?://\s*)?(?:\]\]>|-->)\s*$")


class ExtractionMode:
    MAIN = "main"            # trafilatura main-content extraction
    RECALL = "recall"        # trafilatura with favor_recall
    FULL_TEXT = "full_text"  # all visible text (html2txt), boilerplate included
    EMPTY = "empty"          # nothing usable found


class TrafilaturaHtmlNormalizer(NormalizationPort):
    """
    Turns raw HTML into markdown using trafilatura, plus page metadata and
    embedded structured data (JSON-LD, OpenGraph).

    Content is extracted with a cascade, stopping at the first result that has
    at least `min_chars` characters:
        main -> recall -> full_text
    If none reaches the threshold, the longest non-empty candidate is kept.
    The step used is recorded in metadata["extraction_mode"].
    """

    def __init__(
        self,
        include_links: bool = True,
        include_tables: bool = True,
        include_comments: bool = False,
        favor_precision: bool = False,
        min_chars: int = 100,
    ):
        """
        Args:
            include_links: Keep hyperlinks as markdown links (resolved against the page URL).
            include_tables: Keep tables as markdown tables.
            include_comments: Keep user comment sections.
            favor_precision: Stricter main-content detection on the first pass.
            min_chars: Minimum content length to accept an extraction step.
        """
        self.include_links = include_links
        self.include_tables = include_tables
        self.include_comments = include_comments
        self.favor_precision = favor_precision
        self.min_chars = min_chars

    def supports(self, raw: RawDocument) -> bool:
        return _base_mime(raw.mime_type) in HTML_MIME_TYPES

    def normalize(self, raw: RawDocument) -> NormalizedDocument:
        if not self.supports(raw):
            raise UnsupportedDocumentError(
                f"{type(self).__name__} does not support mime type '{raw.mime_type}'."
            )

        url = raw.metadata.get("url")
        tree = load_html(raw.content) if raw.content.strip() else None

        html = raw.content
        code_blocks: dict[str, tuple[str, str | None]] = {}
        if tree is not None:
            root = tree.getroottree().getroot()
            unwrapped = _unwrap_code_tables(root)
            code_blocks = _protect_code_blocks(root)
            if unwrapped or code_blocks:
                html = lxml.html.tostring(root, encoding="unicode")

        if tree is not None:
            content, mode = self._extract_content(html, url, code_blocks)
        else:
            content, mode = "", ExtractionMode.EMPTY
        if mode != ExtractionMode.MAIN:
            logger.info("HTML normalized with extraction_mode=%s: %s", mode, url or raw.id)

        return NormalizedDocument(
            id=str(uuid.uuid4()),
            raw_document_id=raw.id,
            source_id=raw.source_id,
            content=content,
            format=TextFormat.MARKDOWN,
            metadata={
                # deep copy: later stages must not be able to mutate the raw document through shared nested values
                **copy.deepcopy(raw.metadata),
                "extraction_mode": mode,
                "page_metadata": self._page_metadata(raw.content, tree, url),
                "structured_data": {
                    "json_ld": _json_ld(tree),
                    "opengraph": _opengraph(tree),
                },
            },
        )

    # ------------------------------------------------------------------ #
    # Content
    # ------------------------------------------------------------------ #

    def _extract_content(
        self, html: str, url: str | None, code_blocks: dict[str, tuple[str, str | None]]
    ) -> tuple[str, str]:
        steps = [
            (ExtractionMode.MAIN, lambda: self._trafilatura(html, url, favor_precision=self.favor_precision)),
            (ExtractionMode.RECALL, lambda: self._trafilatura(html, url, favor_recall=True)),
            (ExtractionMode.FULL_TEXT, lambda: trafilatura.html2txt(html)),
        ]

        best_text, best_mode = "", ExtractionMode.EMPTY
        for mode, step in steps:
            try:
                text = clean_text(_restore_code_blocks(step() or "", code_blocks))
            except Exception:
                logger.debug("Extraction step %s failed", mode, exc_info=True)
                continue
            if len(text) >= self.min_chars:
                return text, mode
            if len(text) > len(best_text):
                best_text, best_mode = text, mode
        return best_text, best_mode

    def _trafilatura(self, html: str, url: str | None, **options) -> str | None:
        return trafilatura.extract(
            html,
            url=url,
            output_format="markdown",
            include_links=self.include_links,
            include_tables=self.include_tables,
            include_comments=self.include_comments,
            **options,
        )

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #

    @staticmethod
    def _page_metadata(html: str, tree: HtmlElement | None, url: str | None) -> dict[str, Any]:
        if tree is None:
            return {}

        meta: dict[str, Any] = {}
        try:
            doc = trafilatura.extract_metadata(html, default_url=url)
            for key in ("title", "author", "date", "sitename", "description", "url", "image"):
                value = getattr(doc, key, None)
                if value:
                    meta[key] = value
            for key in ("categories", "tags"):
                values = getattr(doc, key, None)
                if values:
                    meta[key] = list(values)
        except Exception:
            logger.debug("Metadata extraction failed", exc_info=True)

        language = tree.getroottree().getroot().get("lang")
        if language and language.strip():
            meta["language"] = language.strip()
        return meta


_GUTTER_TEXT = re.compile(r"^[\s\d]*$")
# Per-line code cells: GitHub gists / GitHub blob views, highlight.js line-numbers plugin.
_CODE_CELL_CLASSES = frozenset({"blob-code", "blob-code-inner", "js-file-line", "hljs-ln-code"})


def _unwrap_code_tables(root: HtmlElement) -> int:
    """
    Replaces code blocks rendered as tables (line-number gutter + code) with a plain
    <pre><code>, so trafilatura emits a fenced code block instead of a markdown table
    that merges lines, mixes in line numbers and escapes '*'.

    Handles:
      - Pygments/Hexo/Jekyll style: <table><td><pre>1\\n2</pre></td><td><pre>code</pre></td></table>
      - GitHub gist style: one <tr> per line, <td class="blob-num"> + <td class="blob-code">

    A table is only unwrapped when *every* cell is either a gutter (digits/blank) or code,
    so real data tables that merely contain a <pre> somewhere are left untouched.
    Returns the number of tables replaced.
    """
    replaced = 0
    for table in root.xpath("//table"):
        if table.getparent() is None or table.xpath("ancestor::table"):
            continue
        cells = table.xpath(".//td")
        if not cells:
            continue

        if table.xpath(".//pre"):
            if not all(cell.xpath(".//pre") or _GUTTER_TEXT.match(cell.text_content()) for cell in cells):
                continue
            code_blocks = [p.text_content() for p in table.xpath(".//pre") if not _GUTTER_TEXT.match(p.text_content())]
            code = "\n".join(block.rstrip("\n") for block in code_blocks)
        else:
            def is_code_cell(cell) -> bool:
                return bool(set(cell.get("class", "").split()) & _CODE_CELL_CLASSES)

            if not any(is_code_cell(c) for c in cells):
                continue
            if not all(is_code_cell(c) or _GUTTER_TEXT.match(c.text_content()) for c in cells):
                continue
            code = "\n".join(c.text_content().rstrip("\n") for c in cells if is_code_cell(c))

        if not code.strip():
            continue

        pre = lxml.html.Element("pre")
        lxml.html.etree.SubElement(pre, "code").text = code
        pre.tail = table.tail
        table.getparent().replace(table, pre)
        replaced += 1
    return replaced


# --------------------------------------------------------------------------- #
# Code blocks
# --------------------------------------------------------------------------- #
#
# trafilatura drops blank lines inside <pre>, ignores <br> line breaks and loses the
# language class. So every <pre> is replaced by a unique placeholder before extraction;
# trafilatura only decides *where* the code goes (or whether it is boilerplate), and the
# original code is put back afterwards as a fenced block with its language.

# class="language-cpp" (Prism, highlight.js, CommonMark renderers), "lang-cpp" (Google
# prettify), "highlight-cpp" (Sphinx), "brush: cpp" (SyntaxHighlighter), "sourceCode cpp" (pandoc)
_LANGUAGE_CLASS = re.compile(r"(?:^|\s)(?:language|lang|highlight|brush)[-:]\s*([A-Za-z0-9_+#-]+)")
_SOURCECODE_CLASS = re.compile(r"(?:^|\s)sourceCode\s+([A-Za-z0-9_+#-]+)")
# Bare class names are too ambiguous in general ("highlight", "notranslate", ...),
# so only C/C++ ones are trusted (e.g. <pre class="cpp geshi">, class="hljs cpp").
_BARE_LANGUAGE_CLASSES = {
    "cpp": "cpp", "c++": "cpp", "cxx": "cpp", "cc": "cpp", "hpp": "cpp", "c": "c",
}
_LANGUAGE_ALIASES = {
    "c++": "cpp", "cxx": "cpp", "cc": "cpp", "hpp": "cpp", "h++": "cpp", "cplusplus": "cpp", "objective-c++": "objcpp",
}
_NOT_LANGUAGES = frozenset({"none", "text", "plain", "plaintext", "txt", "nohighlight", "default"})
_LANGUAGE_SEARCH_DEPTH = 3  # ancestors of <pre> to look at (Sphinx puts it on the grandparent)


def _protect_code_blocks(root: HtmlElement) -> dict[str, tuple[str, str | None]]:
    """
    Replaces each outermost <pre> with <pre><code>PLACEHOLDER</code></pre> and returns
    {placeholder: (code, language)}. A <pre> inside a table cell is left alone: a fenced
    block would break the markdown table (code tables were already unwrapped).
    """
    blocks: dict[str, tuple[str, str | None]] = {}
    nonce = uuid.uuid4().hex[:12]
    for pre in root.xpath("//pre[not(ancestor::pre) and not(ancestor::td) and not(ancestor::th)]"):
        parent = pre.getparent()
        if parent is None:
            continue
        code = _pre_text(pre)
        if not code.strip():
            continue
        # Letters and digits only: nothing for markdown conversion to escape.
        placeholder = f"proteuscode{nonce}x{len(blocks)}z"
        blocks[placeholder] = (code, _code_language(pre))

        replacement = lxml.html.Element("pre")
        lxml.html.etree.SubElement(replacement, "code").text = placeholder
        replacement.tail = pre.tail
        parent.replace(pre, replacement)
    return blocks


def _pre_text(pre: HtmlElement) -> str:
    """Text of a <pre>, with <br> as line breaks; strips the blank lines around the code."""
    pre = copy.deepcopy(pre)
    for br in pre.iter("br"):
        br.tail = "\n" + (br.tail or "")
    text = pre.text_content().replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines).rstrip()


def _code_language(pre: HtmlElement) -> str | None:
    candidates = [pre, *pre.xpath("./code")]
    node = pre.getparent()
    for _ in range(_LANGUAGE_SEARCH_DEPTH):
        if node is None:
            break
        candidates.append(node)
        node = node.getparent()

    for element in candidates:
        for attribute in ("data-lang", "data-language"):
            language = _clean_language(element.get(attribute))
            if language:
                return language
        classes = element.get("class") or ""
        for pattern in (_LANGUAGE_CLASS, _SOURCECODE_CLASS):
            for match in pattern.finditer(classes):
                language = _clean_language(match.group(1))
                if language:
                    return language
        for name in classes.split():
            bare = _BARE_LANGUAGE_CLASSES.get(name.lower())
            if bare:
                return bare
    return None


def _clean_language(value: str | None) -> str | None:
    value = (value or "").strip().lower()
    if not value or value in _NOT_LANGUAGES or not re.fullmatch(r"[a-z0-9_+#-]{1,30}", value):
        return None
    return _LANGUAGE_ALIASES.get(value, value)


def _restore_code_blocks(text: str, blocks: dict[str, tuple[str, str | None]]) -> str:
    """Puts the protected code back, as fenced blocks, wherever the extractor kept it."""
    if not blocks or "proteuscode" not in text:
        return text

    def fenced(placeholder: str) -> str:
        code, language = blocks[placeholder]
        return fence_code(code, language)

    names = "|".join(re.escape(name) for name in blocks)
    # Inside the fence trafilatura produced (markdown output) ...
    text = re.sub(
        rf"(?m)^[ \t]*(`{{3,}}|~{{3,}})[^\n]*\n[ \t]*({names})[ \t]*\n[ \t]*\1[ \t]*$",
        lambda m: fenced(m.group(2)),
        text,
    )
    # ... or bare, on its own line (html2txt) or inline.
    text = re.sub(rf"(?m)^[ \t]*({names})[ \t]*$", lambda m: fenced(m.group(1)), text)
    text = re.sub(rf"`?({names})`?", lambda m: "\n" + fenced(m.group(1)) + "\n", text)
    return text


def _base_mime(mime_type: str) -> str:
    """'text/html; charset=utf-8' -> 'text/html'"""
    return (mime_type or "").split(";", 1)[0].strip().lower()


def _json_ld(tree: HtmlElement | None) -> list[dict[str, Any]]:
    """
    Parses every <script type="application/ld+json"> block into a flat list of
    objects. Top-level arrays and "@graph" containers are unpacked. Malformed
    blocks are skipped.
    """
    if tree is None:
        return []

    items: list[dict[str, Any]] = []
    for script in tree.xpath("//script[@type]"):
        if script.get("type", "").split(";", 1)[0].strip().lower() != "application/ld+json":
            continue
        payload = _JSON_LD_WRAPPERS.sub("", script.text or "").strip()
        if not payload:
            continue
        try:
            data = json.loads(payload, strict=False)
        except ValueError:
            logger.debug("Skipping malformed JSON-LD block")
            continue
        items.extend(_flatten_json_ld(data))
    return items


def _flatten_json_ld(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for entry in data for item in _flatten_json_ld(entry)]
    if isinstance(data, dict):
        graph = data.get("@graph")
        if isinstance(graph, list):
            return _flatten_json_ld(graph)
        return [data]
    return []


def _opengraph(tree: HtmlElement | None) -> dict[str, str | list[str]]:
    """
    Collects OpenGraph / Twitter card <meta> tags. Repeated keys
    (e.g. several og:image) become lists, in document order.
    """
    if tree is None:
        return {}

    result: dict[str, str | list[str]] = {}
    for meta in tree.xpath("//meta[@content]"):
        key = (meta.get("property") or meta.get("name") or "").strip().lower()
        value = (meta.get("content") or "").strip()
        if not value or not key.startswith(_META_PREFIXES):
            continue
        if key not in result:
            result[key] = value
        elif isinstance(result[key], list):
            result[key].append(value)
        else:
            result[key] = [result[key], value]
    return result
