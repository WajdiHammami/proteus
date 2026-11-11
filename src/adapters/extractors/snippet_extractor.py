import logging
import re

from src.adapters.extractors.candidate_block import CandidateBlock
from src.adapters.extractors.cpp_analysis import analyze
from src.adapters.extractors.plain_code_detector import find_code_blocks
from src.adapters.models.code_snippet import CodeSnippet
from src.adapters.models.normalized_document import NormalizedDocument, TextFormat
from src.adapters.normalizers.code_fence import normalize_language
from src.ports.extraction_port import SnippetExtractorPort


logger = logging.getLogger(__name__)

C_LANGUAGES = frozenset({"cpp", "c"})

_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*([^`\s]*)")
_ATX_HEADING = re.compile(r"^ {0,3}#{1,6}[ \t]+(.+?)[ \t#]*$")
_SETEXT_UNDERLINE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
# At least one of these must appear in an unlabeled block before we call it C/C++.
_C_SIGNAL = re.compile(r";|\{|\}|^\s*#\s*(?:include|define|if|pragma)\b", re.MULTILINE)
# Tokens that only exist in C++ (unlabeled blocks without any are labelled "c" only
# when they show C-specific markers, otherwise "cpp").
_CPP_ONLY = re.compile(
    r"::|\b(?:template|typename|class|namespace|public:|private:|protected:|virtual|override|constexpr|"
    r"nullptr|new|delete|auto|decltype|co_await|concept|requires|operator)\b|#include\s*<[a-z_]+>"
)
_C_ONLY = re.compile(r"#include\s*<\w+\.h>|\b(?:printf|malloc|free|scanf)\s*\(")


class HeuristicSnippetExtractor(SnippetExtractorPort):
    """
    Finds C/C++ snippets in markdown produced by the normalizers (fenced code blocks).

      - blocks labelled cpp/c are kept; blocks labelled with another language are dropped
      - unlabeled blocks are kept if tree-sitter parses them well and they show C/C++ signals
      - snippets shorter than `min_lines` / `min_chars` and exact duplicates are dropped
      - each snippet gets its nearest heading and the paragraph right before it

    PLAIN documents (PDFs, plain text) have no fences: code regions are found line by line
    (see plain_code_detector) and then go through the same checks as unlabeled blocks.

    TODO: indented (4-space) code blocks in markdown files are not picked up.
    """

    def __init__(
        self,
        min_lines: int = 2,
        min_chars: int = 20,
        min_parse_score: float = 0.6,
        max_context_chars: int = 500,
        max_gap: int = 3,
    ):
        """
        Args:
            min_lines: Snippets with fewer lines are dropped (one-liners rarely stand alone).
            min_chars: Snippets with fewer non-blank characters are dropped.
            min_parse_score: tree-sitter parse score an unlabeled block needs to count as C/C++.
            max_context_chars: The context paragraph is cut to its last N characters.
            max_gap: Plain text only: blank/unclear lines a code region may span.
        """
        self.min_lines = min_lines
        self.min_chars = min_chars
        self.min_parse_score = min_parse_score
        self.max_context_chars = max_context_chars
        self.max_gap = max_gap

    def extract(self, document: NormalizedDocument) -> list[CodeSnippet]:
        if document.format == TextFormat.MARKDOWN:
            blocks = _fenced_blocks(document.content)
        else:
            blocks = find_code_blocks(document.content, max_gap=self.max_gap)

        snippets: list[CodeSnippet] = []
        seen: set[str] = set()
        for block in blocks:
            snippet = self._to_snippet(block, index=len(snippets))
            if snippet is None:
                continue
            key = " ".join(snippet.code.split())
            if key in seen:
                continue
            seen.add(key)
            snippets.append(snippet)
        return snippets

    def _to_snippet(self, block: CandidateBlock, index: int) -> CodeSnippet | None:
        code = block.code.strip("\n")
        if len(code.splitlines()) < self.min_lines or len("".join(code.split())) < self.min_chars:
            return None

        label = normalize_language(block.label)
        if label is not None and label not in C_LANGUAGES:
            return None  # explicitly another language (bash, python, ...)

        analysis = analyze(code)
        if label is None:
            if analysis.parse_score < self.min_parse_score or not _C_SIGNAL.search(code):
                return None
            language = "c" if _C_ONLY.search(code) and not _CPP_ONLY.search(code) else "cpp"
        else:
            language = label

        context = block.context
        if len(context) > self.max_context_chars:
            context = "…" + context[-self.max_context_chars:].lstrip()

        return CodeSnippet(
            code=code,
            language=language,
            index=index,
            heading=block.heading,
            context=context,
            language_declared=label is not None,
            parse_score=analysis.parse_score,
            includes=analysis.includes,
            functions=analysis.functions,
            classes=analysis.classes,
        )


def _fenced_blocks(markdown: str) -> list[CandidateBlock]:
    """Walks the markdown once, tracking the current heading and the last prose paragraph."""
    lines = markdown.split("\n")
    blocks: list[CandidateBlock] = []
    heading: str | None = None
    paragraph: list[str] = []       # prose lines being collected
    last_paragraph = ""             # last finished paragraph since the previous block/heading

    def finish_paragraph():
        nonlocal paragraph, last_paragraph
        if paragraph:
            last_paragraph = " ".join(line.strip() for line in paragraph)
            paragraph = []

    i = 0
    while i < len(lines):
        line = lines[i]
        opening = _FENCE_OPEN.match(line)
        if opening:
            finish_paragraph()
            fence = opening.group(1)
            body = []
            i += 1
            while i < len(lines) and not _closes(lines[i], fence):
                body.append(lines[i])
                i += 1
            blocks.append(CandidateBlock("\n".join(body), opening.group(2) or None, heading, last_paragraph))
            last_paragraph = ""
            i += 1
            continue

        atx = _ATX_HEADING.match(line)
        if atx:
            finish_paragraph()
            heading, last_paragraph = atx.group(1).strip(), ""
        elif paragraph and _SETEXT_UNDERLINE.match(line) and len(paragraph) == 1:
            heading, paragraph, last_paragraph = paragraph[0].strip(), [], ""
        elif not line.strip():
            finish_paragraph()
        else:
            paragraph.append(line)
        i += 1
    return blocks


def _closes(line: str, fence: str) -> bool:
    stripped = line.strip()
    return len(stripped) >= len(fence) and set(stripped) == {fence[0]}
