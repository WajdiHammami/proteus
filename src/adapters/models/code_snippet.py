from dataclasses import dataclass, field


@dataclass
class CodeSnippet:
    """One piece of code found in a NormalizedDocument, with what we know about it for free."""

    code: str
    language: str                       # "cpp" or "c"
    index: int                          # position among the document's kept snippets (0-based)
    heading: str | None = None          # nearest heading above the snippet
    context: str = ""                   # prose right before the snippet (what usually explains it)
    language_declared: bool = False     # True if the page labelled it, False if we classified it
    parse_score: float = 1.0            # share of the code tree-sitter parses without errors (0..1)
    includes: list[str] = field(default_factory=list)    # "<vector>", "\"util.h\""
    functions: list[str] = field(default_factory=list)   # names of functions defined
    classes: list[str] = field(default_factory=list)     # names of classes/structs/unions/enums defined

    @property
    def line_count(self) -> int:
        return len(self.code.splitlines())
