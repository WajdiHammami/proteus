from dataclasses import dataclass


@dataclass
class CandidateBlock:
    """A block that may be code, before classification (from a markdown fence or a plain-text region)."""

    code: str
    label: str | None        # language the document declared (fence info string), if any
    heading: str | None
    context: str
