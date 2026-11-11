"""Terminal output helpers: colours (off when not a terminal or NO_COLOR is set), tables, previews."""

import os
import sys


class Style:
    enabled = sys.stdout.isatty() and "NO_COLOR" not in os.environ

    CODES = {
        "bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33",
        "blue": "34", "magenta": "35", "cyan": "36", "gray": "90",
    }

    @classmethod
    def apply(cls, text: str, *styles: str) -> str:
        if not cls.enabled or not styles:
            return text
        codes = ";".join(cls.CODES[s] for s in styles)
        return f"\033[{codes}m{text}\033[0m"


def bold(text): return Style.apply(str(text), "bold")
def dim(text): return Style.apply(str(text), "dim")
def red(text): return Style.apply(str(text), "red")
def green(text): return Style.apply(str(text), "green")
def yellow(text): return Style.apply(str(text), "yellow")
def cyan(text): return Style.apply(str(text), "cyan")


def table(headers: list[str], rows: list[list], max_width: int = 60) -> str:
    """Plain aligned table; cells longer than max_width are cut with '…'."""
    cells = [[_cell(value, max_width) for value in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in cells:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))
    header = "  ".join(bold(h.ljust(widths[i])) for i, h in enumerate(headers))
    rule = "  ".join(dim("─" * w) for w in widths)
    body = ["  ".join(value.ljust(widths[i]) for i, value in enumerate(row)) for row in cells]
    return "\n".join([header, rule, *body])


def key_values(pairs: list[tuple[str, object]]) -> str:
    width = max((len(k) for k, _ in pairs), default=0)
    return "\n".join(f"  {dim(k.ljust(width))}  {v}" for k, v in pairs)


def code(text: str, max_lines: int | None = None) -> str:
    """Code with a dim line-number gutter."""
    lines = text.split("\n")
    shown = lines if max_lines is None else lines[:max_lines]
    width = len(str(len(lines)))
    out = [f"{dim(str(n).rjust(width) + ' │')} {line}" for n, line in enumerate(shown, start=1)]
    if len(shown) < len(lines):
        out.append(dim(f"… {len(lines) - len(shown)} more line(s)"))
    return "\n".join(out)


def preview(text: str, max_lines: int) -> str:
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + "\n" + dim(f"… {len(lines) - max_lines} more line(s) (use --full)")


def _cell(value, max_width: int) -> str:
    text = "" if value is None else " ".join(str(value).split())
    return text if len(text) <= max_width else text[: max_width - 1] + "…"
