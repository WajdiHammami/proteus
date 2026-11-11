import re


_BACKTICK_RUN = re.compile(r"`+")


def fence_code(code: str, language: str | None = None) -> str:
    """
    Wraps code in a markdown fence. The fence is longer than any backtick run inside
    the code, so the block can't be closed early. Returns "" for empty code.
    """
    if not code:
        return ""
    longest = max((len(run) for run in _BACKTICK_RUN.findall(code)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language or ''}\n{code}\n{fence}"
