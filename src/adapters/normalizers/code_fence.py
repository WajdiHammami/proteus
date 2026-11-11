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


# Code fence / class labels that mean "no particular language".
_NOT_LANGUAGES = frozenset({"none", "text", "plain", "plaintext", "txt", "nohighlight", "default", "code"})
_LANGUAGE_ALIASES = {
    "c++": "cpp", "cxx": "cpp", "cc": "cpp", "hpp": "cpp", "h++": "cpp", "cplusplus": "cpp",
    "objective-c++": "objcpp",
}
_LANGUAGE_NAME = re.compile(r"[a-z0-9_+#-]{1,30}")


def normalize_language(value: str | None) -> str | None:
    """'C++' -> 'cpp', 'text' -> None. Returns None for anything that isn't a language name."""
    value = (value or "").strip().lower()
    if not value or value in _NOT_LANGUAGES or not _LANGUAGE_NAME.fullmatch(value):
        return None
    return _LANGUAGE_ALIASES.get(value, value)
