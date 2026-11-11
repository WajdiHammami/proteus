"""
Proteus shell.

    python -m cli                         interactive shell (history, tab completion)
    python -m cli snippets book.pdf       run one command and exit
    python -m cli --debug extract URL     options: --debug, --schema, --env-file, --log
"""

import argparse
import glob
import logging
import os
import sys
import traceback
from pathlib import Path

from cli import commands, render
from cli.render import bold, cyan, dim, red, yellow
from cli.session import CLIError, Session, Settings
from src.adapters.infrastructure.schema_sanitizer import SchemaValidationError
from src.ports.llm_port import LLMError
from src.ports.normalization_port import UnsupportedDocumentError
from src.ports.scraper_port import UnsupportedSourceError

HISTORY_FILE = Path.home() / ".proteus_history"
PROMPT = "proteus> "


def run_line(session: Session, line: str) -> tuple[bool, bool]:
    """Runs one line; returns (keep_going, ok). Errors are printed, never raised."""
    try:
        return commands.execute(session, line), True
    except KeyboardInterrupt:
        session.out(yellow("Interrupted."))
    except (CLIError, LLMError, UnsupportedSourceError, UnsupportedDocumentError, SchemaValidationError) as error:
        session.out(red(f"Error: {error}"))
    except (OSError, ValueError, PermissionError) as error:
        session.out(red(f"Error: {type(error).__name__}: {error}"))
    except Exception as error:  # keep the shell alive; details at /log debug
        session.out(red(f"Unexpected error: {type(error).__name__}: {error}"))
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            session.out(dim(traceback.format_exc()))
        else:
            session.out(dim("(/log debug shows the traceback)"))
    return True, False


def interactive(session: Session) -> None:
    _setup_readline()
    session.out(_banner(session))
    while True:
        try:
            line = input(PROMPT if not render.Style.enabled else f"\001\033[1;36m\002proteus>\001\033[0m\002 ")
        except EOFError:
            session.out("")
            break
        except KeyboardInterrupt:
            session.out(dim("  (Ctrl-D or /exit to leave)"))
            continue
        keep_going, _ = run_line(session, line)
        if not keep_going:
            break
    _save_history()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cli", description="Proteus shell: scrape, normalize, extract C/C++ snippets.")
    parser.add_argument("--debug", action="store_true", help="LLM filling without cost confirmation")
    parser.add_argument("--schema", help="schema file")
    parser.add_argument("--env-file", help="LLM settings file (default: azure.env)")
    parser.add_argument("--log", default="warning", choices=["debug", "info", "warning", "error"])
    parser.add_argument("command", nargs=argparse.REMAINDER, help="a shell command to run once, e.g. snippets book.pdf")
    options = parser.parse_args(argv)

    logging.basicConfig(level=options.log.upper(), format="  %(levelname)s %(name)s: %(message)s")
    settings = Settings(debug=options.debug)
    if options.schema:
        settings.schema = options.schema
    if options.env_file:
        settings.env_file = options.env_file

    session = Session(settings, interactive=sys.stdin.isatty())
    if options.command:
        line = " ".join(_quote(word) for word in options.command)
        _, ok = run_line(session, line)
        return 0 if ok else 1
    interactive(session)
    return 0


# --------------------------------------------------------------------------- #

def _banner(session: Session) -> str:
    s = session.settings
    debug = yellow("on (no cost confirmation)") if s.debug else "off"
    return "\n".join([
        bold(cyan("Proteus")) + dim("  C/C++ snippet extraction"),
        dim(f"schema {s.schema} · debug {debug}" if not render.Style.enabled else f"schema {s.schema} · debug ") + (debug if render.Style.enabled else ""),
        dim("Type /help for commands. Paste a URL or a file path to see its snippets. Ctrl-D to quit."),
        "",
    ])


def _quote(word: str) -> str:
    import shlex
    return shlex.quote(word)


def _setup_readline() -> None:
    try:
        import readline
    except ImportError:  # e.g. Windows without pyreadline
        return
    try:
        readline.read_history_file(HISTORY_FILE)
    except (FileNotFoundError, OSError):
        pass
    readline.set_history_length(1000)
    readline.set_completer_delims(" \t\n")
    readline.set_completer(_complete)
    readline.parse_and_bind("tab: complete")


def _save_history() -> None:
    try:
        import readline
        readline.write_history_file(HISTORY_FILE)
    except Exception:
        pass


def _complete(text: str, state: int):
    import readline
    buffer = readline.get_line_buffer()
    first_word = not buffer.lstrip()[: len(buffer.lstrip()) - len(text)].strip()
    if first_word and (text.startswith("/") or not text):
        names = sorted({*commands.COMMANDS, *commands.ALIASES})
        options = [f"/{n} " for n in names if f"/{n}".startswith(text or "/")]
    else:
        pattern = os.path.expanduser(text) + "*"
        options = [p + ("/" if os.path.isdir(p) else " ") for p in sorted(glob.glob(pattern))]
    return options[state] if state < len(options) else None
