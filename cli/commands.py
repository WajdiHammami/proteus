"""
CLI commands. Each command is a function taking (session, args) registered with @command.
Arguments are parsed with argparse, so every command gets `--help` for free.
"""

import argparse
import dataclasses
import json
import logging
import math
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cli import render
from cli.render import bold, cyan, dim, green, red, yellow
from cli.session import CLIError, Session, SOURCE_TYPES
from src.adapters.cli.console_confirm import console_confirm, format_estimates
from src.adapters.extractors.llm_filler import LLMFieldFiller
from src.adapters.infrastructure.schema_sanitizer import SchemaValidationError
from src.adapters.infrastructure.yaml_loader import load_schema_from_yaml
from src.adapters.models.source import SourceType
from src.core.filling import FillingStage
from src.core.pipeline import ExtractionPipeline
from src.core.record_builder import RecordBuilder
from src.ports.llm_port import LLMError, LLMRequest


@dataclass
class Command:
    name: str
    func: Callable
    summary: str
    parser: argparse.ArgumentParser
    aliases: tuple[str, ...] = ()


COMMANDS: dict[str, Command] = {}
ALIASES: dict[str, str] = {}


class _Parser(argparse.ArgumentParser):
    """argparse that raises instead of exiting the shell."""

    def error(self, message):
        raise CLIError(f"{message}\n{self.format_usage().strip()}")

    def exit(self, status=0, message=None):
        if message:
            raise CLIError(message.strip())
        raise _HelpShown()


class _HelpShown(Exception):
    pass


def command(name: str, summary: str, aliases: tuple[str, ...] = ()):
    def register(func):
        parser = _Parser(prog=f"/{name}", description=summary, add_help=True)
        func(None, parser)  # the function declares its arguments when session is None
        COMMANDS[name] = Command(name, func, summary, parser, aliases)
        for alias in aliases:
            ALIASES[alias] = name
        return func
    return register


def execute(session: Session, line: str) -> bool:
    """Runs one input line. Returns False when the shell should exit."""
    line = line.strip()
    if not line or line.startswith("#"):
        return True
    try:
        words = shlex.split(line)
    except ValueError as error:
        raise CLIError(f"Could not parse the line: {error}")

    name = words[0].lstrip("/").lower()
    name = ALIASES.get(name, name)
    if name not in COMMANDS:
        if _looks_like_source(words[0]):
            name, words = "snippets", ["snippets", *words]   # bare URL/path -> /snippets
        else:
            raise CLIError(f"Unknown command '{words[0]}'. Type /help.")
    cmd = COMMANDS[name]
    try:
        args = cmd.parser.parse_args(words[1:])
    except _HelpShown:
        session.out(cmd.parser.format_help().rstrip())
        return True
    return cmd.func(session, args) is not False


def _looks_like_source(word: str) -> bool:
    """URLs and anything path-like ('/tmp/a.pdf', './x', 'book.pdf'); command names have neither '/' nor '.' inside."""
    if word.startswith(("http://", "https://", "~", "./", "../")):
        return True
    return "/" in word.lstrip("/") or "." in word or Path(word).exists()


def _source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source", help="URL or local file")
    parser.add_argument("--type", choices=list(SOURCE_TYPES), help="force the source type (default: guessed)")


# --------------------------------------------------------------------------- #
# Help, settings
# --------------------------------------------------------------------------- #

@command("help", "List commands, or show one command's options", aliases=("h", "?"))
def cmd_help(session, args):
    if session is None:
        args.add_argument("command", nargs="?")
        return
    if args.command:
        name = ALIASES.get(args.command.lstrip("/"), args.command.lstrip("/"))
        if name not in COMMANDS:
            raise CLIError(f"Unknown command '{args.command}'")
        session.out(COMMANDS[name].parser.format_help().rstrip())
        return
    groups = [
        ("Pipeline stages", ["fetch", "normalize", "snippets", "show", "sitemap"]),
        ("Extraction", ["extract", "estimate", "records", "record", "rejected", "save"]),
        ("Schema & LLM", ["schema", "llm"]),
        ("Session", ["config", "set", "debug", "log", "clear", "help", "exit"]),
    ]
    lines = [bold("Commands") + dim("  (type /<command> --help for options; a bare URL or path runs /snippets)")]
    for title, names in groups:
        lines.append("")
        lines.append(cyan(title))
        for name in names:
            cmd = COMMANDS[name]
            lines.append(f"  /{name.ljust(10)} {cmd.summary}")
    session.out("\n".join(lines))


@command("exit", "Leave the shell", aliases=("quit", "q"))
def cmd_exit(session, args):
    if session is None:
        return
    return False


@command("clear", "Forget the results of the last run")
def cmd_clear(session, args):
    if session is None:
        return
    session.state = type(session.state)()
    session.out(dim("State cleared."))


@command("config", "Show the current settings and LLM configuration")
def cmd_config(session, args):
    if session is None:
        return
    s = session.settings
    pairs = [
        ("schema", s.schema), ("env file", s.env_file),
        ("debug", ("on" if s.debug else "off") + "  " + dim("(on = fill without cost confirmation)")),
        ("max_urls", f"{s.max_urls}  " + dim("(pages taken from a sitemap)")),
        ("lines", f"{s.lines}  " + dim("(preview length)")),
    ]
    try:
        llm = session.llm
        pricing = llm.pricing
        config = getattr(llm, "config", None)
        pairs += [
            ("llm", f"{llm.name} via {type(llm).__name__}"),
            ("endpoint", getattr(config, "endpoint", "-")),
            ("api key", "set" if getattr(config, "api_key", None) else "missing"),
            ("prices / 1M", f"input {pricing.input_per_million}, cached {pricing.cached_input_per_million}, "
                            f"output {pricing.output_per_million} {pricing.currency}" if pricing else "not configured"),
        ]
    except LLMError as error:
        pairs.append(("llm", red(f"not configured: {error}")))
    session.out(bold("Settings") + "\n" + render.key_values(pairs))


@command("set", "Change a setting: schema, env_file, max_urls, lines, debug")
def cmd_set(session, args):
    if session is None:
        args.add_argument("key", choices=["schema", "env_file", "max_urls", "lines", "debug"])
        args.add_argument("value")
        return
    s = session.settings
    if args.key in ("max_urls", "lines"):
        try:
            value = int(args.value)
        except ValueError:
            raise CLIError(f"{args.key} must be a number")
        if value < 1:
            raise CLIError(f"{args.key} must be at least 1")
        setattr(s, args.key, value)
    elif args.key == "debug":
        s.debug = _on_off(args.value)
    elif args.key == "schema":
        _load_schema(args.value)    # validate before switching
        s.schema = args.value
    else:
        s.env_file = args.value
        session.reset_llm()
    session.out(green(f"{args.key} = {getattr(s, args.key)}"))


@command("debug", "Turn debug mode on/off (on = LLM filling without cost confirmation)")
def cmd_debug(session, args):
    if session is None:
        args.add_argument("state", nargs="?", choices=["on", "off"])
        return
    if args.state:
        session.settings.debug = args.state == "on"
    state = "on" if session.settings.debug else "off"
    note = yellow("  LLM filling will run WITHOUT asking for cost confirmation") if session.settings.debug else ""
    session.out(f"debug {bold(state)}{note}")


@command("log", "Set the log level shown in the shell (warning by default)")
def cmd_log(session, args):
    if session is None:
        args.add_argument("level", choices=["debug", "info", "warning", "error"])
        return
    logging.getLogger().setLevel(args.level.upper())
    session.out(f"log level {bold(args.level)}")


# --------------------------------------------------------------------------- #
# Pipeline stages, one at a time
# --------------------------------------------------------------------------- #

@command("fetch", "Scrape a source and show the raw document")
def cmd_fetch(session, args):
    if session is None:
        _source_args(args)
        return
    source = _single_source(session, args)
    raw = session.scraper.fetch(source)
    session.remember(source=source, raw=raw, normalized=None, snippets=[])
    meta = raw.metadata
    pairs = [
        ("source", f"{source.url}  {dim(source.type.value)}"),
        ("scraper", meta.get("scraper")),
        ("mime type", raw.mime_type),
        ("size", f"{len(raw.content):,} chars"),
    ]
    if "rerouted_from" in meta:
        pairs.append(("re-routed", f"from {meta['rerouted_from']} (the URL serves a PDF)"))
    if "encoding" in meta:
        pairs.append(("encoding", meta["encoding"]))
    if "page_count" in meta:
        layer = meta.get("text_layer", {})
        pairs.append(("pages", meta["page_count"]))
        if layer.get("needs_ocr"):
            pairs.append(("OCR needed", yellow(f"pages {layer.get('ocr_pages')} (their text is dropped)")))
    session.out(render.key_values(pairs))


@command("normalize", "Scrape + normalize a source and show the clean text")
def cmd_normalize(session, args):
    if session is None:
        _source_args(args)
        args.add_argument("--full", action="store_true", help="print everything, not just a preview")
        return
    source = _single_source(session, args)
    raw = session.scraper.fetch(source)
    normalized = session.normalizer.normalize(raw)
    session.remember(source=source, raw=raw, normalized=normalized, snippets=[])
    meta = normalized.metadata
    title = (meta.get("page_metadata") or {}).get("title")
    pairs = [
        ("normalizer", meta.get("normalizer")),
        ("format", normalized.format.value),
        ("title", title or dim("-")),
        ("size", f"{len(raw.content):,} → {len(normalized.content):,} chars, "
                 f"{normalized.word_count():,} words, ≈{math.ceil(len(normalized.content) / 4):,} tokens"),
    ]
    if "extraction_mode" in meta:
        pairs.append(("extraction", meta["extraction_mode"]))
    if meta.get("removed_boilerplate"):
        pairs.append(("boilerplate", f"{len(meta['removed_boilerplate'])} line(s) removed"))
    session.out(render.key_values(pairs))
    session.out(dim("─" * 60))
    session.out(normalized.content if args.full else render.preview(normalized.content, session.settings.lines))


@command("snippets", "Scrape + normalize + find the C/C++ snippets of a source", aliases=("s",))
def cmd_snippets(session, args):
    if session is None:
        _source_args(args)
        return
    source = _single_source(session, args)
    raw = session.scraper.fetch(source)
    normalized = session.normalizer.normalize(raw)
    snippets = session.extractor.extract(normalized)
    session.remember(source=source, raw=raw, normalized=normalized, snippets=snippets)
    if not snippets:
        session.out(yellow("No C/C++ snippet found.") + dim(f"  ({normalized.format.value}, "
                    f"{normalized.word_count():,} words; try /normalize to see the text)"))
        return
    rows = [
        [s.index, s.language + ("" if s.language_declared else "*"), s.line_count, f"{s.parse_score:.2f}",
         s.heading or "", ", ".join(dict.fromkeys(s.classes + s.functions))]
        for s in snippets
    ]
    session.out(render.table(["#", "lang", "lines", "parse", "heading", "defines"], rows, max_width=40))
    session.out(dim(f"{len(snippets)} snippet(s). * = language guessed, not labelled. /show <#> to see one."))


@command("show", "Show one snippet of the last run (code, context, analysis)")
def cmd_show(session, args):
    if session is None:
        args.add_argument("index", type=int)
        args.add_argument("--full", action="store_true", help="print all lines")
        return
    snippets = session.state.snippets
    if not snippets:
        raise CLIError("No snippets yet: run /snippets <source> first.")
    if not 0 <= args.index < len(snippets):
        raise CLIError(f"Index out of range: 0..{len(snippets) - 1}")
    s = snippets[args.index]
    pairs = [
        ("language", f"{s.language} " + dim("(labelled)" if s.language_declared else "(guessed)")),
        ("parse score", f"{s.parse_score:.2f}"),
        ("heading", s.heading or dim("-")),
        ("context", s.context or dim("-")),
        ("includes", ", ".join(s.includes) or dim("-")),
        ("functions", ", ".join(s.functions) or dim("-")),
        ("classes", ", ".join(s.classes) or dim("-")),
    ]
    session.out(bold(f"Snippet {s.index}") + "\n" + render.key_values(pairs))
    session.out(render.code(s.code, None if args.full else session.settings.lines))


@command("sitemap", "List the page URLs a sitemap (or site) leads to")
def cmd_sitemap(session, args):
    if session is None:
        args.add_argument("url")
        args.add_argument("--max", type=int, help="cap on URLs (default: the max_urls setting)")
        return
    source = session.make_source(args.url, "sitemap")
    adapter = session.sitemap()
    if args.max:
        adapter.max_urls = args.max
    children = adapter.expand(source)
    rows = [[i, c.type.value, c.url] for i, c in enumerate(children)]
    session.out(render.table(["#", "type", "url"], rows, max_width=100))
    session.out(dim(f"{len(children)} URL(s) (capped at {adapter.max_urls})."))


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

@command("extract", "Full pipeline: snippets → records, then (after confirmation) LLM filling", aliases=("x",))
def cmd_extract(session, args):
    if session is None:
        args.add_argument("sources", nargs="+", help="URLs, files or sitemaps")
        args.add_argument("--type", choices=list(SOURCE_TYPES), help="force the type of every source")
        args.add_argument("--schema", help="schema file (default: the schema setting)")
        args.add_argument("--no-llm", action="store_true", help="free fields only, never call the LLM")
        return
    schema_path = args.schema or session.settings.schema
    _load_schema(schema_path)
    sources = [session.make_source(target, args.type) for target in args.sources]
    fillers = [] if args.no_llm else _fillers_or_warn(session)

    pipeline = ExtractionPipeline(
        session.scraper, session.normalizer, session.extractor, schema_path,
        fillers=fillers, sitemap=session.sitemap(), confirm=_confirm(session), debug=session.settings.debug,
    )
    result = pipeline.run(sources)
    session.remember(result=result)
    _print_records(session, result)
    _print_fill(session, result.fill)


@command("estimate", "What LLM filling would cost for these sources (no LLM call)")
def cmd_estimate(session, args):
    if session is None:
        args.add_argument("sources", nargs="+")
        args.add_argument("--type", choices=list(SOURCE_TYPES))
        args.add_argument("--schema")
        return
    schema_path = args.schema or session.settings.schema
    builder = RecordBuilder(_load_schema(schema_path))
    pending = []
    for target in args.sources:
        source = session.make_source(target, args.type)
        if source.type == SourceType.SITEMAP:
            children = session.sitemap().expand(source)
        else:
            children = [source]
        for child in children:
            normalized = session.normalizer.normalize(session.scraper.fetch(child))
            pending.extend(builder.prepare(normalized, session.extractor.extract(normalized)))

    todo = [p for p in pending if p.missing]
    session.out(f"{len(pending)} snippet(s), {len(todo)} with LLM fields to fill.")
    if not todo:
        return
    filler = LLMFieldFiller(session.llm)
    session.out(format_estimates([filler.estimate([p.job() for p in todo])]))


@command("records", "List the records of the last /extract")
def cmd_records(session, args):
    if session is None:
        return
    _print_records(session, _last_result(session))


@command("record", "Show one record of the last /extract as JSON")
def cmd_record(session, args):
    if session is None:
        args.add_argument("index", type=int)
        return
    records = _last_result(session).records
    if not 0 <= args.index < len(records):
        raise CLIError(f"Index out of range: 0..{len(records) - 1}" if records else "No records.")
    session.out(json.dumps(records[args.index].data, indent=2, ensure_ascii=False))


@command("rejected", "List the snippets of the last /extract that didn't make a valid record")
def cmd_rejected(session, args):
    if session is None:
        return
    rejected = _last_result(session).rejected
    if not rejected:
        session.out(green("No rejected snippet."))
        return
    rows = [[r.snippet.index, r.snippet.heading or "", r.error] for r in rejected]
    session.out(render.table(["#", "heading", "why"], rows, max_width=70))


@command("save", "Save the records of the last /extract to a JSON file")
def cmd_save(session, args):
    if session is None:
        args.add_argument("path")
        args.add_argument("--force", action="store_true", help="overwrite an existing file")
        return
    result = _last_result(session)
    path = Path(args.path).expanduser()
    if path.exists() and not args.force:
        raise CLIError(f"{path} exists (use --force to overwrite)")
    payload = [{"id": r.id, "document_id": r.document_id, "schema_id": r.schema_id, "data": r.data} for r in result.records]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    session.out(green(f"Saved {len(payload)} record(s) to {path}"))


# --------------------------------------------------------------------------- #
# Schema & LLM
# --------------------------------------------------------------------------- #

@command("schema", "Validate a schema and show its fields (and the generated dataclass)")
def cmd_schema(session, args):
    if session is None:
        args.add_argument("path", nargs="?", help="schema file (default: the schema setting)")
        args.add_argument("--use", action="store_true", help="also make it the current schema")
        return
    path = args.path or session.settings.schema
    schema = _load_schema(path)
    builder = RecordBuilder(schema)
    rows = []
    for f in schema.fields:
        free = builder.resolvers[f.name] is not None
        origin = f.source or (f"auto ({'free' if free else 'llm'})")
        rows.append([f.name, f.type.value, "yes" if f.required else "no", "yes" if f.vector else "",
                     green("free") if free else yellow("llm"), origin])
    session.out(bold(f"{schema.name}") + dim(f"  {path}") + (f"\n{schema.description}" if schema.description else ""))
    session.out(render.table(["field", "type", "required", "vector", "filled by", "source"], rows))
    cls = builder.record_class
    signature = ", ".join(f"{f.name}: {getattr(f.type, '__name__', str(f.type))}" for f in dataclasses.fields(cls))
    session.out(dim(f"dataclass {cls.__name__}({signature})"))
    if args.use:
        session.settings.schema = path
        session.out(green(f"Current schema: {path}"))


@command("llm", "LLM tools: '/llm info' or '/llm test' (one tiny real call)")
def cmd_llm(session, args):
    if session is None:
        args.add_argument("action", choices=["info", "test"])
        return
    llm = session.llm
    if args.action == "info":
        pricing = llm.pricing
        session.out(render.key_values([
            ("model", llm.name), ("adapter", type(llm).__name__),
            ("prices / 1M", f"in {pricing.input_per_million}, out {pricing.output_per_million} {pricing.currency}"
             if pricing else "not configured"),
        ]))
        return

    request = LLMRequest(
        system="Describe the C++ snippet in one short sentence.",
        user="```cpp\nstd::unique_ptr<int> p = std::make_unique<int>(3);\n```",
        json_schema={"type": "object", "properties": {"description": {"type": ["string", "null"]}},
                     "required": ["description"], "additionalProperties": False},
        schema_name="cli_test", max_output_tokens=60,
    )
    if not session.settings.debug:
        tokens = llm.count_tokens(request.system) + llm.count_tokens(request.user) + 60
        cost = f" (~{tokens} tokens, under $0.001)" if llm.pricing else ""
        if not _ask(session, f"This makes one real call to {llm.name}{cost}. Continue? [y/N] "):
            session.out(dim("Cancelled."))
            return
    response = llm.generate_json(request)
    cost = llm.pricing.cost(response.usage) if llm.pricing else None
    session.out(green("OK ") + json.dumps(response.data, ensure_ascii=False))
    session.out(dim(f"{response.model}: {response.usage.input_tokens} in / {response.usage.output_tokens} out"
                    + (f", ${cost:.6f}" if cost is not None else "")))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _single_source(session: Session, args):
    source = session.make_source(args.source, args.type)
    if source.type == SourceType.SITEMAP:
        raise CLIError("That's a sitemap: use /sitemap to list its pages, or /extract to process them.")
    return source


def _load_schema(path: str):
    try:
        return load_schema_from_yaml(path)
    except SchemaValidationError as error:
        raise CLIError(str(error))
    except FileNotFoundError:
        raise CLIError(f"No such schema file: {path}")


def _fillers_or_warn(session: Session) -> list:
    try:
        return [LLMFieldFiller(session.llm)]
    except LLMError as error:
        session.out(yellow(f"LLM not available ({error}); extracting free fields only."))
        return []


def _confirm(session: Session):
    def confirm(estimates):
        return console_confirm(estimates, input_fn=session.input_fn, output=session.out, interactive=session.interactive)
    return confirm


def _ask(session: Session, prompt: str) -> bool:
    if session.interactive is False:
        return False
    try:
        return session.input_fn(prompt).strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def _last_result(session: Session):
    if session.state.result is None:
        raise CLIError("No extraction yet: run /extract <source> first.")
    return session.state.result


def _print_records(session: Session, result) -> None:
    if not result.records:
        session.out(yellow("No records.") + (dim(f"  {len(result.rejected)} rejected: /rejected") if result.rejected else ""))
        return
    data = [r.data for r in result.records]
    keys = [k for k in ("language", "heading", "url", "description") if k in data[0]]
    rows = [[i, *[d.get(k) for k in keys], (d.get("code") or "").split("\n", 1)[0]] for i, d in enumerate(data)]
    session.out(render.table(["#", *keys, "first line"], rows, max_width=36))
    extra = f", {len(result.rejected)} rejected (/rejected)" if result.rejected else ""
    session.out(dim(f"{len(result.records)} record(s){extra}. /record <#> for JSON, /save <file> to keep them."))


def _print_fill(session: Session, report) -> None:
    if report is None:
        return
    if report.ran:
        spent = "unknown" if report.spent is None else f"${report.spent:.6f}"
        session.out(green(f"LLM filling done: {report.jobs} snippet(s), spent {spent}."))
    elif report.skipped_reason not in ("no filler configured", "nothing to fill"):
        session.out(yellow(f"LLM filling skipped: {report.skipped_reason}. LLM fields are empty.")
                    + dim("  (/debug on fills without asking)" if report.skipped_reason == "declined" else ""))


def _on_off(value: str) -> bool:
    value = value.lower()
    if value in ("on", "true", "yes", "1"):
        return True
    if value in ("off", "false", "no", "0"):
        return False
    raise CLIError("Use on/off")
