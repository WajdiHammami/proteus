# Proteus

Proteus extracts **C/C++ code snippets** from web pages, sitemaps, PDFs and local files, and turns each snippet into a record of a schema you define in YAML (for an SQL database or a vector store).

It is built for code fidelity and low LLM cost:

- snippets are found **without any LLM**: markdown code blocks, a line-by-line code detector for PDFs, and [tree-sitter](https://tree-sitter.github.io/) to confirm the code is C/C++;
- every field that can be computed (code, heading, includes, functions, URL, …) is filled for free;
- an LLM is only used for fields that need understanding (e.g. a description), runs **last**, and only after a **cost estimate** has been shown and confirmed.

## Quick start

```bash
python3 -m venv .env              # this project uses .env/ for its virtualenv
.env/bin/pip install -r requirements.txt
.env/bin/python -m cli            # interactive shell
```

```
proteus> ~/books/cpp_book.pdf          # a bare path or URL lists its snippets
proteus> /show 1                        # code + heading + context + includes/functions/classes
proteus> /extract ~/books/cpp_book.pdf  # records; LLM fields only after a y/N on the cost
proteus> /save records.json
```

The LLM step is optional: `/extract … --no-llm`, or a schema without LLM fields, never calls a model.

## The CLI

Run `python -m cli` for the shell (history, tab completion), or pass a command to run it once:

```bash
.env/bin/python -m cli snippets https://en.cppreference.com/w/cpp/utility/move
.env/bin/python -m cli --debug extract page.html     # --debug: no cost confirmation
```

| Command | What it does | LLM |
|---|---|---|
| `/fetch <src>` | scrape only: mime type, size, encoding, PDF pages / OCR flags | no |
| `/normalize <src> [--full]` | scrape + normalize: the clean text the extractor sees | no |
| `/snippets <src>` | scrape + normalize + find snippets (table) | no |
| `/show <#> [--full]` | one snippet: code, heading, context, analysis | no |
| `/sitemap <url> [--max N]` | the page URLs a sitemap leads to | no |
| `/extract <src…> [--no-llm] [--schema f]` | full pipeline → records; estimate → y/N → LLM filling | after confirmation |
| `/estimate <src…>` | what LLM filling would cost | no (local token count) |
| `/records`, `/record <#>`, `/rejected` | inspect the last extraction | no |
| `/save <file.json> [--force]` | write the records to JSON | no |
| `/schema [path] [--use]` | validate a schema, show its fields and generated dataclass | no |
| `/llm info`, `/llm test` | model and prices; one tiny real call (asks first) | `test` only |
| `/config`, `/set <key> <value>`, `/debug on\|off`, `/log <level>` | settings | — |

Every command accepts `--help`. Source types are guessed (`*.pdf` URL → PDF, `*.xml` or `…sitemap…` → sitemap, other URL → web page, path → file); force one with `--type url|file|pdf|sitemap`.

## Schemas

A schema describes one record per snippet. It is turned into a Python dataclass with the same fields and types.

```yaml
name: cpp_snippet
description: A C/C++ code snippet with the context that explains it.
fields:
  - name: code
    type: str              # str | int | float | bool | list
    vector: true           # embed this field in a vector store
  - name: url
    type: str
  - name: title
    type: str
    required: false
    source: document.title
  - name: description
    type: str
    required: false
    source: llm
    description: One sentence on what the snippet does.   # also the instruction given to the LLM
```

`source` says where a value comes from:

| Source | Values |
|---|---|
| `snippet.*` | `code`, `language`, `index`, `heading`, `context`, `line_count`, `includes`, `functions`, `classes`, `parse_score`, `language_declared` |
| `document.*` | `url`, `title`, `author`, `date`, `source_id`, `document_id` |
| `llm` | filled by the language model |

Without `source`, a field whose name is one of the keys above maps to it automatically; any other field is an LLM field. See [`schemas/cpp_snippet.yaml`](schemas/cpp_snippet.yaml).

Schemas are validated before use (`src/adapters/infrastructure/schema_sanitizer.py`): safe YAML only (no tags, no aliases, 256 KB max), names must be lowercase `snake_case` (they become Python code and SQL identifiers), no unknown keys, real booleans, a fixed list of types. All problems are reported at once.

## LLM configuration

The pipeline talks to models only through `LLMPort` (`src/ports/llm_port.py`). The current adapter is Azure OpenAI; switching provider means writing another `LLMPort` adapter.

Copy `azure.env.example` to `azure.env` (git-ignored) and fill it in:

```
AZURE_OPENAI_ENDPOINT=https://<resource>.services.ai.azure.com/
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini
AZURE_OPENAI_MODEL=gpt-4.1-mini
AZURE_OPENAI_PRICE_INPUT_PER_1M=0.562
AZURE_OPENAI_PRICE_CACHED_INPUT_PER_1M=0.141
AZURE_OPENAI_PRICE_OUTPUT_PER_1M=2.25
```

Environment variables override the file. Prices are used for the estimate shown before filling; without them the cost is shown as unknown.

## How it works

```
Source ─> ScraperDispatcher ─> NormalizerDispatcher ─> SnippetExtractor ─> RecordBuilder.prepare
          (web, file, PDF;      (HTML, PDF,             (code blocks,        (free fields)
           sitemaps expanded)    markdown, text, C/C++)  PDF code detector,          │
                                                         tree-sitter)                │
                      ... for every source, then once for all snippets:              ▼
                      FillingStage: estimate ─> confirm ─> LLMFieldFiller ─> RecordBuilder.finalize
                                                           (via LLMPort)     (validate → records)
```

The code follows a ports-and-adapters layout:

```
src/
  ports/          interfaces: scraper, sitemap, normalization, extraction, llm
  core/           pipeline, dispatchers, record builder, filling stage, schema → dataclass
  adapters/
    scrapers/     trafilatura (web/files), PDF (pypdf), sitemaps
    normalizers/  HTML, PDF, text/markdown/C++ files
    extractors/   snippet extractor, PDF code detector, tree-sitter analysis, LLM filler/prompt
    llm/          Azure OpenAI adapter + config
    infrastructure/  YAML loading, schema sanitizer
    models/       Source, RawDocument, NormalizedDocument, CodeSnippet, …
cli/              the interactive shell
schemas/          example schema
tests/
```

A few behaviours worth knowing:

- A URL that serves a PDF without a `.pdf` suffix is detected from its content and re-routed to the PDF scraper (downloaded once).
- PDF pages that need OCR are flagged (`text_layer` metadata) and their text is dropped; OCR itself is not done.
- Local files that aren't UTF-8 are decoded with a BOM / UTF-8 / cp1252 / latin-1 fallback.
- HTML code blocks keep their blank lines and language label (`language-cpp`, Sphinx, Prism, GeSHi, …); only `cpp`/`c` blocks, and unlabeled blocks that parse as C/C++, are kept.

## Tests

```bash
.env/bin/python -m pytest -q
```

Tests make no network calls (an LLM fake is in `tests/fakes/`). One test really calls Azure and is skipped unless enabled:

```bash
PROTEUS_LIVE_AZURE=1 .env/bin/python -m pytest tests/adapters/llm -k live -s
```

## Known limitations

- PDF: indentation inside code is approximate; tables and multi-column reading order are not handled; scanned pages are not OCR'd.
- Running headers that the PDF normalizer misses can show up as snippet headings.
- Indented (4-space) code blocks in markdown files are not detected.
- No storage yet: records are returned in memory or saved to JSON with `/save`.
- No crawl politeness for large sitemaps (robots.txt, retries) and no de-duplication across runs.
