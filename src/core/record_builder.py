import dataclasses
import logging
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from src.adapters.models.code_snippet import CodeSnippet
from src.adapters.models.extracted_record import ExtractedRecord
from src.adapters.models.llm_fill import FillJob
from src.adapters.models.normalized_document import NormalizedDocument
from src.adapters.models.schema import Schema
from src.adapters.models.value_objects import FieldType, SchemaField
from src.core.schema_loader import build_record_class, validator_for
from src.ports.extraction_port import FieldFillerPort


logger = logging.getLogger(__name__)

LLM_SOURCE = "llm"

# Values computed for free, by `source` name.
SNIPPET_SOURCES: dict[str, Callable[[CodeSnippet], Any]] = {
    "code": lambda s: s.code,
    "language": lambda s: s.language,
    "index": lambda s: s.index,
    "heading": lambda s: s.heading,
    "context": lambda s: s.context or None,
    "line_count": lambda s: s.line_count,
    "includes": lambda s: s.includes,
    "functions": lambda s: s.functions,
    "classes": lambda s: s.classes,
    "parse_score": lambda s: s.parse_score,
    "language_declared": lambda s: s.language_declared,
}
DOCUMENT_SOURCES: dict[str, Callable[[NormalizedDocument], Any]] = {
    "url": lambda d: d.metadata.get("url") or d.metadata.get("path"),
    "title": lambda d: (d.metadata.get("page_metadata") or {}).get("title"),
    "author": lambda d: (d.metadata.get("page_metadata") or {}).get("author"),
    "date": lambda d: (d.metadata.get("page_metadata") or {}).get("date"),
    "source_id": lambda d: d.source_id,
    "document_id": lambda d: d.id,
}


@dataclass
class Rejected:
    snippet: CodeSnippet
    error: str


@dataclass
class PendingRecord:
    """A snippet's record before validation: free values filled, LLM fields still missing."""

    snippet: CodeSnippet
    document: NormalizedDocument
    values: dict[str, Any]
    missing: list[SchemaField]

    def job(self) -> FillJob:
        return FillJob(self.snippet, list(self.missing), dict(self.values), self.document)


@dataclass
class ExtractionResult:
    records: list[ExtractedRecord] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)   # snippets that didn't make a valid record
    fill: Any = None                                         # FillReport of the filling stage, if it ran


class RecordBuilder:
    """
    Turns each snippet into one record of the user's schema, in two steps so that the
    (paid) LLM filling can run last, for all documents at once, after confirmation:

      prepare():  free fields. Each field's `source` ("snippet.code", "document.url", ...) is
                  resolved; a field without `source` maps by name when it matches a known value
                  ("code", "url", ...), otherwise it is an LLM field, left in `missing`.
      (filling:   see core.filling.FillingStage, which uses fill_pending())
      finalize(): values validated against the dataclass built from the schema; records that
                  fail (e.g. a required field nobody could fill) are returned in `rejected`.
    """

    def __init__(self, schema: Schema):
        self.schema = schema
        self.record_class = build_record_class(schema)
        self.validator = validator_for(self.record_class)
        self.resolvers = {f.name: _resolver(f) for f in schema.fields}   # None = LLM field
        self.llm_fields = [f for f in schema.fields if self.resolvers[f.name] is None]

    def prepare(self, document: NormalizedDocument, snippets: Sequence[CodeSnippet]) -> list[PendingRecord]:
        pending = []
        for snippet in snippets:
            values: dict[str, Any] = {}
            for schema_field in self.schema.fields:
                resolver = self.resolvers[schema_field.name]
                if resolver is not None:
                    values[schema_field.name] = _coerce(resolver(snippet, document), schema_field)
            missing = [f for f in self.llm_fields if values.get(f.name) is None]
            pending.append(PendingRecord(snippet, document, values, missing))
        return pending

    def finalize(self, pending: Sequence[PendingRecord]) -> ExtractionResult:
        result = ExtractionResult()
        for item in pending:
            # Optional fields nobody filled are simply None; required ones fail validation.
            present = {name: value for name, value in item.values.items() if value is not None}
            try:
                instance = self.validator.validate_python(present)
            except ValidationError as error:
                message = "; ".join(
                    f"{'.'.join(str(p) for p in e['loc']) or '?'}: {e['msg']}" for e in error.errors()
                )
                logger.warning("Snippet %d of %s rejected: %s", item.snippet.index, item.document.id, message)
                result.rejected.append(Rejected(item.snippet, message))
                continue
            result.records.append(ExtractedRecord(
                id=str(uuid.uuid4()),
                document_id=item.document.id,
                schema_id=self.schema.id,
                data=dataclasses.asdict(instance),
            ))
        return result

    def build(self, document: NormalizedDocument, snippets: Sequence[CodeSnippet]) -> ExtractionResult:
        """Free fields only (no filling): prepare + finalize."""
        return self.finalize(self.prepare(document, snippets))


def fill_pending(pending: Sequence[PendingRecord], fillers: Sequence[FieldFillerPort]) -> None:
    """
    Runs the fillers in order (cheapest first) on every pending record; each filler only
    sees the fields still missing. Updates the records in place.
    """
    for item in pending:
        for filler in fillers:
            if not item.missing:
                break
            filled = filler.fill(item.snippet, list(item.missing), dict(item.values), item.document)
            for schema_field in item.missing:
                value = filled.get(schema_field.name)
                if value is not None:
                    item.values[schema_field.name] = _coerce(value, schema_field)
            item.missing = [f for f in item.missing if item.values.get(f.name) is None]


def _resolver(schema_field: SchemaField) -> Callable[[CodeSnippet, NormalizedDocument], Any] | None:
    source = (schema_field.source or "").strip()
    if source == LLM_SOURCE:
        return None
    if not source:
        if schema_field.name in SNIPPET_SOURCES:
            source = f"snippet.{schema_field.name}"
        elif schema_field.name in DOCUMENT_SOURCES:
            source = f"document.{schema_field.name}"
        else:
            return None

    scope, _, key = source.partition(".")
    if scope == "snippet" and key in SNIPPET_SOURCES:
        getter = SNIPPET_SOURCES[key]
        return lambda snippet, document: getter(snippet)
    if scope == "document" and key in DOCUMENT_SOURCES:
        getter = DOCUMENT_SOURCES[key]
        return lambda snippet, document: getter(document)
    known = [f"snippet.{k}" for k in SNIPPET_SOURCES] + [f"document.{k}" for k in DOCUMENT_SOURCES] + [LLM_SOURCE]
    raise ValueError(f"Field '{schema_field.name}' has unknown source '{source}'. Known: {', '.join(known)}")


def _coerce(value: Any, schema_field: SchemaField) -> Any:
    """Small shape fixes between what we compute and what the schema asks for."""
    if value is None:
        return None
    if schema_field.type == FieldType.STR and isinstance(value, list):
        return ", ".join(str(v) for v in value) or None
    if schema_field.type == FieldType.LIST and isinstance(value, str):
        return [value]
    return value
