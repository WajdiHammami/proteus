from dataclasses import dataclass, field
from functools import lru_cache

import tree_sitter_cpp
from tree_sitter import Language, Node, Parser


# A MISSING node (e.g. an inserted ';') costs as much as this many erroneous bytes.
_MISSING_NODE_COST = 10
_CLASS_NODES = {"class_specifier", "struct_specifier", "union_specifier", "enum_specifier"}
_NAME_NODES = {
    "identifier", "field_identifier", "qualified_identifier", "operator_name",
    "destructor_name", "template_function", "type_identifier",
}


@dataclass
class CppAnalysis:
    parse_score: float                                   # 0..1, share of the code that parses cleanly
    includes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def _parser() -> Parser:
    return Parser(Language(tree_sitter_cpp.language()))


def analyze(code: str) -> CppAnalysis:
    """
    Parses the code with tree-sitter-cpp. The C++ grammar also covers almost all C.

    parse_score = 1 - (bytes inside ERROR nodes + cost of MISSING nodes) / non-blank bytes,
    clamped to 0..1. Prose and shell commands score low; snippets with one typo still score high.
    """
    data = code.encode("utf-8")
    if not data.strip():
        return CppAnalysis(parse_score=0.0)

    root = _parser().parse(data).root_node
    error_bytes, missing = _error_cost(root)
    significant = max(1, len(data) - data.count(b" ") - data.count(b"\n") - data.count(b"\t"))
    score = 1.0 - (error_bytes + missing * _MISSING_NODE_COST) / significant
    score = round(min(1.0, max(0.0, score)), 3)

    includes, functions, classes = [], [], []
    _collect(root, data, includes, functions, classes)
    return CppAnalysis(
        parse_score=score,
        includes=_unique(includes),
        functions=_unique(functions),
        classes=_unique(classes),
    )


def _error_cost(root: Node) -> tuple[int, int]:
    error_bytes, missing = 0, 0
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_missing:
            missing += 1
            continue
        if node.type == "ERROR":
            error_bytes += node.end_byte - node.start_byte
            continue  # don't count nested errors twice
        if node.has_error:
            stack.extend(node.children)
    return error_bytes, missing


def _collect(root: Node, data: bytes, includes: list, functions: list, classes: list) -> None:
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "preproc_include":
            path = node.child_by_field_name("path")
            if path is not None:
                includes.append(_text(path, data))
        elif node.type == "function_definition":
            name = _function_name(node, data)
            if name:
                functions.append(name)
        elif node.type in _CLASS_NODES and node.child_by_field_name("body") is not None:
            name = node.child_by_field_name("name")
            if name is not None:
                classes.append(_text(name, data))
        stack.extend(reversed(node.children))


def _function_name(definition: Node, data: bytes) -> str | None:
    """Follows declarator fields (pointer/reference wrappers) down to the function's name."""
    node = definition.child_by_field_name("declarator")
    while node is not None:
        if node.type == "function_declarator":
            name = node.child_by_field_name("declarator")
            return _text(name, data) if name is not None and name.type in _NAME_NODES else None
        inner = node.child_by_field_name("declarator")
        if inner is None:  # reference_declarator has no field name for its child
            inner = next((child for child in node.named_children if "declarator" in child.type), None)
        node = inner
    return None


def _text(node: Node, data: bytes) -> str:
    return " ".join(data[node.start_byte:node.end_byte].decode("utf-8", "replace").split())


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
