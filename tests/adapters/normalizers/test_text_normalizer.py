import copy
import json

import pytest

from src.adapters.models.normalized_document import TextFormat
from src.adapters.models.raw_document import RawDocument
from src.adapters.models.source import SourceType
from src.adapters.normalizers.text_normalizer import TextNormalizer, _split_code
from src.adapters.scrapers.trafilatura_adapter import TrafilaturaScraperAdapter
from src.ports.normalization_port import UnsupportedDocumentError


def raw(content: str, mime: str, path: str | None = "/notes/file.txt", **metadata) -> RawDocument:
    meta = {"path": path} if path else {}
    meta.update(metadata)
    return RawDocument(id="raw-1", source_id="src-1", content=content, mime_type=mime, metadata=meta)


def md(content: str, **metadata):
    return TextNormalizer().normalize(raw(content, "text/markdown", path="/docs/README.md", **metadata))


def cpp(content: str, mime: str = "text/x-c++src", path: str | None = "/src/stack.cpp"):
    return TextNormalizer().normalize(raw(content, mime, path=path))


# --------------------------------------------------------------------------- #
# supports() / basics
# --------------------------------------------------------------------------- #

class TestSupports:
    @pytest.mark.parametrize("mime", [
        "text/plain", "text/markdown", "text/x-markdown", "TEXT/MARKDOWN", "text/plain; charset=utf-8",
        "text/x-csrc", "text/x-chdr", "text/x-c++src", "text/x-c++hdr",
    ])
    def test_supported(self, mime):
        assert TextNormalizer().supports(raw("x", mime))

    @pytest.mark.parametrize("mime", ["text/html", "application/pdf", "text/x-python", "", "text/plainx"])
    def test_unsupported(self, mime):
        assert not TextNormalizer().supports(raw("x", mime))
        with pytest.raises(UnsupportedDocumentError):
            TextNormalizer().normalize(raw("x", mime))

    def test_document_fields(self):
        doc = TextNormalizer().normalize(raw("hello", "text/plain"))
        assert doc.id
        assert doc.raw_document_id == "raw-1"
        assert doc.source_id == "src-1"

    @pytest.mark.parametrize("mime", ["text/plain", "text/markdown", "text/x-c++src"])
    def test_raw_not_mutated_or_aliased(self, mime):
        document = raw("content", mime, extra={"nested": [1]})
        before = copy.deepcopy(document)
        doc = TextNormalizer().normalize(document)
        doc.metadata["extra"]["nested"].append(2)
        assert document == before

    @pytest.mark.parametrize("mime", ["text/plain", "text/markdown", "text/x-c++src"])
    @pytest.mark.parametrize("content", ["", "   \n\n  ", "﻿"])
    def test_empty_inputs(self, mime, content):
        doc = TextNormalizer().normalize(raw(content, mime))
        assert doc.content == ""

    def test_metadata_is_json_serializable(self):
        doc = md("---\ndate: 2026-03-01\nupdated: 2026-03-02T10:00:00\nnested: {when: 2026-01-01}\n---\nbody")
        json.dumps(doc.metadata)


# --------------------------------------------------------------------------- #
# Plain text
# --------------------------------------------------------------------------- #

class TestPlainText:
    def test_format_and_metadata(self):
        doc = TextNormalizer().normalize(raw("hello", "text/plain"))
        assert doc.format == TextFormat.PLAIN
        assert doc.metadata["page_metadata"] == {}
        assert doc.metadata["path"] == "/notes/file.txt"

    def test_shared_cleanup_applied(self):
        doc = TextNormalizer().normalize(raw("﻿pri​ce: 10  \r\n\r\n\r\n\r\nnext", "text/plain"))
        assert doc.content == "price: 10\n\nnext"

    def test_no_reflow(self):
        text = "line one\nline two\n    indented line"
        assert TextNormalizer().normalize(raw(text, "text/plain")).content == text

    def test_markdown_syntax_left_alone_in_plain_text(self):
        text = "![not an image](x) <!-- not a comment -->"
        assert TextNormalizer().normalize(raw(text, "text/plain")).content == text


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #

class TestMarkdownCleanup:
    def test_format(self):
        assert md("# Hi").format == TextFormat.MARKDOWN

    def test_images_removed(self):
        assert md("Intro ![diagram](img/d.png) text.").content == "Intro text."

    def test_reference_style_image_removed(self):
        assert md("See ![logo][logo-ref] here").content == "See here"

    def test_html_img_removed(self):
        assert md('Hi <img src="a.png" alt="x"/> there').content == "Hi there"

    def test_badge_line_removed_entirely(self):
        text = ("# Project\n\n"
                "[![Build](https://img.shields.io/b.svg)](https://ci) [![Ver](https://img.shields.io/v.svg)](https://x)\n\n"
                "Real text.")
        assert md(text).content == "# Project\n\nReal text."

    def test_links_kept(self):
        assert md("Read [the docs](https://example.com/docs).").content == "Read [the docs](https://example.com/docs)."

    def test_html_comment_removed(self):
        assert md("before <!-- hidden --> after").content == "before after"

    def test_multiline_html_comment_removed(self):
        text = "# T\n\n<!-- TODO:\n     expand this\n-->\nBody text."
        assert md(text).content == "# T\n\nBody text."

    def test_unclosed_comment_left_alone(self):
        assert "<!-- oops" in md("text <!-- oops never closed").content

    def test_cleanup_applied(self):
        assert md("a b\r\n\r\n\r\n\r\nc").content == "a b\n\nc"


class TestMarkdownCodeProtection:
    def test_fenced_code_untouched(self):
        code = "```cpp\n// see ![img](x) and <!-- not a comment -->\nint main() { return 0; }\n```"
        assert md(f"Intro\n\n{code}\n\nOutro").content == f"Intro\n\n{code}\n\nOutro"

    def test_fence_language_hint_kept(self):
        assert "```cpp\n" in md("```cpp\nint x;\n```").content

    def test_tilde_fence(self):
        code = "~~~\n<!-- keep -->\n~~~"
        assert md(code).content == code

    def test_longer_fence_not_closed_by_shorter(self):
        code = "````\n```\n<!-- keep -->\n```\n````"
        assert md(code + "\n\n<!-- drop -->text").content == code + "\n\ntext"

    def test_unclosed_fence_protects_until_end(self):
        text = "Intro\n\n```cpp\nint x; // <!-- keep -->"
        assert md(text).content == text

    def test_indented_code_untouched(self):
        text = "Intro:\n\n    auto img = \"![x](y)\"; // <!-- keep -->\n    return img;\n\nAfter ![drop](z)."
        assert md(text).content == "Intro:\n\n    auto img = \"![x](y)\"; // <!-- keep -->\n    return img;\n\nAfter."

    def test_indented_continuation_line_is_not_code(self):
        # inside a paragraph, an indented line is a continuation (CommonMark)
        assert md("para line\n    continued ![drop](x)").content == "para line\n    continued"

    def test_split_code_roundtrip(self):
        text = "a\n\n```\nb\n```\n\n    c\n\nd ![x](y)\n~~~\ne"
        assert "".join(chunk for _, chunk in _split_code(text)) == text

    def test_cpp_in_markdown_preserved(self):
        code = ("```cpp\ntemplate <typename T>\nclass Stack {\npublic:\n    T* top() { return &data_.back(); }\n"
                "private:\n    std::vector<T> data_;\n};\nif (p && p->next || a << 2) {}\n```")
        assert md(code).content == code


class TestFrontMatter:
    def test_parsed_and_removed(self):
        doc = md("---\ntitle: Stack in C++\nauthor: Jane Doe\ndescription: A guide\n"
                 "date: 2026-03-01\ntags: [cpp, templates]\n---\n# Heading\n\nBody")
        assert doc.content == "# Heading\n\nBody"
        assert doc.metadata["page_metadata"] == {
            "title": "Stack in C++", "author": "Jane Doe", "description": "A guide",
            "date": "2026-03-01", "tags": ["cpp", "templates"],
        }

    def test_raw_front_matter_kept_in_metadata(self):
        doc = md("---\ntitle: T\ncustom_field: 42\n---\nbody")
        assert doc.metadata["front_matter"] == {"title": "T", "custom_field": 42}

    def test_alternative_keys(self):
        doc = md("---\nauthors: [Jane, Bob]\nsummary: S\npublished: 2025-12-31\nkeywords: a, b; c\n---\nx")
        assert doc.metadata["page_metadata"] == {
            "author": "Jane, Bob", "description": "S", "date": "2025-12-31", "tags": ["a", "b", "c"],
        }

    def test_datetime_date_truncated(self):
        doc = md("---\ndate: 2026-03-01T10:30:00\n---\nx")
        assert doc.metadata["page_metadata"]["date"] == "2026-03-01"

    def test_dots_terminator(self):
        doc = md("---\ntitle: T\n...\nbody")
        assert doc.content == "body"
        assert doc.metadata["page_metadata"]["title"] == "T"

    def test_bom_before_front_matter(self):
        assert md("﻿---\ntitle: T\n---\nbody").content == "body"

    @pytest.mark.parametrize("text", [
        "---\ntitle: [unclosed\n---\nbody",   # malformed YAML
        "---\n- just\n- a list\n---\nbody",   # not a mapping
        "---\njust a string\n---\nbody",
    ])
    def test_invalid_front_matter_left_in_content(self, text):
        doc = md(text)
        assert doc.content.startswith("---")
        assert doc.metadata["front_matter"] == {}

    def test_horizontal_rule_later_in_doc_is_not_front_matter(self):
        doc = md("Intro\n\n---\n\ntitle: nope\n\n---\nmore")
        assert "title: nope" in doc.content
        assert doc.metadata["front_matter"] == {}

    def test_empty_values_omitted(self):
        doc = md("---\ntitle: ''\ntags: []\nauthor:\n---\n# Real\nx")
        assert doc.metadata["page_metadata"] == {"title": "Real"}


class TestTitleFallback:
    def test_first_atx_h1(self):
        assert md("intro\n\n# Main Title #\n\n# Second").metadata["page_metadata"]["title"] == "Main Title"

    def test_setext_h1(self):
        assert md("Main Title\n==========\n\nbody").metadata["page_metadata"]["title"] == "Main Title"

    def test_h2_not_used(self):
        assert "title" not in md("## Only H2\n\nbody").metadata["page_metadata"]

    def test_h1_inside_code_ignored(self):
        doc = md("```\n# not a title\n```\n\n# Real Title")
        assert doc.metadata["page_metadata"]["title"] == "Real Title"

    def test_preprocessor_line_in_code_not_a_title(self):
        doc = md("```cpp\n#include <vector>\n```")
        assert "title" not in doc.metadata["page_metadata"]

    def test_front_matter_title_wins(self):
        assert md("---\ntitle: FM\n---\n# H1").metadata["page_metadata"]["title"] == "FM"


# --------------------------------------------------------------------------- #
# C / C++ source files
# --------------------------------------------------------------------------- #

CPP_SOURCE = (
    "#include <vector>\n"
    "\n"
    "template <typename T>\n"
    "class Stack {\n"
    "public:\n"
    "    void push(const T& v) { data_.push_back(v); }\n"
    "private:\n"
    "    std::vector<T> data_;  // <!-- not html --> ![not](image)\n"
    "};"
)


class TestCodeFiles:
    def test_wrapped_in_fenced_block_with_file_name(self):
        doc = cpp(CPP_SOURCE)
        assert doc.content == f"stack.cpp\n```cpp\n{CPP_SOURCE}\n```"
        assert doc.format == TextFormat.MARKDOWN
        assert doc.metadata["language"] == "cpp"

    @pytest.mark.parametrize("mime, language", [
        ("text/x-csrc", "c"), ("text/x-chdr", "c"), ("text/x-c++src", "cpp"), ("text/x-c++hdr", "cpp"),
    ])
    def test_language_per_mime(self, mime, language):
        doc = cpp("int x;", mime=mime)
        assert f"```{language}\n" in doc.content
        assert doc.metadata["language"] == language

    def test_code_content_not_markdown_processed(self):
        assert "// <!-- not html --> ![not](image)" in cpp(CPP_SOURCE).content

    def test_indentation_and_tabs_kept(self):
        code = "void f() {\n\tif (x) {\n\t\tg();\n\t}\n}"
        assert f"\n{code}\n" in cpp(code).content

    def test_crlf_and_trailing_space_cleaned(self):
        assert cpp("int a;   \r\nint b;\r\n").content == "stack.cpp\n```cpp\nint a;\nint b;\n```"

    def test_code_with_backtick_fence_uses_longer_fence(self):
        code = 'auto s = R"(```)";'
        doc = cpp(code)
        assert doc.content == f"stack.cpp\n````cpp\n{code}\n````"

    def test_without_path_no_file_name_line(self):
        assert cpp("int x;", path=None).content == "```cpp\nint x;\n```"

    def test_file_name_from_url(self):
        doc = TextNormalizer().normalize(RawDocument(
            id="r", source_id="s", content="int x;", mime_type="text/x-c++hdr",
            metadata={"url": "https://example.com/include/stack.hpp"}))
        assert doc.content.startswith("stack.hpp\n")


# --------------------------------------------------------------------------- #
# End to end with the scraper
# --------------------------------------------------------------------------- #

class TestEndToEnd:
    @pytest.mark.parametrize("name, language", [
        ("main.cpp", "cpp"), ("stack.hpp", "cpp"), ("legacy.c", "c"), ("api.h", "c"), ("Widget.C", "cpp"),
    ])
    def test_cpp_file_scraped_and_normalized(self, tmp_path, make_source, name, language):
        path = tmp_path / name
        path.write_text("int main() {\n    return 0;\n}\n", encoding="utf-8")
        raw_doc = TrafilaturaScraperAdapter().fetch(make_source(str(path), SourceType.FILE))
        doc = TextNormalizer().normalize(raw_doc)
        assert doc.content == f"{name}\n```{language}\nint main() {{\n    return 0;\n}}\n```"

    def test_readme_scraped_and_normalized(self, tmp_path, make_source):
        path = tmp_path / "README.md"
        path.write_text("---\ntitle: Lib\n---\n[![ci](b.svg)](ci)\n\n# Lib\n\n```cpp\nint x;\n```\n", encoding="utf-8")
        raw_doc = TrafilaturaScraperAdapter().fetch(make_source(str(path), SourceType.FILE))
        doc = TextNormalizer().normalize(raw_doc)
        assert doc.content == "# Lib\n\n```cpp\nint x;\n```"
        assert doc.metadata["page_metadata"]["title"] == "Lib"
