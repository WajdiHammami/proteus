import copy
import logging

import pytest

from src.adapters.models.normalized_document import TextFormat
from src.adapters.models.raw_document import RawDocument
from src.adapters.models.source import SourceType
from src.adapters.normalizers import pdf_normalizer
from src.adapters.normalizers.pdf_normalizer import PdfNormalizer, _parse_pdf_date
from src.adapters.scrapers.pdf_adapter import PdfScraperAdapter
from src.ports.normalization_port import UnsupportedDocumentError


def raw_pdf(pages: list[str], ocr_pages=(), pdf_metadata=None, **metadata) -> RawDocument:
    meta = {
        "path": "/docs/book.pdf",
        "page_count": len(pages),
        "page_separator": "\f",
        "pdf_metadata": pdf_metadata or {},
        "text_layer": {"needs_ocr": bool(ocr_pages), "ocr_pages": list(ocr_pages), "empty_pages": []},
    }
    meta.update(metadata)
    return RawDocument(
        id="raw-1", source_id="src-1", content="\f".join(pages),
        mime_type="application/pdf", metadata=meta,
    )


def normalize(pages, **kwargs):
    adapter_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in
                      ("margin_lines", "min_repeat_ratio", "min_pages_for_repeats", "max_indent")}
    return PdfNormalizer(**adapter_kwargs).normalize(raw_pdf(pages, **kwargs))


def page_text(doc, index: int) -> str:
    start, end = doc.metadata["page_spans"][index]
    return doc.content[start:end]


PYTHON_CODE = (
    "def add(a, b):\n"
    "    if a > b:\n"
    "        return a - b\n"
    "    return a + b"
)


# --------------------------------------------------------------------------- #
# supports() / basics
# --------------------------------------------------------------------------- #

class TestBasics:
    @pytest.mark.parametrize("mime", ["application/pdf", "APPLICATION/PDF", "application/pdf; v=1"])
    def test_supports_pdf(self, mime):
        raw = raw_pdf(["x"])
        raw.mime_type = mime
        assert PdfNormalizer().supports(raw)

    @pytest.mark.parametrize("mime", ["text/html", "text/plain", "", "application/pdfx"])
    def test_rejects_other_types(self, mime):
        raw = raw_pdf(["x"])
        raw.mime_type = mime
        assert not PdfNormalizer().supports(raw)
        with pytest.raises(UnsupportedDocumentError):
            PdfNormalizer().normalize(raw)

    def test_document_fields(self):
        doc = normalize(["hello"])
        assert doc.id
        assert doc.raw_document_id == "raw-1"
        assert doc.source_id == "src-1"
        assert doc.format == TextFormat.PLAIN

    def test_raw_metadata_carried_forward(self):
        doc = normalize(["hello"], ocr_pages=[], crawl_depth=3)
        assert doc.metadata["path"] == "/docs/book.pdf"
        assert doc.metadata["crawl_depth"] == 3
        assert "text_layer" in doc.metadata

    def test_raw_document_not_mutated_or_aliased(self):
        raw = raw_pdf(["a", "b"], ocr_pages=[1])
        before = copy.deepcopy(raw)
        doc = PdfNormalizer().normalize(raw)
        doc.metadata["text_layer"]["ocr_pages"].append(99)
        assert raw == before

    @pytest.mark.parametrize("pages", [[], [""], ["", ""], ["   \n\n  ", "\n"]])
    def test_empty_inputs(self, pages):
        doc = PdfNormalizer().normalize(raw_pdf(pages) if pages else RawDocument(
            id="r", source_id="s", content="", mime_type="application/pdf", metadata={"page_count": 0}))
        assert doc.content == ""
        assert doc.is_empty()


# --------------------------------------------------------------------------- #
# Code preservation (main purpose)
# --------------------------------------------------------------------------- #

class TestCodePreservation:
    def test_line_breaks_and_indentation_kept(self):
        assert normalize([PYTHON_CODE]).content == PYTHON_CODE

    def test_lines_are_not_unwrapped(self):
        code = "x = compute(\n    a,\n    b,\n)"
        assert normalize([code]).content == code

    def test_common_margin_removed_relative_indent_kept(self):
        indented = "\n".join("      " + line for line in PYTHON_CODE.split("\n"))
        assert normalize([indented]).content == PYTHON_CODE

    def test_layout_padding_collapsed_inside_lines(self):
        # pypdf layout mode pads gaps between text runs with many spaces
        assert normalize(["result   =    add(1,      2)"]).content == "result = add(1, 2)"

    def test_tabs_in_indentation_kept(self):
        assert normalize(["func main() {\n\tfmt.Println(1)\n}"]).content == "func main() {\n\tfmt.Println(1)\n}"

    def test_deep_indentation_beyond_max_is_stripped(self):
        doc = normalize([" " * 60 + "Centered Title\nbody"])
        assert doc.content == "Centered Title\nbody"

    def test_max_indent_is_configurable(self):
        doc = normalize([" " * 10 + "x\ny"], max_indent=4)
        assert doc.content == "x\ny"

    def test_closing_braces_at_page_ends_are_kept(self):
        pages = [f"function f{i}() {{\n  return {i};\n}}" for i in range(5)]
        doc = normalize(pages)
        assert doc.content.count("}") == 5

    def test_identical_code_line_at_page_end_is_kept(self):
        pages = [f"Recipe number {i} explains something new\n\n{PYTHON_CODE}" for i in range(4)]
        doc = normalize(pages)
        assert doc.content.count("    return a + b") == 4

    def test_code_minus_at_line_end_not_dehyphenated_when_next_line_indented(self):
        code = "total = count -\n    offset"
        assert normalize([code]).content == code

    def test_cli_flag_line_not_treated_as_page_number(self):
        doc = normalize(["Run it:\n$ tool --page 3\nnext"])
        assert "--page 3" in doc.content


# --------------------------------------------------------------------------- #
# Text cleanup
# --------------------------------------------------------------------------- #

class TestTextCleanup:
    @pytest.mark.parametrize("ligature, expected", [
        ("\ufb01le", "file"), ("\ufb02ow", "flow"), ("o\ufb00set", "offset"),
        ("e\ufb03cient", "efficient"), ("ba\ufb04e", "baffle"),
    ])
    def test_ligatures_expanded(self, ligature, expected):
        assert normalize([ligature]).content == expected

    def test_superscripts_and_fractions_not_mangled(self):
        assert normalize(["area 5 m² and ½ cup"]).content == "area 5 m² and ½ cup"

    def test_hyphenated_line_break_rejoined(self):
        assert normalize(["the infor-\nmation is here"]).content == "the information is here"

    def test_hyphen_before_capital_kept(self):
        # "Anglo-\nSaxon" style compounds with a capital are left alone
        assert normalize(["Anglo-\nSaxon"]).content == "Anglo-\nSaxon"

    def test_hyphenation_across_page_break(self):
        doc = normalize(["we explain the infor-", "mation on this page"])
        assert doc.content == "we explain the information on this page"
        assert page_text(doc, 0) == "we explain the infor"
        assert page_text(doc, 1) == "mation on this page"

    def test_blank_line_runs_collapsed(self):
        assert normalize(["a\n\n\n\n\nb"]).content == "a\n\nb"

    def test_invisible_chars_and_nbsp_cleaned(self):
        assert normalize(["pri\u200bce:\u00a010"]).content == "price: 10"


# --------------------------------------------------------------------------- #
# Pages: OCR drops, joining and spans
# --------------------------------------------------------------------------- #

class TestPages:
    def test_pages_joined_with_single_newline(self):
        assert normalize(["page one", "page two"]).content == "page one\npage two"

    def test_page_spans_point_to_each_page(self):
        doc = normalize(["first page", "second", "third one"])
        assert [page_text(doc, i) for i in range(3)] == ["first page", "second", "third one"]

    def test_ocr_flagged_pages_are_dropped(self):
        doc = normalize(["good text", "\ue001\ue002 garbled junk", "more good"], ocr_pages=[1])
        assert doc.content == "good text\nmore good"
        assert doc.metadata["dropped_pages"] == [1]
        assert page_text(doc, 1) == ""

    def test_empty_page_gets_zero_length_span(self):
        doc = normalize(["a", "", "b"])
        start, end = doc.metadata["page_spans"][1]
        assert start == end
        assert doc.content == "a\nb"

    def test_all_pages_dropped_gives_empty_document(self):
        doc = normalize(["x", "y"], ocr_pages=[0, 1])
        assert doc.content == ""
        assert doc.metadata["page_spans"] == [[0, 0], [0, 0]]

    def test_out_of_range_ocr_indices_ignored(self):
        doc = normalize(["a"], ocr_pages=[5])
        assert doc.content == "a"
        assert doc.metadata["dropped_pages"] == []

    def test_missing_text_layer_metadata(self):
        raw = RawDocument(id="r", source_id="s", content="a\fb", mime_type="application/pdf", metadata={})
        doc = PdfNormalizer().normalize(raw)
        assert doc.content == "a\nb"
        assert doc.metadata["dropped_pages"] == []

    def test_page_count_mismatch_treated_as_single_page(self, caplog):
        raw = raw_pdf(["a", "b"])
        raw.content = "a\fb\fextra"  # e.g. a form feed inside a page's own text
        with caplog.at_level(logging.WARNING, logger=pdf_normalizer.__name__):
            doc = PdfNormalizer().normalize(raw)
        assert len(doc.metadata["page_spans"]) == 1
        assert "mismatch" in caplog.records[0].getMessage()

    def test_custom_page_separator(self):
        raw = raw_pdf(["a", "b"], page_separator="<PAGE>")
        raw.content = "a<PAGE>b"
        assert PdfNormalizer().normalize(raw).content == "a\nb"


# --------------------------------------------------------------------------- #
# Headers / footers / page numbers
# --------------------------------------------------------------------------- #

def book(n: int, header=None, footer=None) -> list[str]:
    pages = []
    for i in range(n):
        body = f"Unique body text number {i} talks about topic {chr(65 + i)}."
        parts = ([header(i)] if header else []) + [body] + ([footer(i)] if footer else [])
        pages.append("\n".join(parts))
    return pages


class TestBoilerplate:
    def test_repeated_header_removed(self):
        doc = normalize(book(4, header=lambda i: "ACME Corp - Annual Report"))
        assert "ACME" not in doc.content
        assert "Unique body text number 3" in doc.content
        assert doc.metadata["removed_boilerplate"] == ["acme corp - annual report"]

    def test_header_with_changing_numbers_removed(self):
        doc = normalize(book(4, header=lambda i: f"Chapter {i + 1} - Getting started"))
        assert "Chapter" not in doc.content

    def test_repeated_footer_removed(self):
        doc = normalize(book(3, footer=lambda i: "Confidential - do not distribute"))
        assert "Confidential" not in doc.content

    def test_header_on_half_the_pages_removed(self):
        pages = book(4)
        pages[0] = "Running Header Text\n" + pages[0]
        pages[2] = "Running Header Text\n" + pages[2]
        assert "Running Header" not in normalize(pages).content

    def test_line_on_too_few_pages_kept(self):
        pages = book(5)
        pages[0] = "Running Header Text\n" + pages[0]
        pages[1] = "Running Header Text\n" + pages[1]
        assert normalize(pages).content.count("Running Header Text") == 2

    def test_too_few_pages_for_repeat_detection(self):
        doc = normalize(book(2, header=lambda i: "ACME Corp - Annual Report"))
        assert doc.content.count("ACME") == 2

    def test_repeat_in_middle_of_page_kept(self):
        pages = [f"Top line {chr(65+i)} unique\nImportant note here\nmiddle\nmore\nend {chr(65+i)} unique"
                 for i in range(4)]
        assert normalize(pages, margin_lines=1).content.count("Important note here") == 4

    def test_same_line_at_different_positions_not_matched(self):
        # appears once as the 1st line and once as the last line: different slots
        pages = book(4)
        pages[0] = "Shared sentence here\n" + pages[0]
        pages[1] = pages[1] + "\nShared sentence here"
        assert normalize(pages).content.count("Shared sentence here") == 2

    @pytest.mark.parametrize("footer", [
        lambda i: f"Page {i + 1}",
        lambda i: f"Page {i + 1} of 4",
        lambda i: f"page {i + 1}/4",
        lambda i: f"{i + 1} of 4",
        lambda i: f"{i + 1}/4",
        lambda i: f"- {i + 1} -",
        lambda i: f"— {i + 1} —",
        lambda i: f"p. {i + 1}",
    ])
    def test_page_labels_removed(self, footer):
        doc = normalize(book(4, footer=footer))
        assert doc.content.count("\n") == 3  # just the 4 body lines remain
        assert "of 4" not in doc.content

    def test_page_label_on_single_page_document(self):
        assert normalize(["Body text\nPage 1 of 1"]).content == "Body text"

    def test_bare_page_numbers_tracking_index_removed(self):
        doc = normalize(book(4, footer=lambda i: str(i + 11)))  # printed numbers offset from index
        assert not any(line.strip().isdigit() for line in doc.content.split("\n"))

    def test_bare_number_not_tracking_index_kept(self):
        pages = book(4)
        pages[2] += "\n42"
        assert "42" in normalize(pages).content

    def test_bare_numbers_on_single_page_kept(self):
        assert normalize(["Body\n7"]).content == "Body\n7"

    def test_header_detection_ignores_dropped_pages(self):
        pages = book(4, header=lambda i: "ACME Corp - Annual Report")
        doc = normalize(pages, ocr_pages=[0, 1])
        # only 2 pages with text left -> below min_pages_for_repeats, header kept
        assert doc.content.count("ACME") == 2

    @pytest.mark.parametrize("line", ["return result", "end", "done", "import os", "pass"])
    def test_code_keyword_lines_never_boilerplate(self, line):
        pages = [f"Section {chr(65+i)} intro text\n{line}" for i in range(4)]
        assert normalize(pages).content.count(line) == 4

    def test_code_like_lines_never_boilerplate(self):
        pages = [f"Section {chr(65+i)} intro\nprint(result)" for i in range(4)]
        assert normalize(pages).content.count("print(result)") == 4

    def test_blank_lines_left_by_removal_are_collapsed(self):
        doc = normalize(book(3, header=lambda i: "ACME Corp Report\n"))
        assert not doc.content.startswith("\n")
        assert "\n\n\n" not in doc.content


# --------------------------------------------------------------------------- #
# Metadata mapping
# --------------------------------------------------------------------------- #

class TestPageMetadata:
    def test_maps_pdf_info(self):
        doc = normalize(["x"], pdf_metadata={
            "Title": " Python Recipes ", "Author": "Jane Doe", "Subject": "Cookbook",
            "Keywords": "python, recipes; code ,", "CreationDate": "D:20260301120000+01'00'",
        })
        assert doc.metadata["page_metadata"] == {
            "title": "Python Recipes", "author": "Jane Doe", "description": "Cookbook",
            "tags": ["python", "recipes", "code"], "date": "2026-03-01",
        }

    def test_empty_fields_omitted(self):
        doc = normalize(["x"], pdf_metadata={"Title": "  ", "Keywords": ""})
        assert doc.metadata["page_metadata"] == {}

    def test_mod_date_used_when_no_creation_date(self):
        doc = normalize(["x"], pdf_metadata={"ModDate": "D:20251231"})
        assert doc.metadata["page_metadata"]["date"] == "2025-12-31"

    def test_no_pdf_metadata_key(self):
        raw = RawDocument(id="r", source_id="s", content="x", mime_type="application/pdf", metadata={})
        assert PdfNormalizer().normalize(raw).metadata["page_metadata"] == {}

    @pytest.mark.parametrize("value, expected", [
        ("D:20260301120000Z", "2026-03-01"),
        ("D:2026", "2026-01-01"),
        ("D:202604", "2026-04-01"),
        ("  D:20260301", "2026-03-01"),
        ("D:20261301", None),   # month 13
        ("D:20260100", None),   # day 00
        ("2026-03-01", None),   # not PDF date syntax
        ("", None),
        ("garbage", None),
    ])
    def test_parse_pdf_date(self, value, expected):
        assert _parse_pdf_date(value) == expected


# --------------------------------------------------------------------------- #
# End to end with the real scraper
# --------------------------------------------------------------------------- #

class TestEndToEnd:
    CODE_LINES = [(0, "def add(a, b):"), (24, "if a > b:"), (48, "return a - b"), (24, "return a + b")]

    def test_scraper_and_normalizer_preserve_code_structure(self, write_pdf, make_source):
        pages = [{"text": f"ACME Python Guide {i}", "lines": [(0, f"Example {chr(65 + i)} shows adding")]
                  + self.CODE_LINES} for i in range(3)]
        raw = PdfScraperAdapter().fetch(make_source(write_pdf(pages), SourceType.PDF))
        doc = PdfNormalizer().normalize(raw)

        lines = doc.content.split("\n")
        def indent(prefix):
            line = next(l for l in lines if l.strip().startswith(prefix))
            return len(line) - len(line.lstrip())

        assert indent("def add") == 0
        assert 0 < indent("if a > b") < indent("return a - b")
        assert indent("return a + b") == indent("if a > b")
        assert "ACME Python Guide" not in doc.content
        assert len(doc.content) < len(raw.content)

    def test_scanned_page_dropped_end_to_end(self, write_pdf, make_source):
        pages = [{"text": "A real page with a proper text layer."}, {"image": True}]
        raw = PdfScraperAdapter().fetch(make_source(write_pdf(pages), SourceType.PDF))
        doc = PdfNormalizer().normalize(raw)
        assert doc.metadata["dropped_pages"] == [1]
        assert doc.content == "A real page with a proper text layer."


# --------------------------------------------------------------------------- #
# C++ (primary target)
# --------------------------------------------------------------------------- #

CPP_CLASS = (
    "template <typename T>\n"
    "class Stack {\n"
    "public:\n"
    "    virtual ~Stack() = default;\n"
    "    void push(const T& v) { data_.push_back(v); }\n"
    "    T* top() { return &data_.back(); }\n"
    "private:\n"
    "    std::vector<T> data_;\n"
    "};"
)


class TestCpp:
    def test_cpp_code_untouched(self):
        assert normalize([CPP_CLASS]).content == CPP_CLASS

    def test_operators_and_pointers_survive(self):
        code = "if (p && p->next != nullptr || a << 2) { std::cout << *p << std::endl; }"
        assert normalize([code]).content == code

    def test_preprocessor_lines_survive(self):
        code = "#include <vector>\n#define MAX(a, b) ((a) > (b) ? (a) : (b))\n#ifdef DEBUG\n#endif"
        assert normalize([code]).content == code

    @pytest.mark.parametrize("line", [
        "public:", "private:", "protected:", "default:", "case RED:", "};", "}", "else",
        "std::sort(v.begin(), v.end());", "ptr->next", "template <class T>",
        "virtual void draw() const", "return nullptr;", "delete ptr", "new Widget",
    ])
    def test_cpp_lines_at_page_edges_never_boilerplate(self, line):
        pages = [f"{line}\nUnique body {chr(65 + i)} here\n{line}" for i in range(4)]
        assert normalize(pages).content.count(line) == 8

    @pytest.mark.parametrize("header", [
        "Modern C++ Patterns - Chapter {n}",
        "Effective C++ Item {n}",
        "C++20 in Practice",
        "Learning C# and C++ Together",
    ])
    def test_headers_mentioning_cpp_are_still_removed(self, header):
        pages = book(4, header=lambda i: header.format(n=i + 1))
        doc = normalize(pages)
        assert "C++" not in doc.content and "C#" not in doc.content

    def test_language_name_does_not_hide_real_code(self):
        pages = [f"Section {chr(65 + i)} talks about C++\nx += 1" for i in range(4)]
        assert normalize(pages).content.count("x += 1") == 4

    def test_capitalized_header_starting_with_keyword_word_removed(self):
        # keyword guard is case-sensitive: "New ..." is prose, "new Widget" is code
        doc = normalize(book(4, header=lambda i: "New Features Overview"))
        assert "New Features" not in doc.content

    def test_arrow_not_dehyphenated(self):
        code = "node-\n>next = nullptr;"
        assert normalize([code]).content == code

    def test_default_max_indent_strips_centered_headers_but_keeps_nesting(self):
        page = " " * 30 + "Chapter Title\nvoid f() {\n        if (x) {\n                g();\n        }\n}"
        assert normalize([page]).content == "Chapter Title\nvoid f() {\n        if (x) {\n                g();\n        }\n}"

    def test_cpp_book_end_to_end(self, write_pdf, make_source):
        pages = [{
            "text": "Modern C++ Patterns",
            "lines": [(0, f"Listing for topic {chr(65 + i)}"), (0, "class Stack {"), (0, "public:"),
                      (24, "void push(int v);"), (0, "private:"), (24, "std::vector<int> data_;"), (0, "};")],
        } for i in range(3)]
        raw = PdfScraperAdapter().fetch(make_source(write_pdf(pages), SourceType.PDF))
        doc = PdfNormalizer().normalize(raw)

        assert "Modern C++ Patterns" not in doc.content
        for line in ("class Stack {", "public:", "void push(int v);", "private:", "std::vector<int> data_;", "};"):
            assert doc.content.count(line) == 3, line
