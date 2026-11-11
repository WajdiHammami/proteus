import pytest

from src.adapters.extractors.plain_code_detector import LineKind, classify_line, find_code_blocks
from src.adapters.extractors.snippet_extractor import HeuristicSnippetExtractor
from src.adapters.models.normalized_document import NormalizedDocument, TextFormat
from src.adapters.models.source import SourceType
from src.adapters.normalizers.pdf_normalizer import PdfNormalizer
from src.adapters.scrapers.pdf_adapter import PdfScraperAdapter
from tests.conftest import build_pdf


class TestClassifyLine:
    @pytest.mark.parametrize("line", [
        "#include <vector>", "  # define MAX 10", "#pragma once",
        "int x = 5;", "return 0;", "}", "};", "{", "else",
        "class Buffer {", "template <typename T>", "struct Point {", "namespace app {",
        "public:", "  private:", "case 3:", "default:",
        "Buffer(Buffer&& other) noexcept", ": data_(other.data_), size_(other.size_) {",
        "other.data_ = nullptr;", "std::cout << x << '\\n';", "for (auto& v : values) {",
        "Buffer b = std::move(a); // a is now empty", "// Moves the buffer into place.",
        "/* old style comment", " * continued comment", "virtual ~Shape() = default;",
        "int main()", "void push(const T& v)", "12 int x = 1;", "3   return 0;",
    ])
    def test_code(self, line):
        assert classify_line(line) == LineKind.CODE

    @pytest.mark.parametrize("line", [
        "Moving an object transfers its resources instead of copying them. This",
        "The call std::move(a) casts a to an rvalue reference.",
        "You can now write the following code:",
        "using this approach, we avoid copying the buffer at all.",
        "static members are shared by all instances of the class.",
        "Return to chapter three for the full discussion of this topic.",
        "and the compiler calls the move constructor. Running the program prints:",
        "2024 was a good year for C++ compiler vendors and users alike.",
    ])
    def test_prose(self, line):
        assert classify_line(line) == LineKind.PROSE

    @pytest.mark.parametrize("line", ["3.2 Move Semantics", "moved 1024 bytes", "Listing 3.4 A movable buffer", "Output"])
    def test_neutral(self, line):
        assert classify_line(line) == LineKind.NEUTRAL

    @pytest.mark.parametrize("line", ["", "   ", "\t"])
    def test_blank(self, line):
        assert classify_line(line) == LineKind.BLANK


class TestRegions:
    def test_blank_lines_inside_code_kept(self):
        text = "Intro sentence explaining the code below.\nint a = 1;\n\nint b = 2;\nNow some more prose follows here."
        [block] = find_code_blocks(text)
        assert block.code == "int a = 1;\n\nint b = 2;"

    def test_prose_line_ends_region(self):
        text = "int a = 1;\nint b = 2;\nThis sentence is clearly prose and ends it.\nint c = 3;\nint d = 4;"
        blocks = find_code_blocks(text)
        assert [b.code for b in blocks] == ["int a = 1;\nint b = 2;", "int c = 3;\nint d = 4;"]

    def test_neutral_lines_bridged(self):
        text = "void f(int a,\n       some_wrapped_argument)\n{\n    g();\n}"
        [block] = find_code_blocks(text)
        assert "some_wrapped_argument" in block.code

    def test_large_gap_splits(self):
        text = "int a = 1;\n\n\n\n\nint b = 2;"
        assert len(find_code_blocks(text)) == 2
        assert len(find_code_blocks(text, max_gap=5)) == 1

    def test_trailing_neutral_lines_not_included(self):
        [block] = find_code_blocks("int a = 1;\nint b = 2;\n   moved 1024 bytes")
        assert block.code == "int a = 1;\nint b = 2;"

    def test_common_indentation_removed_relative_kept(self):
        [block] = find_code_blocks("   int f() {\n       return 1;\n   }")
        assert block.code == "int f() {\n    return 1;\n}"

    def test_margin_line_numbers_stripped(self):
        [block] = find_code_blocks("1 #include <iostream>\n2 int main() {\n3     return 0;\n4 }")
        assert block.code == "#include <iostream>\nint main() {\n    return 0;\n}"

    def test_numbers_kept_when_not_increasing(self):
        [block] = find_code_blocks("3 int a = 1;\n2 int b = 2;")
        assert block.code.startswith("3 int a")

    def test_numbers_kept_when_only_some_lines_have_them(self):
        [block] = find_code_blocks("int a = 1;\nint b = 2;\nint c = 3;\n10 * x;\nint d = 4;")
        assert "10 * x;" in block.code


class TestContext:
    def test_heading_caption_and_paragraph(self):
        text = (
            "3.2 Move Semantics\n\nMoving transfers resources instead of copying them.\n"
            "The following listing shows how:\n\nListing 3.4 A movable buffer\n"
            "class Buffer {\npublic:\n    int* data_;\n};"
        )
        [block] = find_code_blocks(text)
        assert block.heading == "3.2 Move Semantics"
        assert block.context == ("Moving transfers resources instead of copying them. The following listing shows how: "
                                 "Listing 3.4 A movable buffer")

    @pytest.mark.parametrize("heading", ["Chapter 4: Templates", "Section 2 Basics", "1. Introduction", "4.1.2 Rvalue References"])
    def test_heading_forms(self, heading):
        [block] = find_code_blocks(f"{heading}\nint a = 1;\nint b = 2;")
        assert block.heading == heading

    def test_numbered_code_line_is_not_a_heading(self):
        blocks = find_code_blocks("2.1 Basics\nIntro text before the listing.\n1 Buffer b;\n2 Buffer c;")
        assert blocks[0].heading == "2.1 Basics"

    def test_program_output_breaks_context(self):
        text = "Running it prints:\n   moved 1024 bytes\nExample with line numbers:\nint a = 1;\nint b = 2;"
        [block] = find_code_blocks(text)
        assert block.context == "Example with line numbers:"

    def test_context_and_caption_reset_after_block(self):
        text = "Listing 1 First\nint a = 1;\nint b = 2;\nThis is a prose line between blocks.\n\nint c = 3;\nint d = 4;"
        blocks = find_code_blocks(text)
        assert blocks[0].context == "Listing 1 First"
        assert blocks[1].context == "This is a prose line between blocks."


def plain(content: str) -> NormalizedDocument:
    return NormalizedDocument(id="n1", raw_document_id="r1", source_id="s1", content=content, format=TextFormat.PLAIN)


class TestThroughExtractor:
    def test_shell_and_output_regions_dropped(self):
        text = "Compile and run it as shown:\n$ g++ -O2 main.cpp -o main;\n$ ./main;\nmoved\ndone"
        assert HeuristicSnippetExtractor().extract(plain(text)) == []

    def test_prose_only(self):
        assert HeuristicSnippetExtractor().extract(plain("Just a paragraph of prose.\nAnd another sentence here.")) == []

    def test_c_detected_in_plain_text(self):
        text = "#include <stdio.h>\nint main(void) {\n    printf(\"hi\\n\");\n    return 0;\n}"
        assert HeuristicSnippetExtractor().extract(plain(text))[0].language == "c"


class TestPdfEndToEnd:
    """Scraper -> normalizer -> extractor on a generated C++ book page."""

    PAGE = [
        (0, "3.2 Move Semantics"), (0, ""),
        (0, "Moving an object transfers its resources instead of copying them. This"),
        (0, "following listing shows a buffer with a move constructor:"), (0, ""),
        (0, "Listing 3.4 A movable buffer"),
        (12, "#include <utility>"), (12, ""), (12, "class Buffer {"), (12, "public:"),
        (36, "Buffer(Buffer&& other) noexcept"), (48, ": data_(other.data_) {"),
        (60, "other.data_ = nullptr;"), (48, "}"), (12, "private:"), (36, "int* data_;"), (12, "};"),
        (0, ""), (0, "Note that the moved-from object is left in a valid but unspecified state."),
        (0, "Running the program prints:"), (12, "moved 1024 bytes"),
    ]

    def test_listing_found_with_context(self, write_pdf, make_source):
        raw = PdfScraperAdapter().fetch(make_source(write_pdf([{"lines": self.PAGE}]), SourceType.PDF))
        [snippet] = HeuristicSnippetExtractor().extract(PdfNormalizer().normalize(raw))
        assert snippet.language == "cpp"
        assert snippet.code.startswith("#include <utility>\n\nclass Buffer {")
        assert snippet.code.endswith("};")
        assert "moved 1024 bytes" not in snippet.code
        assert snippet.heading == "3.2 Move Semantics"
        assert snippet.context.endswith("Listing 3.4 A movable buffer")
        assert snippet.classes == ["Buffer"]
        assert snippet.includes == ["<utility>"]

    def test_listing_split_across_pages(self, write_pdf, make_source):
        first = [(0, "The whole class is shown below in one listing:"), (0, "class Stack {"), (0, "public:"),
                 (24, "void push(int v) { data_.push_back(v); }")]
        second = [(24, "int top() const { return data_.back(); }"), (0, "private:"),
                  (24, "std::vector<int> data_;"), (0, "};")]
        raw = PdfScraperAdapter().fetch(make_source(write_pdf([{"lines": first}, {"lines": second}]), SourceType.PDF))
        [snippet] = HeuristicSnippetExtractor().extract(PdfNormalizer().normalize(raw))
        assert snippet.code.startswith("class Stack {")
        assert "int top() const" in snippet.code and snippet.code.endswith("};")
        assert snippet.functions == ["push", "top"]
