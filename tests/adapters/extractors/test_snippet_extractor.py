import pytest

from src.adapters.extractors.snippet_extractor import HeuristicSnippetExtractor
from src.adapters.models.normalized_document import NormalizedDocument, TextFormat


CPP = "#include <vector>\nint main() {\n    std::vector<int> v;\n    return 0;\n}"


def doc(content: str, format=TextFormat.MARKDOWN) -> NormalizedDocument:
    return NormalizedDocument(id="n1", raw_document_id="r1", source_id="s1", content=content, format=format)


def extract(content: str, **options):
    return HeuristicSnippetExtractor(**options).extract(doc(content))


class TestLabelled:
    @pytest.mark.parametrize("label, language", [("cpp", "cpp"), ("c++", "cpp"), ("C++", "cpp"), ("cxx", "cpp"), ("c", "c")])
    def test_c_and_cpp_kept(self, label, language):
        [snippet] = extract(f"```{label}\n{CPP}\n```")
        assert snippet.code == CPP
        assert snippet.language == language
        assert snippet.language_declared is True

    @pytest.mark.parametrize("label", ["bash", "python", "cmake", "json", "rust"])
    def test_other_languages_dropped(self, label):
        assert extract(f"```{label}\n{CPP}\n```") == []

    def test_labelled_cpp_kept_even_if_it_parses_badly(self):
        code = "int main() {\n  foo(\n  bar[[[\n}"
        [snippet] = extract(f"```cpp\n{code}\n```")
        assert snippet.parse_score < 0.6

    def test_info_string_with_extra_words(self):
        assert extract(f"```cpp title=main.cpp\n{CPP}\n```")[0].language == "cpp"

    def test_tilde_fences_and_longer_closing(self):
        assert len(extract(f"~~~cpp\n{CPP}\n~~~~")) == 1

    def test_longer_fence_holds_inner_backticks(self):
        code = "// see ```this```\nint main() { return 0; }"
        [snippet] = extract(f"````cpp\n{code}\n````")
        assert snippet.code == code


class TestUnlabelled:
    def test_cpp_detected(self):
        [snippet] = extract(f"```\n{CPP}\n```")
        assert snippet.language == "cpp"
        assert snippet.language_declared is False
        assert snippet.parse_score >= 0.9

    def test_c_detected(self):
        code = "#include <stdio.h>\nint main(void) {\n    printf(\"hi\\n\");\n    return 0;\n}"
        assert extract(f"```\n{code}\n```")[0].language == "c"

    def test_ambiguous_c_or_cpp_is_cpp(self):
        code = "int add(int a, int b) {\n    return a + b;\n}"
        assert extract(f"```\n{code}\n```")[0].language == "cpp"

    @pytest.mark.parametrize("text", [
        "$ g++ -O2 main.cpp -o main\n./main",
        "moved\ndone\nexit code 0",
        "def f(x):\n    return x * 2\nprint(f(3))",
        "The quick brown fox jumps\nover the lazy dog many times.",
        "1 2 3 4 5 6 7 8\n9 10 11 12 13 14",
    ])
    def test_non_code_dropped(self, text):
        assert extract(f"```\n{text}\n```") == []

    def test_threshold_is_configurable(self):
        code = "int main() {\n  printf(\"%d\", 1)\n  return 0;\n}"
        assert len(extract(f"```\n{code}\n```")) == 1
        assert extract(f"```\n{code}\n```", min_parse_score=0.95) == []


class TestFilters:
    def test_one_liner_dropped(self):
        assert extract("```cpp\nstd::vector<int> v{1, 2, 3};\n```") == []

    def test_short_code_dropped(self):
        assert extract("```cpp\nint a;\nint b;\n```") == []

    def test_limits_are_configurable(self):
        assert len(extract("```cpp\nstd::vector<int> v{1, 2, 3};\n```", min_lines=1)) == 1

    def test_duplicates_dropped_ignoring_whitespace(self):
        other = CPP.replace("    ", "  ")
        snippets = extract(f"```cpp\n{CPP}\n```\n\ntext\n\n```cpp\n{other}\n```")
        assert len(snippets) == 1

    def test_blank_lines_around_code_trimmed(self):
        assert extract(f"```cpp\n\n{CPP}\n\n```")[0].code == CPP

    def test_unclosed_fence_runs_to_end(self):
        assert extract(f"```cpp\n{CPP}")[0].code == CPP


class TestContext:
    def test_heading_and_paragraph_before(self):
        markdown = (
            "# Guide\n\nIntro paragraph.\n\n## Vectors\n\nFirst paragraph.\n\n"
            "Here is how to use a vector:\n\n"
            f"```cpp\n{CPP}\n```"
        )
        [snippet] = extract(markdown)
        assert snippet.heading == "Vectors"
        assert snippet.context == "Here is how to use a vector:"

    def test_setext_heading(self):
        [snippet] = extract(f"Vectors\n=======\n\nUse it:\n\n```cpp\n{CPP}\n```")
        assert snippet.heading == "Vectors"

    def test_multiline_paragraph_joined(self):
        [snippet] = extract(f"This example\nshows vectors.\n```cpp\n{CPP}\n```")
        assert snippet.context == "This example shows vectors."

    def test_context_not_reused_by_next_block(self):
        second = CPP.replace("v;", "w;")
        snippets = extract(f"Explain.\n\n```cpp\n{CPP}\n```\n```cpp\n{second}\n```")
        assert [s.context for s in snippets] == ["Explain.", ""]

    def test_heading_resets_context(self):
        [snippet] = extract(f"Old paragraph.\n\n## New\n\n```cpp\n{CPP}\n```")
        assert snippet.context == ""
        assert snippet.heading == "New"

    def test_headings_inside_code_ignored(self):
        code = "#include <map>\n#define X 1\nint main() { return X; }"
        markdown = f"## Real\n\n```cpp\n{code}\n```\n\n```cpp\n{CPP}\n```"
        assert [s.heading for s in extract(markdown)] == ["Real", "Real"]

    def test_long_context_cut_from_the_start(self):
        paragraph = "word " * 200 + "the end."
        [snippet] = extract(f"{paragraph}\n\n```cpp\n{CPP}\n```", max_context_chars=50)
        assert snippet.context.startswith("…")
        assert snippet.context.endswith("the end.")
        assert len(snippet.context) <= 51

    def test_code_file_from_text_normalizer(self):
        [snippet] = extract(f"main.cpp\n```cpp\n{CPP}\n```")
        assert snippet.context == "main.cpp"


class TestDocument:
    def test_indexes_follow_kept_snippets(self):
        second = CPP.replace("v;", "w;")
        markdown = f"```bash\nls -la\nls\n```\n```cpp\n{CPP}\n```\n```cpp\n{second}\n```"
        assert [s.index for s in extract(markdown)] == [0, 1]

    def test_analysis_fields_filled(self):
        [snippet] = extract(f"```cpp\n{CPP}\n```")
        assert snippet.includes == ["<vector>"]
        assert snippet.functions == ["main"]
        assert snippet.line_count == 5

    def test_plain_documents_use_the_line_detector(self):
        [snippet] = HeuristicSnippetExtractor().extract(doc(f"Some intro text that explains it.\n{CPP}", format=TextFormat.PLAIN))
        assert snippet.code == CPP
        assert snippet.language_declared is False

    def test_no_code(self):
        assert extract("# Title\n\nJust prose.") == []
        assert extract("") == []
