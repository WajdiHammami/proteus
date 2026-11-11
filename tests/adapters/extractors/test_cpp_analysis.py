import pytest

from src.adapters.extractors.cpp_analysis import analyze


class TestParseScore:
    @pytest.mark.parametrize("code", [
        "#include <vector>\nint main() { std::vector<int> v; return 0; }",
        "std::vector<int> v;\nv.push_back(3);",                       # loose statements
        "template <typename T>\nT add(T a, T b) { return a + b; }",
        "#include <stdio.h>\nint main(void) { printf(\"hi\"); return 0; }",   # C
        "auto [a, b] = std::pair{1, 2};\nconstexpr int n = 3;",
    ])
    def test_valid_code_scores_high(self, code):
        assert analyze(code).parse_score >= 0.9

    @pytest.mark.parametrize("code", [
        "The quick brown fox jumps over the lazy dog.",
        "$ g++ -O2 main.cpp -o main\n./main",
        "pip install numpy",
        "1 2 3\n4 5 6",
        "Hello, world!\nValue: 42",
    ])
    def test_non_code_scores_low(self, code):
        assert analyze(code).parse_score < 0.6

    def test_one_typo_still_passes_threshold(self):
        code = "int main() {\n  printf(\"%d\", 1)\n  return 0;\n}"
        assert 0.6 <= analyze(code).parse_score < 1.0

    def test_python_scores_low(self):
        assert analyze("def f(x):\n    return x * 2\n\nprint(f(3))").parse_score < 0.6

    @pytest.mark.parametrize("code", ["", "   \n\n"])
    def test_blank(self, code):
        assert analyze(code).parse_score == 0.0


class TestCollected:
    CODE = (
        '#include <vector>\n#include "util.h"\n#include <vector>\n'
        "template <typename T> class Box { T v; };\n"
        "struct Point { int x, y; };\n"
        "struct Forward;\n"
        "enum class Color { Red };\n"
        "int* S::get() const { return p; }\n"
        "void ns::A::operator+=(int) {}\n"
        "A::~A() {}\n"
        "int& ref() { static int x; return x; }\n"
        "int main() { return 0; }\n"
    )

    def test_includes_in_order_without_duplicates(self):
        assert analyze(self.CODE).includes == ["<vector>", '"util.h"']

    def test_functions_with_qualified_names(self):
        assert analyze(self.CODE).functions == ["S::get", "ns::A::operator+=", "A::~A", "ref", "main"]

    def test_only_defined_classes(self):
        assert analyze(self.CODE).classes == ["Box", "Point", "Color"]   # not the forward declaration

    def test_declarations_are_not_functions(self):
        assert analyze("int f(int);\nvoid g();").functions == []

    def test_methods_inside_class(self):
        a = analyze("class C {\npublic:\n    int size() const { return n; }\n    int n;\n};")
        assert a.classes == ["C"]
        assert a.functions == ["size"]
