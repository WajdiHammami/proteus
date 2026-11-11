import json

import pytest

from cli import commands
from cli.render import Style
from cli.session import CLIError, Session, Settings
from cli.shell import main, run_line
from src.adapters.models.source import SourceType
from tests.fakes.fake_llm import FakeLLM

Style.enabled = False

PAGE = """<html><head><title>Move semantics</title></head><body><article><h1>Move semantics in C++</h1>
<p>Moving lets you transfer resources instead of copying them, which matters for large buffers.</p>
<h2>A movable buffer</h2><p>Here is a buffer with a move constructor:</p>
<pre><code class="language-cpp">#include &lt;utility&gt;\n\nclass Buffer {\npublic:\n    int* data_;\n};</code></pre>
<p>And a usage example:</p><pre>Buffer a = make();\nBuffer b = std::move(a);</pre></article></body></html>"""


@pytest.fixture
def page(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(PAGE)
    return str(path)


@pytest.fixture
def shell():
    out = []
    answers = []
    session = Session(Settings(), out=out.append, input_fn=lambda prompt: answers.pop(0), interactive=True)
    session._llm = FakeLLM([{"description": "A movable buffer class.", "concepts": ["move semantics"]}] * 10)

    def run(line):
        out.clear()
        keep_going, ok = run_line(session, line)
        return "\n".join(out), ok

    run.session, run.answers = session, answers
    return run


class TestBasics:
    def test_help_lists_commands(self, shell):
        text, ok = shell("/help")
        assert ok and "/snippets" in text and "/extract" in text

    def test_command_help(self, shell):
        text, ok = shell("/extract --help")
        assert ok and "--no-llm" in text

    def test_unknown_command(self, shell):
        text, ok = shell("/nope")
        assert not ok and "Unknown command" in text

    def test_exit(self, shell):
        assert commands.execute(shell.session, "/exit") is False

    def test_comments_and_blank_lines(self, shell):
        assert shell("")[1] and shell("# note")[1]


class TestStages:
    def test_snippets_table(self, shell, page):
        text, ok = shell(f"/snippets {page}")
        assert ok and "A movable buffer" in text and "2 snippet(s)" in text

    def test_bare_path_runs_snippets(self, shell, page):
        text, _ = shell(page)
        assert "2 snippet(s)" in text

    def test_show(self, shell, page):
        shell(f"/snippets {page}")
        text, ok = shell("/show 0")
        assert ok and "#include <utility>" in text and "<utility>" in text

    def test_show_without_run(self, shell):
        text, ok = shell("/show 0")
        assert not ok and "/snippets" in text

    def test_show_out_of_range(self, shell, page):
        shell(f"/snippets {page}")
        assert "out of range" in shell("/show 9")[0]

    def test_fetch_and_normalize(self, shell, page):
        assert "text/html" in shell(f"/fetch {page}")[0]
        text, _ = shell(f"/normalize {page}")
        assert "TrafilaturaHtmlNormalizer" in text and "```cpp" in text

    def test_missing_file(self, shell):
        text, ok = shell("/snippets /no/such/file.pdf")
        assert not ok and "No such file" in text


class TestExtract:
    def test_no_llm(self, shell, page):
        text, ok = shell(f"/extract {page} --no-llm")
        assert ok and "2 record(s)" in text
        assert shell.session._llm.requests == []

    def test_confirmed_fills(self, shell, page):
        shell.answers.append("y")
        text, ok = shell(f"/extract {page}")
        assert ok and "cost estimate" in text and "LLM filling done" in text
        record = json.loads(shell("/record 0")[0])
        assert record["description"] == "A movable buffer class."

    def test_declined_spends_nothing(self, shell, page):
        shell.answers.append("n")
        text, _ = shell(f"/extract {page}")
        assert "skipped: declined" in text
        assert shell.session._llm.requests == []

    def test_debug_skips_question(self, shell, page):
        shell("/debug on")
        text, _ = shell(f"/extract {page}")
        assert "cost estimate" not in text and "LLM filling done" in text

    def test_estimate_makes_no_call(self, shell, page):
        text, ok = shell(f"/estimate {page}")
        assert ok and "calls:          2" in text
        assert shell.session._llm.requests == []

    def test_records_rejected_save(self, shell, page, tmp_path):
        shell(f"/extract {page} --no-llm")
        assert "2 record(s)" in shell("/records")[0]
        assert "No rejected" in shell("/rejected")[0]
        target = tmp_path / "out.json"
        assert shell(f"/save {target}")[1]
        assert len(json.loads(target.read_text())) == 2
        text, ok = shell(f"/save {target}")
        assert not ok and "--force" in text
        assert shell(f"/save {target} --force")[1]

    def test_records_before_extract(self, shell):
        assert "/extract" in shell("/records")[0]


class TestSchemaAndSettings:
    def test_schema_table(self, shell):
        text, ok = shell("/schema")
        assert ok and "description" in text and "llm" in text and "dataclass CppSnippet(" in text

    def test_unsafe_schema_rejected(self, shell, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("name: s\nfields:\n  - {name: __init__, type: str}\n")
        text, ok = shell(f"/schema {bad}")
        assert not ok and "__init__" in text

    def test_set_validates(self, shell, tmp_path):
        assert not shell("/set lines zero")[1]
        assert shell("/set lines 5")[1] and shell.session.settings.lines == 5
        bad = tmp_path / "bad.yaml"
        bad.write_text("name: s\nfields: []\n")
        assert not shell(f"/set schema {bad}")[1]
        assert shell.session.settings.schema == "schemas/cpp_snippet.yaml"

    def test_config_never_shows_key(self, shell):
        shell.session._llm = None   # load the real azure.env if present
        text, _ = shell("/config")
        assert "schema" in text
        import os
        if os.path.isfile("azure.env"):
            from dotenv import dotenv_values
            key = dotenv_values("azure.env").get("AZURE_OPENAI_API_KEY")
            assert key and key not in text

    def test_llm_test_asks_first(self, shell):
        shell.answers.append("n")
        text, _ = shell("/llm test")
        assert "Cancelled" in text and shell.session._llm.requests == []


class TestSources:
    @pytest.mark.parametrize("target, expected", [
        ("https://example.com/page", SourceType.URL),
        ("https://example.com/book.pdf", SourceType.PDF),
        ("https://example.com/sitemap.xml", SourceType.SITEMAP),
        ("https://example.com/sitemap_index.xml.gz", SourceType.SITEMAP),
        ("https://example.com/post-sitemap", SourceType.SITEMAP),
    ])
    def test_guessed_types(self, target, expected):
        assert Session(out=lambda s: None).make_source(target).type == expected

    def test_local_file(self, page):
        assert Session(out=lambda s: None).make_source(page).type == SourceType.FILE

    def test_forced_type(self):
        assert Session(out=lambda s: None).make_source("https://example.com/download?id=1", "pdf").type == SourceType.PDF

    def test_directory_rejected(self, tmp_path):
        with pytest.raises(CLIError, match="directory"):
            Session(out=lambda s: None).make_source(str(tmp_path))

    def test_sitemap_refused_by_single_source_commands(self, shell):
        text, ok = shell("/snippets https://example.com/sitemap.xml")
        assert not ok and "/sitemap" in text


class TestOneShot:
    def test_exit_codes(self, page, capsys):
        assert main(["snippets", page]) == 0
        assert "2 snippet(s)" in capsys.readouterr().out
        assert main(["show", "0"]) == 1
