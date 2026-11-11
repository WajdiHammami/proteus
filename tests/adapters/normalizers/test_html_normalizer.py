import copy
import logging

import pytest

from src.adapters.models.normalized_document import TextFormat
from src.adapters.models.raw_document import RawDocument
from src.adapters.normalizers import html_normalizer
from src.adapters.normalizers.html_normalizer import ExtractionMode, TrafilaturaHtmlNormalizer
from src.ports.normalization_port import UnsupportedDocumentError


PAGE_URL = "https://example.com/blog/widget"

PARAGRAPH = (
    "The widget is built from recycled aluminium and has been tested for over ten "
    "thousand cycles in our lab, which makes it one of the most durable widgets available."
)

ARTICLE_HTML = f"""<!doctype html>
<html lang="en-GB">
<head>
  <title>Best Widget Review - Widget Weekly</title>
  <meta name="author" content="Jane Doe">
  <meta name="description" content="An in-depth review of the best widget">
  <meta property="og:title" content="Best Widget Review">
  <meta property="og:type" content="article">
  <meta property="og:image" content="https://example.com/a.jpg">
  <meta property="og:image" content="https://example.com/b.jpg">
  <meta property="article:published_time" content="2026-03-01">
  <meta name="twitter:card" content="summary">
  <meta name="viewport" content="width=device-width">
  <script type="application/ld+json">
    {{"@context": "https://schema.org", "@type": "Article", "headline": "Best Widget Review"}}
  </script>
</head>
<body>
  <nav><a href="/">Home</a> | <a href="/about">About us</a> | <a href="/shop">Shop</a></nav>
  <article>
    <h1>Best Widget Review</h1>
    <p>{PARAGRAPH}</p>
    <p>{PARAGRAPH} Read <a href="/docs/spec.pdf">the full specification</a> for details.</p>
    <h2>Specifications</h2>
    <table>
      <tr><th>Spec</th><th>Value</th></tr>
      <tr><td>Weight</td><td>2kg</td></tr>
      <tr><td>Colour</td><td>Silver</td></tr>
    </table>
  </article>
  <footer>Copyright 2026 Widget Weekly. All rights reserved. Privacy policy.</footer>
</body>
</html>"""


def raw_html(content: str, mime_type: str = "text/html", **metadata) -> RawDocument:
    return RawDocument(
        id="raw-1",
        source_id="src-1",
        content=content,
        mime_type=mime_type,
        metadata={"url": PAGE_URL, **metadata},
    )


@pytest.fixture
def normalizer():
    return TrafilaturaHtmlNormalizer()


@pytest.fixture
def article_doc(normalizer):
    return normalizer.normalize(raw_html(ARTICLE_HTML))


# --------------------------------------------------------------------------- #
# supports()
# --------------------------------------------------------------------------- #

class TestSupports:
    @pytest.mark.parametrize("mime", [
        "text/html", "application/xhtml+xml", "TEXT/HTML", "text/html; charset=utf-8", " text/html ",
    ])
    def test_html_mime_types(self, normalizer, mime):
        assert normalizer.supports(raw_html("", mime_type=mime))

    @pytest.mark.parametrize("mime", ["application/pdf", "text/plain", "text/markdown", "", "text/htmlx"])
    def test_other_mime_types(self, normalizer, mime):
        assert not normalizer.supports(raw_html("", mime_type=mime))

    def test_normalize_rejects_unsupported(self, normalizer):
        with pytest.raises(UnsupportedDocumentError, match="application/pdf"):
            normalizer.normalize(raw_html("%PDF-1.4", mime_type="application/pdf"))


# --------------------------------------------------------------------------- #
# Main job: clean markdown content
# --------------------------------------------------------------------------- #

class TestContent:
    def test_document_fields(self, article_doc):
        assert article_doc.id
        assert article_doc.raw_document_id == "raw-1"
        assert article_doc.source_id == "src-1"
        assert article_doc.format == TextFormat.MARKDOWN

    def test_main_content_kept(self, article_doc):
        assert "# Best Widget Review" in article_doc.content
        assert "recycled aluminium" in article_doc.content
        assert article_doc.metadata["extraction_mode"] == ExtractionMode.MAIN

    def test_boilerplate_removed(self, article_doc):
        assert "About us" not in article_doc.content
        assert "All rights reserved" not in article_doc.content

    def test_tables_kept_as_markdown(self, article_doc):
        assert "| Weight | 2kg |" in article_doc.content
        assert "| Colour | Silver |" in article_doc.content

    def test_tables_can_be_disabled(self):
        doc = TrafilaturaHtmlNormalizer(include_tables=False).normalize(raw_html(ARTICLE_HTML))
        assert "| Weight |" not in doc.content

    def test_links_kept_and_resolved_against_page_url(self, article_doc):
        assert "[the full specification](https://example.com/docs/spec.pdf)" in article_doc.content

    def test_links_can_be_disabled(self):
        doc = TrafilaturaHtmlNormalizer(include_links=False).normalize(raw_html(ARTICLE_HTML))
        assert "the full specification" in doc.content
        assert "](https://" not in doc.content

    def test_text_is_cleaned(self, normalizer):
        html = ARTICLE_HTML.replace("recycled aluminium", "recycled alu​minium")
        doc = normalizer.normalize(raw_html(html))
        assert "recycled aluminium" in doc.content
        assert "\n\n\n" not in doc.content

    def test_each_call_gets_new_id(self, normalizer):
        raw = raw_html(ARTICLE_HTML)
        assert normalizer.normalize(raw).id != normalizer.normalize(raw).id

    def test_raw_document_is_not_mutated(self, normalizer):
        raw = raw_html(ARTICLE_HTML, extra={"nested": [1, 2]})
        before = copy.deepcopy(raw)
        normalizer.normalize(raw)
        assert raw == before

    def test_metadata_does_not_alias_raw_document(self, normalizer):
        raw = raw_html(ARTICLE_HTML, extra={"nested": [1, 2]})
        doc = normalizer.normalize(raw)
        doc.metadata["extra"]["nested"].append(3)
        assert raw.metadata["extra"]["nested"] == [1, 2]

    def test_raw_metadata_is_carried_forward(self, normalizer):
        doc = normalizer.normalize(raw_html(ARTICLE_HTML, crawl_depth=2))
        assert doc.metadata["url"] == PAGE_URL
        assert doc.metadata["crawl_depth"] == 2

    def test_works_without_url(self, normalizer):
        raw = RawDocument(id="r", source_id="s", content=ARTICLE_HTML, metadata={"path": "/tmp/page.html"})
        doc = normalizer.normalize(raw)
        assert "recycled aluminium" in doc.content
        assert doc.metadata["path"] == "/tmp/page.html"

    def test_xml_declaration_is_handled(self, normalizer):
        doc = normalizer.normalize(raw_html('<?xml version="1.0" encoding="utf-8"?>\n' + ARTICLE_HTML))
        assert "recycled aluminium" in doc.content

    def test_non_latin_content(self, normalizer):
        arabic = "هذه فقرة طويلة باللغة العربية تصف المنتج بالتفصيل وتشرح مزاياه واستخداماته المختلفة. " * 3
        html = f'<html lang="ar"><body><article><h1>منتج</h1><p>{arabic}</p></article></body></html>'
        doc = normalizer.normalize(raw_html(html))
        assert "باللغة العربية" in doc.content
        assert doc.metadata["page_metadata"]["language"] == "ar"


# --------------------------------------------------------------------------- #
# Fallback cascade
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_trafilatura(monkeypatch):
    """Controls what each cascade step returns: main / recall (extract) and full_text (html2txt)."""
    calls = []

    def install(main=None, recall=None, full_text="", main_error=None):
        def _extract(html, favor_recall=False, **kwargs):
            step = "recall" if favor_recall else "main"
            calls.append(step)
            if step == "main" and main_error:
                raise main_error
            return recall if favor_recall else main

        def _html2txt(html, **kwargs):
            calls.append("full_text")
            return full_text

        monkeypatch.setattr(html_normalizer.trafilatura, "extract", _extract)
        monkeypatch.setattr(html_normalizer.trafilatura, "html2txt", _html2txt)
        return calls

    return install


LONG = "x" * 150
SHORT = "short bit"


class TestFallback:
    def test_main_accepted_when_long_enough(self, normalizer, fake_trafilatura):
        calls = fake_trafilatura(main=LONG, recall="unused", full_text="unused")
        doc = normalizer.normalize(raw_html("<html><body>x</body></html>"))
        assert doc.content == LONG
        assert doc.metadata["extraction_mode"] == ExtractionMode.MAIN
        assert calls == ["main"]

    def test_recall_used_when_main_is_none(self, normalizer, fake_trafilatura):
        calls = fake_trafilatura(main=None, recall=LONG)
        doc = normalizer.normalize(raw_html("<html><body>x</body></html>"))
        assert doc.metadata["extraction_mode"] == ExtractionMode.RECALL
        assert calls == ["main", "recall"]

    def test_recall_used_when_main_too_short(self, normalizer, fake_trafilatura):
        fake_trafilatura(main=SHORT, recall=LONG)
        doc = normalizer.normalize(raw_html("<html><body>x</body></html>"))
        assert doc.content == LONG
        assert doc.metadata["extraction_mode"] == ExtractionMode.RECALL

    def test_full_text_used_as_last_resort(self, normalizer, fake_trafilatura):
        calls = fake_trafilatura(main=None, recall=None, full_text=LONG)
        doc = normalizer.normalize(raw_html("<html><body>x</body></html>"))
        assert doc.metadata["extraction_mode"] == ExtractionMode.FULL_TEXT
        assert calls == ["main", "recall", "full_text"]

    def test_longest_short_candidate_kept_when_none_reach_threshold(self, normalizer, fake_trafilatura):
        fake_trafilatura(main="tiny", recall="a bit longer text", full_text="mid text")
        doc = normalizer.normalize(raw_html("<html><body>x</body></html>"))
        assert doc.content == "a bit longer text"
        assert doc.metadata["extraction_mode"] == ExtractionMode.RECALL

    def test_tie_prefers_earlier_step(self, normalizer, fake_trafilatura):
        fake_trafilatura(main="same", recall="same", full_text="same")
        doc = normalizer.normalize(raw_html("<html><body>x</body></html>"))
        assert doc.metadata["extraction_mode"] == ExtractionMode.MAIN

    def test_all_empty_gives_empty_mode(self, normalizer, fake_trafilatura):
        fake_trafilatura(main=None, recall="", full_text="   ")
        doc = normalizer.normalize(raw_html("<html><body>x</body></html>"))
        assert doc.content == ""
        assert doc.is_empty()
        assert doc.metadata["extraction_mode"] == ExtractionMode.EMPTY

    def test_step_exception_falls_through(self, normalizer, fake_trafilatura):
        fake_trafilatura(main_error=RuntimeError("lxml exploded"), recall=LONG)
        doc = normalizer.normalize(raw_html("<html><body>x</body></html>"))
        assert doc.metadata["extraction_mode"] == ExtractionMode.RECALL

    def test_threshold_is_configurable(self, fake_trafilatura):
        fake_trafilatura(main=SHORT, recall=LONG)
        doc = TrafilaturaHtmlNormalizer(min_chars=5).normalize(raw_html("<html><body>x</body></html>"))
        assert doc.metadata["extraction_mode"] == ExtractionMode.MAIN

    def test_fallback_is_logged(self, normalizer, fake_trafilatura, caplog):
        fake_trafilatura(main=None, recall=LONG)
        with caplog.at_level(logging.INFO, logger=html_normalizer.__name__):
            normalizer.normalize(raw_html("<html><body>x</body></html>"))
        assert any("extraction_mode=recall" in r.getMessage() for r in caplog.records)

    def test_no_log_on_main(self, normalizer, fake_trafilatura, caplog):
        fake_trafilatura(main=LONG)
        with caplog.at_level(logging.INFO, logger=html_normalizer.__name__):
            normalizer.normalize(raw_html("<html><body>x</body></html>"))
        assert caplog.records == []


class TestEmptyInputs:
    @pytest.mark.parametrize("html", ["", "   \n ", "<html><body></body></html>", "<html></html>"])
    def test_empty_pages_are_valid_empty_documents(self, normalizer, html):
        doc = normalizer.normalize(raw_html(html))
        assert doc.content == ""
        assert doc.metadata["extraction_mode"] == ExtractionMode.EMPTY
        assert doc.metadata["structured_data"] == {"json_ld": [], "opengraph": {}}

    def test_blank_content_skips_trafilatura(self, normalizer, fake_trafilatura):
        calls = fake_trafilatura(main=LONG)
        normalizer.normalize(raw_html("   "))
        assert calls == []


# --------------------------------------------------------------------------- #
# Page metadata
# --------------------------------------------------------------------------- #

class TestPageMetadata:
    def test_basic_fields(self, article_doc):
        meta = article_doc.metadata["page_metadata"]
        assert meta["title"] == "Best Widget Review"
        assert meta["author"] == "Jane Doe"
        assert meta["description"] == "An in-depth review of the best widget"
        assert meta["language"] == "en-GB"
        assert meta["date"] == "2026-03-01"

    def test_missing_fields_are_omitted_not_none(self, normalizer):
        html = f"<html><body><article><p>{PARAGRAPH}</p></article></body></html>"
        meta = normalizer.normalize(raw_html(html)).metadata["page_metadata"]
        assert "author" not in meta
        assert "language" not in meta
        assert None not in meta.values()

    def test_blank_lang_attribute_ignored(self, normalizer):
        html = f'<html lang="  "><body><p>{PARAGRAPH}</p></body></html>'
        assert "language" not in normalizer.normalize(raw_html(html)).metadata["page_metadata"]

    def test_metadata_failure_does_not_break_normalization(self, normalizer, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("metadata crashed")

        monkeypatch.setattr(html_normalizer.trafilatura, "extract_metadata", boom)
        doc = normalizer.normalize(raw_html(ARTICLE_HTML))
        assert "recycled aluminium" in doc.content
        assert doc.metadata["page_metadata"] == {"language": "en-GB"}


# --------------------------------------------------------------------------- #
# Structured data: JSON-LD
# --------------------------------------------------------------------------- #

def page_with_scripts(*scripts: str, script_type: str = "application/ld+json") -> str:
    tags = "".join(f'<script type="{script_type}">{s}</script>' for s in scripts)
    return f"<html><head>{tags}</head><body><article><p>{PARAGRAPH}</p></article></body></html>"


def json_ld_of(normalizer, html: str) -> list:
    return normalizer.normalize(raw_html(html)).metadata["structured_data"]["json_ld"]


class TestJsonLd:
    def test_single_object(self, article_doc):
        assert article_doc.metadata["structured_data"]["json_ld"] == [{
            "@context": "https://schema.org", "@type": "Article", "headline": "Best Widget Review",
        }]

    def test_multiple_blocks_in_order(self, normalizer):
        html = page_with_scripts('{"@type": "Product", "name": "A"}', '{"@type": "BreadcrumbList"}')
        assert [i["@type"] for i in json_ld_of(normalizer, html)] == ["Product", "BreadcrumbList"]

    def test_top_level_array_is_flattened(self, normalizer):
        html = page_with_scripts('[{"@type": "Product"}, {"@type": "Offer"}]')
        assert [i["@type"] for i in json_ld_of(normalizer, html)] == ["Product", "Offer"]

    def test_graph_is_flattened(self, normalizer):
        html = page_with_scripts(
            '{"@context": "https://schema.org", "@graph": [{"@type": "WebPage"}, {"@type": "Product", "name": "W"}]}'
        )
        assert json_ld_of(normalizer, html) == [{"@type": "WebPage"}, {"@type": "Product", "name": "W"}]

    def test_nested_values_preserved(self, normalizer):
        html = page_with_scripts(
            '{"@type": "Product", "offers": {"@type": "Offer", "price": "9.99", "priceCurrency": "EUR"}}'
        )
        [product] = json_ld_of(normalizer, html)
        assert product["offers"]["price"] == "9.99"

    def test_malformed_block_skipped_others_kept(self, normalizer):
        html = page_with_scripts('{ "broken": ', '{"@type": "Product"}')
        assert json_ld_of(normalizer, html) == [{"@type": "Product"}]

    @pytest.mark.parametrize("payload", ["", "   ", "null", "42", '"just a string"'])
    def test_empty_or_scalar_payloads_ignored(self, normalizer, payload):
        assert json_ld_of(normalizer, page_with_scripts(payload)) == []

    @pytest.mark.parametrize("wrapped", [
        '<!--{"@type": "Product"}-->',
        '//<![CDATA[\n{"@type": "Product"}\n//]]>',
        '<![CDATA[{"@type": "Product"}]]>',
    ])
    def test_cms_wrappers_stripped(self, normalizer, wrapped):
        assert json_ld_of(normalizer, page_with_scripts(wrapped)) == [{"@type": "Product"}]

    def test_literal_newlines_inside_strings_tolerated(self, normalizer):
        html = page_with_scripts('{"@type": "Product", "description": "line one\nline two"}')
        assert json_ld_of(normalizer, html)[0]["description"] == "line one\nline two"

    @pytest.mark.parametrize("script_type", ["APPLICATION/LD+JSON", "application/ld+json; charset=utf-8"])
    def test_type_attribute_variants(self, normalizer, script_type):
        html = page_with_scripts('{"@type": "Product"}', script_type=script_type)
        assert json_ld_of(normalizer, html) == [{"@type": "Product"}]

    @pytest.mark.parametrize("script_type", ["application/json", "text/javascript"])
    def test_other_script_types_ignored(self, normalizer, script_type):
        html = page_with_scripts('{"@type": "Product"}', script_type=script_type)
        assert json_ld_of(normalizer, html) == []

    def test_json_ld_in_body_is_found(self, normalizer):
        html = (f'<html><body><article><p>{PARAGRAPH}</p></article>'
                '<script type="application/ld+json">{"@type": "Event"}</script></body></html>')
        assert json_ld_of(normalizer, html) == [{"@type": "Event"}]

    def test_json_ld_not_leaked_into_content(self, normalizer):
        html = page_with_scripts('{"@type": "Product", "name": "SecretSKU123"}')
        assert "SecretSKU123" not in normalizer.normalize(raw_html(html)).content


# --------------------------------------------------------------------------- #
# Structured data: OpenGraph / Twitter
# --------------------------------------------------------------------------- #

class TestOpenGraph:
    def test_collects_prefixed_tags(self, article_doc):
        og = article_doc.metadata["structured_data"]["opengraph"]
        assert og["og:title"] == "Best Widget Review"
        assert og["og:type"] == "article"
        assert og["article:published_time"] == "2026-03-01"
        assert og["twitter:card"] == "summary"

    def test_repeated_keys_become_list(self, article_doc):
        og = article_doc.metadata["structured_data"]["opengraph"]
        assert og["og:image"] == ["https://example.com/a.jpg", "https://example.com/b.jpg"]

    def test_three_repeats(self, normalizer):
        metas = "".join(f'<meta property="og:image" content="{i}.jpg">' for i in range(3))
        html = f"<html><head>{metas}</head><body><p>{PARAGRAPH}</p></body></html>"
        og = normalizer.normalize(raw_html(html)).metadata["structured_data"]["opengraph"]
        assert og["og:image"] == ["0.jpg", "1.jpg", "2.jpg"]

    def test_unrelated_meta_tags_ignored(self, article_doc):
        og = article_doc.metadata["structured_data"]["opengraph"]
        assert "viewport" not in og
        assert "author" not in og
        assert "description" not in og

    def test_empty_content_and_case(self, normalizer):
        html = ('<html><head><meta property="og:title" content="  "><meta property="OG:Site_Name" content="Shop">'
                f"</head><body><p>{PARAGRAPH}</p></body></html>")
        og = normalizer.normalize(raw_html(html)).metadata["structured_data"]["opengraph"]
        assert og == {"og:site_name": "Shop"}


# --------------------------------------------------------------------------- #
# C++ code blocks (primary target)
# --------------------------------------------------------------------------- #

INTRO = f"<p>{PARAGRAPH}</p>"


def code_page(body: str) -> str:
    return f"<html><body><article><h1>C++ guide</h1>{INTRO}{body}</article></body></html>"


def content_of(normalizer, body: str) -> str:
    return normalizer.normalize(raw_html(code_page(body))).content


CPP_ESCAPED = (
    "#include &lt;vector&gt;\n"
    "template &lt;typename T&gt;\n"
    "class Stack {\n"
    "public:\n"
    "    void push(const T&amp; v) { data_.push_back(v); }\n"
    "    T* top() { return &amp;data_.back(); }\n"
    "    int __count_ = 0; // _private_ *note*\n"
    "private:\n"
    "    std::vector&lt;T&gt; data_;\n"
    "};\n"
    "if (p &amp;&amp; p-&gt;next || a &lt;&lt; 2) { std::cout &lt;&lt; *p; }"
)
CPP_EXPECTED = [
    "#include <vector>",
    "template <typename T>",
    "class Stack {",
    "public:",
    "    void push(const T& v) { data_.push_back(v); }",
    "    T* top() { return &data_.back(); }",
    "    int __count_ = 0; // _private_ *note*",
    "private:",
    "    std::vector<T> data_;",
    "};",
    "if (p && p->next || a << 2) { std::cout << *p; }",
]


class TestCppCode:
    def test_pre_code_block_preserved_exactly(self, normalizer):
        content = content_of(normalizer, f'<pre><code class="language-cpp">{CPP_ESCAPED}</code></pre>')
        assert "```cpp\n" + "\n".join(CPP_EXPECTED) + "\n```" in content

    def test_no_markdown_escaping_inside_code(self, normalizer):
        content = content_of(normalizer, f"<pre><code>{CPP_ESCAPED}</code></pre>")
        assert "\\*" not in content and "\\_" not in content

    def test_syntax_highlighted_spans_reassembled(self, normalizer):
        body = ('<div class="highlight"><pre><span class="cp">#include</span> <span class="cpf">&lt;map&gt;</span>\n'
                '<span class="k">auto</span> <span class="n">m</span> <span class="o">=</span> '
                '<span class="n">std</span><span class="o">::</span><span class="n">map</span>'
                '<span class="o">&lt;</span><span class="kt">int</span><span class="p">,</span> '
                '<span class="kt">int</span><span class="o">&gt;</span><span class="p">{};</span></pre></div>')
        content = content_of(normalizer, body)
        assert "#include <map>\nauto m = std::map<int, int>{};" in content

    def test_inline_code_kept(self, normalizer):
        content = content_of(normalizer, "<p>Use <code>std::vector&lt;T&gt;</code> and <code>int *p</code> here.</p>")
        assert "`std::vector<T>`" in content
        assert "`int *p`" in content


class TestCodeTables:
    def test_pygments_line_number_table_unwrapped(self, normalizer):
        body = ('<table class="highlighttable"><tr><td class="linenos"><pre>1\n2\n3</pre></td>'
                '<td class="code"><pre>int x = 1;\nx *= 2;\nreturn x;\n</pre></td></tr></table>')
        content = content_of(normalizer, body)
        assert "```\nint x = 1;\nx *= 2;\nreturn x;\n```" in content
        assert "|" not in content
        assert "\\*" not in content

    def test_github_gist_table_unwrapped(self, normalizer):
        rows = "".join(
            f'<tr><td class="blob-num js-line-number" data-line-number="{n}"></td>'
            f'<td class="blob-code blob-code-inner js-file-line">{code}</td></tr>'
            for n, code in enumerate(["int main() {", "  int *p = &amp;x;", "  return *p;", "}"], start=1)
        )
        content = content_of(normalizer, f'<table class="highlight">{rows}</table>')
        assert "```\nint main() {\n  int *p = &x;\n  return *p;\n}\n```" in content
        assert "|" not in content

    def test_highlightjs_line_numbers_table_unwrapped(self, normalizer):
        rows = "".join(
            f'<tr><td class="hljs-ln-numbers"><div class="hljs-ln-n" data-line-number="{n}"></div></td>'
            f'<td class="hljs-ln-code"><div class="hljs-ln-line">{code}</div></td></tr>'
            for n, code in enumerate(["void f() {", "  g();", "}"], start=1)
        )
        content = content_of(normalizer, f'<pre><code><table class="hljs-ln">{rows}</table></code></pre>')
        assert "void f() {\n  g();\n}" in content

    def test_text_after_code_table_kept(self, normalizer):
        body = ('<table><tr><td><pre>1</pre></td><td><pre>int x;</pre></td></tr></table>'
                '<p>And this paragraph after the code block must remain in the output text.</p>')
        content = content_of(normalizer, body)
        assert "int x;" in content
        assert "must remain in the output" in content

    def test_data_table_with_pre_cells_not_unwrapped(self, normalizer):
        body = ("<table><tr><th>Method</th><th>Example</th></tr>"
                "<tr><td>size</td><td><pre>v.size()</pre></td></tr>"
                "<tr><td>count</td><td><pre>v.count()</pre></td></tr></table>")
        content = content_of(normalizer, body)
        assert "| size | v.size() |" in content

    def test_numeric_data_table_not_unwrapped(self, normalizer):
        body = ("<table><tr><th>Rank</th><th>Name</th></tr>"
                "<tr><td>1</td><td>Alice</td></tr><tr><td>2</td><td>Bob</td></tr></table>")
        assert "| 1 | Alice |" in content_of(normalizer, body)

    def test_cell_class_merely_containing_code_not_unwrapped(self, normalizer):
        body = ('<table><tr><td class="country-code">FR</td><td>France</td></tr>'
                '<tr><td class="country-code">DE</td><td>Germany</td></tr></table>')
        assert "| FR | France |" in content_of(normalizer, body)

    def test_unwrap_counts_tables(self):
        from src.adapters.normalizers.html_normalizer import _unwrap_code_tables
        from trafilatura.utils import load_html

        tree = load_html("<html><body><table><tr><td><pre>1</pre></td><td><pre>a;</pre></td></tr></table>"
                         "<table><tr><td>x</td><td>y</td></tr></table></body></html>")
        root = tree.getroottree().getroot()
        assert _unwrap_code_tables(root) == 1
        assert len(root.xpath("//table")) == 1
        assert root.xpath("//pre/code")[0].text == "a;"


class TestCodeBlockFidelity:
    """trafilatura alone drops blank lines, <br> breaks and the language; we restore them."""

    def test_blank_lines_inside_code_kept(self, normalizer):
        code = "#include &lt;vector&gt;\n\nint main() {\n    int x = 1;\n\n    return x;\n}"
        content = content_of(normalizer, f"<pre><code>{code}</code></pre>")
        assert "```\n#include <vector>\n\nint main() {\n    int x = 1;\n\n    return x;\n}\n```" in content

    def test_br_line_breaks_kept(self, normalizer):
        content = content_of(normalizer, "<pre><code>int a;<br>int b;<br><br>int c;</code></pre>")
        assert "int a;\nint b;\n\nint c;" in content

    def test_crlf_and_surrounding_blank_lines(self, normalizer):
        content = content_of(normalizer, "<pre><code>\n\r\nint a;\r\nint b;\r\n\r\n</code></pre>")
        assert "```\nint a;\nint b;\n```" in content

    @pytest.mark.parametrize("html, language", [
        ('<pre><code class="language-cpp">int x;</code></pre>', "cpp"),
        ('<pre><code class="hljs language-c++">int x;</code></pre>', "cpp"),
        ('<pre class="prettyprint lang-cpp">int x;</pre>', "cpp"),
        ('<pre class="brush: cpp; gutter: true">int x;</pre>', "cpp"),
        ('<pre class="sourceCode cpp"><code class="sourceCode cpp">int x;</code></pre>', "cpp"),
        ('<pre class="cpp geshi">int x;</pre>', "cpp"),
        ('<pre><code class="hljs cpp">int x;</code></pre>', "cpp"),
        ('<pre data-lang="C++">int x;</pre>', "cpp"),
        ('<div class="highlight-cpp notranslate"><div class="highlight"><pre>int x;</pre></div></div>', "cpp"),
        ('<pre><code class="language-c">int x;</code></pre>', "c"),
        ('<pre><code class="language-bash">int x;</code></pre>', "bash"),
        ('<pre><code class="language-cxx">int x;</code></pre>', "cpp"),
    ])
    def test_language_hint_kept(self, normalizer, html, language):
        assert f"```{language}\nint x;\n```" in content_of(normalizer, html)

    @pytest.mark.parametrize("html", [
        "<pre><code>int x;</code></pre>",
        '<pre><code class="language-none">int x;</code></pre>',
        '<pre><code class="nohighlight">int x;</code></pre>',
        '<pre class="highlight notranslate">int x;</pre>',   # bare classes are not languages
        '<pre class="python">int x;</pre>',                 # only C/C++ bare classes are trusted
    ])
    def test_no_language_when_unknown(self, normalizer, html):
        assert "```\nint x;\n```" in content_of(normalizer, html)

    def test_several_blocks_keep_order_and_languages(self, normalizer):
        body = ('<pre><code class="language-cpp">int a;</code></pre><p>Then run it from the shell like this:</p>'
                '<pre><code class="language-bash">g++ a.cpp</code></pre>')
        content = content_of(normalizer, body)
        assert content.index("```cpp\nint a;\n```") < content.index("```bash\ng++ a.cpp\n```")

    def test_nested_pre_is_one_block(self, normalizer):
        content = content_of(normalizer, "<pre>int a;\n<pre>int b;</pre></pre>")
        assert content.count("```") == 2

    def test_pre_in_table_cell_left_to_table(self, normalizer):
        body = ("<table><tr><th>Method</th><th>Example</th></tr>"
                "<tr><td>size</td><td><pre>v.size()</pre></td></tr></table>")
        content = content_of(normalizer, body)
        assert "| size | v.size() |" in content
        assert "```" not in content

    def test_unwrapped_code_table_gets_language(self, normalizer):
        body = ('<div class="highlight-cpp"><table class="highlighttable"><tr><td class="linenos"><pre>1\n2</pre></td>'
                '<td class="code"><pre>int x;\n\nreturn x;</pre></td></tr></table></div>')
        assert "```cpp\nint x;\n\nreturn x;\n```" in content_of(normalizer, body)

    def test_code_with_backticks_gets_longer_fence(self, normalizer):
        content = content_of(normalizer, '<pre><code class="language-markdown">```cpp\nx\n```</code></pre>')
        assert "````markdown\n```cpp\nx\n```\n````" in content

    def test_full_text_fallback_also_fences_code(self, monkeypatch):
        monkeypatch.setattr(html_normalizer.trafilatura, "extract", lambda html, **kwargs: None)
        normalizer = TrafilaturaHtmlNormalizer(min_chars=1)
        html = '<html><body><pre><code class="language-cpp">int a;\n\nint b;</code></pre></body></html>'
        doc = normalizer.normalize(raw_html(html))
        assert doc.metadata["extraction_mode"] == "full_text"
        assert "```cpp\nint a;\n\nint b;\n```" in doc.content

    @pytest.mark.parametrize("body", [
        '<pre><code class="language-cpp">int x;</code></pre>',
        "<nav><pre>menu</pre></nav>",
    ])
    def test_placeholders_never_leak(self, normalizer, body):
        assert "proteuscode" not in content_of(normalizer, body)

    def test_raw_document_is_not_mutated(self, normalizer):
        html = code_page('<pre><code class="language-cpp">int a;\n\nint b;</code></pre>')
        document = raw_html(html)
        normalizer.normalize(document)
        assert document.content == html
