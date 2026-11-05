import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pypdf import PdfReader
from pypdf.generic import DictionaryObject

pdf_renderer = pytest.importorskip(
    "uipath_langchain.agent.tools.internal_tools.pdf_renderer",
    reason="needs the pdf extra",
)
render_markdown_pdf = pdf_renderer.render_markdown_pdf


def pdf_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() for page in reader.pages)


def page_count(data: bytes) -> int:
    return len(PdfReader(io.BytesIO(data)).pages)


def as_dict(value: Any) -> DictionaryObject:
    resolved = value.get_object()
    assert isinstance(resolved, DictionaryObject)
    return resolved


def fonts_used(data: bytes) -> set[str]:
    names = set()
    for page in PdfReader(io.BytesIO(data)).pages:
        for font in as_dict(as_dict(page["/Resources"])["/Font"]).values():
            names.add(str(as_dict(font)["/BaseFont"]).split("+")[-1])
    return names


def link_targets(data: bytes) -> list[str]:
    targets = []
    for page in PdfReader(io.BytesIO(data)).pages:
        for annotation in page.get("/Annots") or []:
            action = as_dict(annotation).get("/A")
            if action is not None and "/URI" in as_dict(action):
                targets.append(str(as_dict(action)["/URI"]))
    return targets


def right_edges(data: bytes) -> list[float]:
    edges = []

    def visit(text: str, cm: Any, tm: Any, font: Any, size: float) -> None:
        drawn = text.rstrip("\n")
        if font is not None and drawn:
            advance = 1.0 if "-cjk-" in str(font["/BaseFont"]) else 0.6
            edges.append(cm[4] + tm[4] + len(drawn) * advance * size)

    for page in PdfReader(io.BytesIO(data)).pages:
        page.extract_text(visitor_text=visit)
    return edges


RIGHT_MARGIN_X = pdf_renderer.A4[0] - pdf_renderer._PAGE_MARGIN
CODE_PADDING = pdf_renderer._CODE_PADDING


@pytest.mark.usefixtures("pdf_fonts")
class TestStructure:
    def test_the_result_is_a_pdf(self) -> None:
        rendered = render_markdown_pdf("Hello")

        assert rendered.content.startswith(b"%PDF-")
        assert page_count(rendered.content) == 1

    def test_an_empty_document_is_one_blank_page(self) -> None:
        rendered = render_markdown_pdf("")

        assert page_count(rendered.content) == 1
        assert rendered.missing == []

    def test_markdown_syntax_becomes_layout_not_literal_text(self) -> None:
        markdown = "# Quarterly report\n\nRevenue is **up** and _steady_.\n"

        text = pdf_text(render_markdown_pdf(markdown).content)

        assert "Quarterly report" in text
        assert "Revenue is up and steady." in text
        assert "#" not in text
        assert "**" not in text

    def test_list_items_keep_their_order(self) -> None:
        text = pdf_text(
            render_markdown_pdf("- first\n- second\n\n1. one\n2. two\n").content
        )

        assert text.index("first") < text.index("second")
        assert text.index("one") < text.index("two")

    def test_block_elements_render_their_text(self) -> None:
        markdown = (
            "> quoted words\n\n---\n\n###### small heading\n\n"
            "line one  \nline two\nsame line\n\n~~gone~~ kept\n\n3. three\n4. four\n"
        )

        text = pdf_text(render_markdown_pdf(markdown).content)

        assert "quoted words" in text
        assert "small heading" in text
        assert "line one\nline two same line" in text
        assert "gone kept" in text
        assert "~~" not in text
        assert text.index("3") < text.index("three")

    def test_table_cells_are_rendered(self) -> None:
        markdown = "| Region | Revenue |\n|---|---|\n| Europe | 12 |\n| US | 34 |\n"

        text = pdf_text(render_markdown_pdf(markdown).content)

        for cell in ("Region", "Revenue", "Europe", "12", "US", "34"):
            assert cell in text
        assert "|" not in text

    def test_a_table_cell_taller_than_a_page_splits_across_pages(self) -> None:
        markdown = "| Note |\n|---|\n| " + "word " * 3000 + "|\n"

        rendered = render_markdown_pdf(markdown)

        assert page_count(rendered.content) > 1
        assert pdf_text(rendered.content).count("word") == 3000

    def test_code_keeps_its_lines_and_indentation(self) -> None:
        markdown = "Run `make all`.\n\n```\nfor x in y:\n    print(x)\n```\n"

        text = pdf_text(render_markdown_pdf(markdown).content)

        assert "make all" in text
        assert "for x in y:\n    print(x)" in text

    def test_tabs_in_code_become_spaces(self) -> None:
        text = pdf_text(render_markdown_pdf("```\nname\tvalue\n```\n").content)

        assert "name    value" in text
        assert "\x00" not in text

    def test_long_code_lines_wrap_inside_the_page(self) -> None:
        line = ", ".join(f"col_{i}" for i in range(40))

        text = pdf_text(render_markdown_pdf(f"```\n{line}\n```\n").content)

        lines = [part for part in text.split("\n") if part]
        assert max(len(part) for part in lines) <= 84
        assert "".join(lines) == line

    @pytest.mark.parametrize(
        "markdown",
        [
            "```\n{line}\n```\n",
            "> ```\n> {line}\n> ```\n",
            "- item\n\n  ```\n  {line}\n  ```\n",
            "1. a\n\n   1. b\n\n      1. c\n\n         ```\n         {line}\n         ```\n",
            "> 1. a\n>\n>    ```\n>    {line}\n>    ```\n",
        ],
        ids=["page", "quote", "list", "nested-lists", "list-in-quote"],
    )
    def test_code_boxes_stay_inside_the_right_margin(self, markdown: str) -> None:
        line = "x" * 200

        rendered = render_markdown_pdf(markdown.format(line=line))

        assert max(right_edges(rendered.content)) + CODE_PADDING <= RIGHT_MARGIN_X
        assert pdf_text(rendered.content).count("x") == 200

    @pytest.mark.parametrize(
        ("markdown", "expected"),
        [
            ("```\n中\t文\n```\n", "中   文"),
            ("    中\t文\n", "中   文"),
            ("`中文`", "中文"),
        ],
        ids=["fence", "indented", "inline"],
    )
    def test_cjk_only_code_is_drawn(self, markdown: str, expected: str) -> None:
        rendered = render_markdown_pdf(markdown)

        assert expected in pdf_text(rendered.content)
        assert rendered.missing == []

    def test_markup_characters_in_text_survive(self) -> None:
        text = pdf_text(render_markdown_pdf("If a < b & c > d then.").content)

        assert "If a < b & c > d then." in text

    def test_links_keep_their_text_and_only_web_links_are_clickable(self) -> None:
        rendered = render_markdown_pdf(
            "See [the docs](https://example.com/x) and [notes](./notes.md)."
        )

        assert "See the docs and notes." in pdf_text(rendered.content)
        assert link_targets(rendered.content) == ["https://example.com/x"]

    def test_web_link_schemes_match_in_any_case(self) -> None:
        rendered = render_markdown_pdf(
            "See [the docs](HTTPS://example.com/x) and [run](javascript:alert(1))."
        )

        assert link_targets(rendered.content) == ["HTTPS://example.com/x"]

    def test_an_image_is_replaced_by_its_alt_text(self) -> None:
        text = pdf_text(
            render_markdown_pdf(
                "Chart: ![Revenue chart](https://example.com/c.png)"
            ).content
        )

        assert "Revenue chart" in text

    def test_a_long_document_spans_pages(self) -> None:
        markdown = "\n\n".join(f"Paragraph {i}" for i in range(200))

        assert page_count(render_markdown_pdf(markdown).content) > 1


class TestFonts:
    @pytest.mark.usefixtures("pdf_fonts")
    def test_romanian_letters_use_the_latin_system_font(self) -> None:
        rendered = render_markdown_pdf("Venituri în creștere.")

        assert "Venituri în creștere." in pdf_text(rendered.content)
        assert rendered.missing == []

    @pytest.mark.usefixtures("pdf_fonts")
    def test_cjk_text_renders_from_the_cff_collection(self) -> None:
        rendered = render_markdown_pdf("報告書 한국어 中文报告")

        assert "報告書 한국어 中文报告" in pdf_text(rendered.content)
        assert rendered.missing == []

    @pytest.mark.usefixtures("pdf_fonts")
    def test_mixed_scripts_in_one_line(self) -> None:
        rendered = render_markdown_pdf("Raport 報告 creștere")

        assert "Raport 報告 creștere" in pdf_text(rendered.content)
        assert rendered.missing == []

    @pytest.mark.usefixtures("pdf_fonts")
    def test_cjk_inside_bold_and_italic_text_renders(self) -> None:
        rendered = render_markdown_pdf("**報告 total** and *中文 note*")

        assert "報告 total and 中文 note" in pdf_text(rendered.content)
        assert rendered.missing == []

    @pytest.mark.usefixtures("pdf_fonts")
    @pytest.mark.parametrize("markdown", ["中 &#x5831; B", "A &#x5831; B"])
    def test_characters_from_entities_are_drawn(self, markdown: str) -> None:
        rendered = render_markdown_pdf(markdown)

        assert "報" in pdf_text(rendered.content)
        assert "\x00" not in pdf_text(rendered.content)
        assert rendered.missing == []

    @pytest.mark.usefixtures("pdf_fonts")
    def test_bold_text_uses_the_bold_face(self) -> None:
        bold = render_markdown_pdf("**strong** words")
        heading = render_markdown_pdf("# Title")
        plain = render_markdown_pdf("plain words")

        assert {"TestSans-Bold", "TestSans-Regular"} <= fonts_used(bold.content)
        assert "TestSans-Bold" in fonts_used(heading.content)
        assert "TestSans-Bold" not in fonts_used(plain.content)

    def test_code_uses_the_monospace_font_and_body_text_does_not(
        self, install_pdf_fonts: Callable[..., Path]
    ) -> None:
        install_pdf_fonts(mono=True)

        with_code = render_markdown_pdf("Body `inline`\n\n```\nblock\n```\n")
        body_only = render_markdown_pdf("Body only")

        assert {"TestMono-Regular", "TestSans-Regular"} <= fonts_used(with_code.content)
        assert "TestMono-Regular" not in fonts_used(body_only.content)

    def test_code_with_symbols_from_the_cjk_font_stays_inside_the_margin(
        self, install_pdf_fonts: Callable[..., Path]
    ) -> None:
        install_pdf_fonts(mono=True)

        rendered = render_markdown_pdf("```\n" + "★" * 90 + "\n```\n")

        assert max(right_edges(rendered.content)) + CODE_PADDING <= RIGHT_MARGIN_X
        assert pdf_text(rendered.content).count("★") == 90

    def test_fonts_in_nested_folders_are_found(
        self, install_pdf_fonts: Callable[..., Path]
    ) -> None:
        root = install_pdf_fonts()
        nested = root / "truetype" / "liberation"
        nested.mkdir(parents=True)
        for font in root.glob("*.tt?"):
            font.rename(nested / font.name)

        rendered = render_markdown_pdf("creștere 報告")

        assert "TestSans-Regular" in fonts_used(rendered.content)
        assert rendered.missing == []

    def test_fonts_are_found_across_font_folders(
        self,
        install_pdf_fonts: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        latin = install_pdf_fonts(latin=True, cjk=False)
        cjk = install_pdf_fonts(latin=False, cjk=True)
        monkeypatch.setattr(pdf_renderer, "FONT_DIRS", (latin, cjk))

        rendered = render_markdown_pdf("creștere 報告")

        assert rendered.missing == []

    def test_a_missing_bold_font_falls_back_to_the_regular_one(
        self, install_pdf_fonts: Callable[..., Path]
    ) -> None:
        root = install_pdf_fonts()
        (root / "LiberationSans-Bold.ttf").unlink()

        rendered = render_markdown_pdf("**creștere**")

        assert "creștere" in pdf_text(rendered.content)
        assert "TestSans-Regular" in fonts_used(rendered.content)
        assert rendered.missing == []

    @pytest.mark.usefixtures("pdf_fonts")
    def test_characters_no_font_covers_are_reported_once(self) -> None:
        rendered = render_markdown_pdf("Thai ก and ก again, rocket 🚀")

        assert rendered.content.startswith(b"%PDF-")
        assert "Thai" in pdf_text(rendered.content)
        assert rendered.missing == ["ก", "🚀"]

    @pytest.mark.usefixtures("pdf_fonts")
    def test_invisible_characters_are_not_reported_missing(self) -> None:
        rendered = render_markdown_pdf("ok 🚀\ufe0f a\u200db\tc")

        assert rendered.missing == ["🚀"]

    def test_without_system_fonts_basic_text_still_renders(
        self, install_pdf_fonts: Callable[..., Path]
    ) -> None:
        install_pdf_fonts(latin=False, cjk=False)

        rendered = render_markdown_pdf("Revenue up, creștere 報告")

        assert "Revenue up" in pdf_text(rendered.content)
        assert rendered.missing == ["ș", "報", "告"]

    def test_without_latin_fonts_latin_text_uses_the_base_font_not_cjk(
        self, install_pdf_fonts: Callable[..., Path]
    ) -> None:
        install_pdf_fonts(latin=False)

        rendered = render_markdown_pdf("Revenue 報告")

        assert "Revenue 報告" in pdf_text(rendered.content)
        assert rendered.missing == []

    def test_without_cjk_fonts_only_cjk_is_missing(
        self, install_pdf_fonts: Callable[..., Path]
    ) -> None:
        install_pdf_fonts(cjk=False)

        rendered = render_markdown_pdf("creștere 報告")

        assert "creștere" in pdf_text(rendered.content)
        assert rendered.missing == ["報", "告"]


@pytest.mark.usefixtures("pdf_fonts")
class TestRepeatedRenders:
    def test_each_document_gets_its_own_cjk_glyphs(self) -> None:
        render_markdown_pdf("報告")

        rendered = render_markdown_pdf("中文")

        assert "中文" in pdf_text(rendered.content)

    def test_bold_cjk_in_a_later_document_is_drawn(self) -> None:
        render_markdown_pdf("**報告**")

        rendered = render_markdown_pdf("**中文**")

        assert "中文" in pdf_text(rendered.content)
