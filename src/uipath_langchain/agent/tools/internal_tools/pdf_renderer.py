import argparse
import io
import json
import sys
import unicodedata
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlsplit
from xml.sax.saxutils import escape

from fontTools import subset  # type: ignore[import-untyped]
from fontTools.pens.cu2quPen import Cu2QuPen  # type: ignore[import-untyped]
from fontTools.pens.ttGlyphPen import TTGlyphPen  # type: ignore[import-untyped]
from fontTools.ttLib import TTFont as FontFile  # type: ignore[import-untyped]
from fontTools.ttLib import newTable
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    HRFlowable,
    ListFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    XPreformatted,
)

__all__ = [
    "EXIT_OUT_OF_MEMORY",
    "EXIT_PDF_TOO_LARGE",
    "FONT_DIRS",
    "RenderedPdf",
    "main",
    "render_markdown_pdf",
]

EXIT_OUT_OF_MEMORY = 3
EXIT_PDF_TOO_LARGE = 4

FONT_DIRS: tuple[Path, ...] = (Path("/usr/share/fonts"), Path("/usr/local/share/fonts"))


class Weight(NamedTuple):
    bold: bool = False
    italic: bool = False


REGULAR = Weight()
BOLD = Weight(bold=True)
ITALIC = Weight(italic=True)
BOLD_ITALIC = Weight(bold=True, italic=True)


@dataclass(frozen=True)
class FontSource:
    key: str
    files: dict[Weight, str]
    code_only: bool = False
    can_be_default: bool = True
    embed_subset: bool = False


FONT_SOURCES: tuple[FontSource, ...] = (
    FontSource(
        key="mono",
        files={
            REGULAR: "LiberationMono-Regular.ttf",
            BOLD: "LiberationMono-Bold.ttf",
            ITALIC: "LiberationMono-Italic.ttf",
            BOLD_ITALIC: "LiberationMono-BoldItalic.ttf",
        },
        code_only=True,
    ),
    FontSource(
        key="latin",
        files={
            REGULAR: "LiberationSans-Regular.ttf",
            BOLD: "LiberationSans-Bold.ttf",
            ITALIC: "LiberationSans-Italic.ttf",
            BOLD_ITALIC: "LiberationSans-BoldItalic.ttf",
        },
    ),
    FontSource(
        key="cjk",
        files={
            REGULAR: "NotoSansCJK-Regular.ttc",
            BOLD: "NotoSansCJK-Bold.ttc",
            ITALIC: "NotoSansCJK-Regular.ttc",
            BOLD_ITALIC: "NotoSansCJK-Bold.ttc",
        },
        can_be_default=False,
        embed_subset=True,
    ),
)


@dataclass(frozen=True)
class RenderedPdf:
    content: bytes
    missing: list[str]


_BASE_FONTS: dict[Weight, str] = {
    REGULAR: "Helvetica",
    BOLD: "Helvetica-Bold",
    ITALIC: "Helvetica-Oblique",
    BOLD_ITALIC: "Helvetica-BoldOblique",
}
_BASE_CODE_FONTS: dict[Weight, str] = {
    REGULAR: "Courier",
    BOLD: "Courier-Bold",
    ITALIC: "Courier-Oblique",
    BOLD_ITALIC: "Courier-BoldOblique",
}
_PAGE_MARGIN = 2 * cm
_BASE_SIZE = 10.5
_CODE_SIZE = 9.5
_TABLE_SIZE = 9.5
_CODE_PADDING = 4
_CODE_TAB_SIZE = 4
_FRAME_PADDING = 6
_LIST_INDENT = 0.6 * cm
_QUOTE_INDENT = 0.6 * cm
_HEADING_SIZES = {1: 18.0, 2: 15.0, 3: 13.0, 4: 11.5, 5: 11.0, 6: 10.5}
_SAFE_LINK_SCHEMES = frozenset({"http", "https", "mailto"})
_LIGHT_GREY = colors.HexColor("#f2f2f2")
_DRAWN_NODE_TYPES = ("text", "code_inline", "fence", "code_block")
_UNUSED_LAYOUT_TABLES = ["GSUB", "GPOS", "GDEF", "BASE"]


def render_markdown_pdf(
    markdown: str, font_dirs: tuple[Path, ...] | None = None
) -> RenderedPdf:
    tree = SyntaxTreeNode(
        MarkdownIt("commonmark", {"html": False})
        .enable(["table", "strikethrough"])
        .parse(markdown)
    )
    drawn_text = "".join(
        node.content.expandtabs(_CODE_TAB_SIZE)
        for node in tree.walk()
        if node.type in _DRAWN_NODE_TYPES
    )
    fonts = _Fonts(
        _available_sources(FONT_DIRS if font_dirs is None else font_dirs), drawn_text
    )
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=_PAGE_MARGIN,
        rightMargin=_PAGE_MARGIN,
        topMargin=_PAGE_MARGIN,
        bottomMargin=_PAGE_MARGIN,
    )
    width = document.width - 2 * _FRAME_PADDING
    story = _Blocks(fonts).blocks(tree.children, indent=0, width=width)
    document.build(story or [Spacer(1, 1)])
    return RenderedPdf(content=buffer.getvalue(), missing=fonts.missing)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--font-dir", action="append", type=Path, default=[])
    parser.add_argument("--max-memory-bytes", type=int)
    parser.add_argument("--max-pdf-bytes", type=int)
    args = parser.parse_args(argv)
    markdown = sys.stdin.buffer.read().decode("utf-8")
    if args.max_memory_bytes:
        _limit_memory(args.max_memory_bytes)
    try:
        rendered = render_markdown_pdf(markdown, tuple(args.font_dir) or None)
    except MemoryError:
        return EXIT_OUT_OF_MEMORY
    except Exception as exc:
        if isinstance(exc.__context__, MemoryError):
            return EXIT_OUT_OF_MEMORY
        raise
    if args.max_pdf_bytes and len(rendered.content) > args.max_pdf_bytes:
        return EXIT_PDF_TOO_LARGE
    header = json.dumps({"missing": rendered.missing}).encode("utf-8")
    sys.stdout.buffer.write(header + b"\n" + rendered.content)
    return 0


def _limit_memory(max_bytes: int) -> None:
    try:
        import resource
    except ImportError:
        return
    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))


@dataclass(frozen=True)
class _Found:
    source: FontSource
    paths: dict[Weight, Path]

    @cached_property
    def cmap(self) -> frozenset[int]:
        font = FontFile(str(self.paths[REGULAR]), fontNumber=0, lazy=True)
        try:
            return frozenset(font.getBestCmap() or {})
        finally:
            font.close()


def _available_sources(font_dirs: tuple[Path, ...]) -> list[_Found]:
    found = []
    for source in FONT_SOURCES:
        paths = {
            weight: path
            for weight, name in source.files.items()
            if (path := _find_font_file(font_dirs, name)) is not None
        }
        regular = paths.get(REGULAR)
        if regular is not None:
            found.append(
                _Found(
                    source=source,
                    paths={
                        weight: paths.get(weight, regular) for weight in source.files
                    },
                )
            )
    return found


def _find_font_file(font_dirs: tuple[Path, ...], name: str) -> Path | None:
    for folder in font_dirs:
        if folder.is_dir():
            match = next(folder.rglob(name), None)
            if match is not None:
                return match
    return None


class _Fonts:
    def __init__(self, sources: list[_Found], drawn_text: str) -> None:
        self._sources = sources
        self._drawn_text = drawn_text
        self._names: dict[Path, str] = {}
        self._missing: dict[str, None] = {}

    @property
    def missing(self) -> list[str]:
        return list(self._missing)

    def body(self, weight: Weight = REGULAR) -> str:
        return self._registered_name(self._default(code=False), weight, code=False)

    def markup(self, text: str, weight: Weight, code: bool) -> str:
        return "".join(
            f'<font name="{self._registered_name(found, weight, code)}">'
            f"{escape(chunk)}</font>"
            for found, chunk in self._chunks(text, code)
        )

    def widths(self, text: str, size: float, code: bool) -> Iterator[tuple[str, float]]:
        for found, chunk in self._chunks(text, code):
            name = self._registered_name(found, REGULAR, code)
            for char in chunk:
                yield char, pdfmetrics.stringWidth(char, name, size)

    def _chunks(self, text: str, code: bool) -> Iterator[tuple[_Found | None, str]]:
        current: _Found | None = None
        chunk = ""
        for char in text:
            if _is_invisible(char):
                continue
            keeps_font = (
                char.isspace() and current is not None and ord(char) in current.cmap
            )
            found = current if keeps_font else self._choose(char, code)
            if chunk and found is not current:
                yield current, chunk
                chunk = ""
            current = found
            chunk += char
        if chunk:
            yield current, chunk

    def _usable(self, code: bool) -> list[_Found]:
        return [found for found in self._sources if code or not found.source.code_only]

    def _default(self, code: bool) -> _Found | None:
        return next(
            (found for found in self._usable(code) if found.source.can_be_default),
            None,
        )

    def _choose(self, char: str, code: bool) -> _Found | None:
        match = next((f for f in self._usable(code) if ord(char) in f.cmap), None)
        if match is not None:
            return match
        default = self._default(code)
        drawn_by_base_font = default is None and _base_font_covers(char)
        if not char.isspace() and not drawn_by_base_font:
            self._missing[char] = None
        return default

    def _registered_name(self, found: _Found | None, weight: Weight, code: bool) -> str:
        if found is None:
            return (_BASE_CODE_FONTS if code else _BASE_FONTS)[weight]
        path = found.paths[weight]
        if path not in self._names:
            name = f"UiPathPdf-{found.source.key}-{uuid.uuid4().hex}"
            if found.source.embed_subset:
                subset_font = _truetype_subset(path, self._drawn_text, name)
                font = TTFont(name, io.BytesIO(subset_font))
            else:
                font = TTFont(name, str(path))
            pdfmetrics.registerFont(font)
            self._names[path] = name
        return self._names[path]


def _truetype_subset(path: Path, text: str, face_name: str) -> bytes:
    font = FontFile(str(path), fontNumber=0)
    options = subset.Options()
    options.notdef_outline = True
    options.layout_features = []
    options.drop_tables += _UNUSED_LAYOUT_TABLES
    options.name_IDs = ["*"]
    subsetter = subset.Subsetter(options)
    subsetter.populate(text=text)
    subsetter.subset(font)
    if "CFF " in font:
        _cff_to_glyf(font)
    _rename_face(font, face_name)
    output = io.BytesIO()
    font.save(output)
    return output.getvalue()


def _rename_face(font: FontFile, face_name: str) -> None:
    names: Any = font["name"]
    names.names = [record for record in names.names if record.nameID != 6]
    names.setName(face_name, 6, 3, 1, 0x409)
    names.setName(face_name, 6, 1, 0, 0)


def _cff_to_glyf(font: FontFile) -> None:
    order = font.getGlyphOrder()
    glyph_set = font.getGlyphSet()
    glyf: Any = newTable("glyf")
    glyf.glyphOrder = order
    glyf.glyphs = {}
    for name in order:
        pen = TTGlyphPen(glyph_set)
        glyph_set[name].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
        glyf[name] = pen.glyph()
    font["glyf"] = glyf
    font["loca"] = newTable("loca")
    del font["CFF "]
    maxp: Any = newTable("maxp")
    maxp.tableVersion = 0x00010000
    for field in (
        "maxTwilightPoints",
        "maxStorage",
        "maxFunctionDefs",
        "maxInstructionDefs",
        "maxStackElements",
        "maxSizeOfInstructions",
        "maxComponentElements",
    ):
        setattr(maxp, field, 0)
    maxp.maxZones = 1
    font["maxp"] = maxp
    font["head"].glyphDataFormat = 0
    font["head"].indexToLocFormat = 0
    post: Any = font["post"]
    post.formatType = 2.0
    post.extraNames = []
    post.mapping = {}
    post.glyphOrder = order
    font.sfntVersion = "\x00\x01\x00\x00"


def _is_invisible(char: str) -> bool:
    return unicodedata.category(char) == "Cf" or 0xFE00 <= ord(char) <= 0xFE0F


def _base_font_covers(char: str) -> bool:
    try:
        char.encode("cp1252")
    except UnicodeEncodeError:
        return False
    return True


class _Blocks:
    def __init__(self, fonts: _Fonts) -> None:
        self._fonts = fonts

    def blocks(
        self, nodes: Sequence[SyntaxTreeNode], indent: float, width: float
    ) -> list[Flowable]:
        story: list[Flowable] = []
        for node in nodes:
            story.extend(self._block(node, indent, width))
        return story

    def _block(
        self, node: SyntaxTreeNode, indent: float, width: float
    ) -> list[Flowable]:
        if node.type == "heading":
            size = _HEADING_SIZES[int(node.tag[1])]
            return [
                Paragraph(self._inline(node, BOLD), self._style(size, BOLD, indent))
            ]
        if node.type == "paragraph":
            return [
                Paragraph(self._inline(node), self._style(_BASE_SIZE, REGULAR, indent))
            ]
        if node.type in ("bullet_list", "ordered_list"):
            return [self._list(node, indent, width)]
        if node.type == "table":
            return [self._table(node)]
        if node.type in ("fence", "code_block"):
            return [self._code(node.content, indent, width)]
        if node.type == "blockquote":
            return self.blocks(node.children, indent + _QUOTE_INDENT, width)
        if node.type == "hr":
            return [
                HRFlowable(width="100%", thickness=0.5, color=colors.grey, spaceAfter=6)
            ]
        return self.blocks(node.children, indent, width)

    def _style(self, size: float, weight: Weight, indent: float) -> ParagraphStyle:
        return ParagraphStyle(
            name="uipath-pdf",
            fontName=self._fonts.body(weight),
            fontSize=size,
            leading=size * 1.35,
            leftIndent=indent,
            spaceAfter=size * 0.5,
        )

    def _code(self, text: str, indent: float, width: float) -> Flowable:
        line_width = width - indent - 2 * _CODE_PADDING
        lines = [
            piece
            for line in text.rstrip("\n").split("\n")
            for piece in self._wrap(line.expandtabs(_CODE_TAB_SIZE), line_width)
        ]
        markup = "\n".join(
            self._fonts.markup(line, REGULAR, code=True) for line in lines
        )
        style = self._style(_CODE_SIZE, REGULAR, indent)
        style.backColor = _LIGHT_GREY
        style.borderPadding = _CODE_PADDING
        style.spaceBefore = 4
        style.spaceAfter = 8
        return XPreformatted(markup, style)

    def _wrap(self, line: str, line_width: float) -> list[str]:
        pieces = []
        piece = ""
        used = 0.0
        for char, advance in self._fonts.widths(line, _CODE_SIZE, code=True):
            if piece and used + advance > line_width:
                pieces.append(piece)
                piece = ""
                used = 0.0
            piece += char
            used += advance
        pieces.append(piece)
        return pieces

    def _list(self, node: SyntaxTreeNode, indent: float, width: float) -> Flowable:
        ordered = node.type == "ordered_list"
        item_width = width - indent - _LIST_INDENT
        return ListFlowable(
            [self.blocks(item.children, 0, item_width) for item in node.children],
            bulletType="1" if ordered else "bullet",
            start=int(str(node.attrs.get("start", 1))) if ordered else "•",
            bulletFontName=self._fonts.body(),
            bulletFontSize=_BASE_SIZE,
            leftIndent=indent + _LIST_INDENT,
        )

    def _table(self, node: SyntaxTreeNode) -> Flowable:
        rows: list[list[Flowable]] = []
        has_header = False
        for section in node.children:
            header = section.type == "thead"
            has_header = has_header or header
            weight = BOLD if header else REGULAR
            style = self._style(_TABLE_SIZE, weight, 0)
            for row in section.children:
                rows.append(
                    [
                        Paragraph(self._inline(cell, weight), style)
                        for cell in row.children
                    ]
                )
        table = Table(
            rows, repeatRows=1 if has_header else 0, hAlign="LEFT", splitInRow=1
        )
        table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BACKGROUND", (0, 0), (-1, 0 if has_header else -1), _LIGHT_GREY),
                ]
            )
        )
        return table

    def _inline(self, node: SyntaxTreeNode, weight: Weight = REGULAR) -> str:
        return "".join(self._spans(child, weight, None) for child in node.children)

    def _spans(self, node: SyntaxTreeNode, weight: Weight, href: str | None) -> str:
        if node.type == "text":
            markup = self._fonts.markup(node.content, weight, code=False)
            return self._link(markup, href)
        if node.type == "code_inline":
            markup = self._fonts.markup(node.content, weight, code=True)
            return self._link(markup, href)
        if node.type == "softbreak":
            return " "
        if node.type == "hardbreak":
            return "<br/>"
        if node.type == "strong":
            weight = weight._replace(bold=True)
        elif node.type == "em":
            weight = weight._replace(italic=True)
        elif node.type == "link":
            target = str(node.attrs.get("href", ""))
            href = target if urlsplit(target).scheme in _SAFE_LINK_SCHEMES else None
        return "".join(self._spans(child, weight, href) for child in node.children)

    @staticmethod
    def _link(markup: str, href: str | None) -> str:
        if href is None or not markup:
            return markup
        return f'<a href="{escape(href, {chr(34): "&quot;"})}">{markup}</a>'


if __name__ == "__main__":
    sys.exit(main())
