from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

LATIN_CHARS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    " .,:;!?%&<>=+-*/()[]#|'\"_ăâîșțĂÂÎȘȚ"
)
CJK_CHARS = " ★報告書の概要한국어보고서中文报告"
STYLES = ("Regular", "Bold", "Italic", "BoldItalic")


def _box_font(chars: str, font_name: str, *, cff: bool, advance: int = 600) -> Any:
    from fontTools.fontBuilder import FontBuilder  # type: ignore[import-untyped]
    from fontTools.pens.t2CharStringPen import (  # type: ignore[import-untyped]
        T2CharStringPen,
    )
    from fontTools.pens.ttGlyphPen import TTGlyphPen  # type: ignore[import-untyped]

    glyphs = {char: f"uni{ord(char):04X}" for char in dict.fromkeys(chars)}
    order = [".notdef", *glyphs.values()]
    builder = FontBuilder(1000, isTTF=not cff)
    builder.setupGlyphOrder(order)
    builder.setupCharacterMap({ord(char): glyph for char, glyph in glyphs.items()})
    pens = {
        glyph: T2CharStringPen(advance, None) if cff else TTGlyphPen(None)
        for glyph in order
    }
    for pen in pens.values():
        pen.moveTo((100, 0))
        pen.lineTo((100, 700))
        pen.lineTo((500, 700))
        pen.lineTo((500, 0))
        pen.closePath()
    if cff:
        charstrings = {glyph: pen.getCharString() for glyph, pen in pens.items()}
        builder.setupCFF(font_name, {"FullName": font_name}, charstrings, {})
    else:
        builder.setupGlyf({glyph: pen.glyph() for glyph, pen in pens.items()})
    builder.setupHorizontalMetrics({glyph: (advance, 100) for glyph in order})
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupNameTable(
        {"familyName": font_name, "styleName": "Regular", "psName": font_name}
    )
    builder.setupOS2(sTypoAscender=800, usWinAscent=800, usWinDescent=200)
    builder.setupPost()
    return builder.font


@pytest.fixture
def install_pdf_fonts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., Path]:
    pdf_renderer = pytest.importorskip(
        "uipath_langchain.agent.tools.internal_tools.pdf_renderer",
        reason="needs the pdf extra",
    )
    from fontTools.ttLib.ttCollection import (  # type: ignore[import-untyped]
        TTCollection,
    )

    def install(latin: bool = True, cjk: bool = True, mono: bool = False) -> Path:
        root = tmp_path / f"fonts-latin{latin}-cjk{cjk}-mono{mono}"
        root.mkdir()
        for family, wanted in (("Sans", latin), ("Mono", mono)):
            if wanted:
                for style in STYLES:
                    _box_font(LATIN_CHARS, f"Test{family}-{style}", cff=False).save(
                        str(root / f"Liberation{family}-{style}.ttf")
                    )
        if cjk:
            for weight in ("Regular", "Bold"):
                collection = TTCollection()
                collection.fonts = [
                    _box_font(CJK_CHARS, f"TestCJK-{weight}", cff=True, advance=1000)
                ]
                collection.save(str(root / f"NotoSansCJK-{weight}.ttc"))
        monkeypatch.setattr(pdf_renderer, "FONT_DIRS", (root,))
        return root

    return install


@pytest.fixture
def pdf_fonts(install_pdf_fonts: Callable[..., Path]) -> Path:
    return install_pdf_fonts()
