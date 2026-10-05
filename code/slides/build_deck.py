#!/usr/bin/env python3
"""Mid-evaluation slide deck as an editable PowerPoint file.

    python -m slides.build_deck          # from code/; writes ../Slides/CLARITY_mideval.pptx

Text, tables, equations and diagrams are native PowerPoint objects; the data plots are the
report-style figures from figures/deck_figures.py (docs/figures/deck/). Needs python-pptx.
Every number on a slide is traced in its speaker notes to the file it comes from.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

from qevasion.paths import FIGURES, ROOT

FIG = FIGURES / "deck"
OUT = ROOT / "Slides" / "CLARITY_mideval.pptx"

# Palette: slate text on an off-white page, white panels, one steel accent for "what changed".
BG, INK, TITLE, MUTED = "F8F9FA", "1E293B", "0F172A", "64748B"
CARD, RULE, HEAD, ZEBRA = "FFFFFF", "CBD5E1", "E2E8F0", "F8FAFC"
ACCENT, ACCENT_SOFT, SAGE, SAGE_SOFT, CODE_BG = "2F5D7C", "E8EEF4", "5B7B6F", "EEF3F0", "EEF2F6"

W, H = 13.333, 7.5
LEFT_X, LEFT_W = 0.6, 4.4           # bullet column
CARD_X, CARD_W = 5.3, 7.43          # visual column (white panel); figures are drawn 7.0 in wide to fit it 1:1
TOP, BOTTOM = 1.55, 6.85            # content band below the title, above the footer
PAD = 0.3                           # inner padding of a panel
FOOTER = "Nier_ANLP  ·  SemEval-2026 Task 6 (CLARITY)  ·  all scores on the 308-item dev set"

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "a14": "http://schemas.microsoft.com/office/drawing/2010/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}


def rgb(hex_: str) -> RGBColor:
    return RGBColor.from_string(hex_)


# --------------------------------------------------------------------------- deck setup

def setup_theme(prs: Presentation) -> None:
    """Theme fonts (Cambria headings, Calibri body) and colours, so the deck restyles from Design > Variants."""
    part = prs.slide_master.part.part_related_by(RT.THEME)
    xml = part.blob.decode("utf-8")
    xml = re.sub(r"(<a:majorFont>\s*<a:latin typeface=\")[^\"]*", r"\1Cambria", xml)
    xml = re.sub(r"(<a:minorFont>\s*<a:latin typeface=\")[^\"]*", r"\1Calibri", xml)
    scheme = {"dk1": INK, "lt1": "FFFFFF", "dk2": TITLE, "lt2": BG, "accent1": ACCENT, "accent2": SAGE,
              "accent3": "94A3B8", "accent4": MUTED, "accent5": "8FA9BF", "accent6": "A3B8AE"}
    for tag, val in scheme.items():
        xml = re.sub(rf"<a:{tag}>.*?</a:{tag}>", f'<a:{tag}><a:srgbClr val="{val}"/></a:{tag}>', xml, flags=re.S)
    part._blob = xml.encode("utf-8")


def setup_master(prs: Presentation):
    """Off-white background, left-aligned 30 pt title; footer and slide number on the content layout."""
    master = prs.slide_master
    master.background.fill.solid()
    master.background.fill.fore_color.rgb = rgb(BG)
    lvl1 = master._element.find(".//" + qn("p:titleStyle") + "/" + qn("a:lvl1pPr"))
    lvl1.set("algn", "l")
    rpr = lvl1.find(qn("a:defRPr"))
    rpr.set("sz", "3000")
    rpr.set("b", "1")

    layout = prs.slide_layouts[5]  # "Title Only"
    title = layout.placeholders[0]
    title.left, title.top, title.width, title.height = Inches(0.6), Inches(0.72), Inches(12.13), Inches(0.66)
    body_pr = title._element.find(".//" + qn("a:bodyPr"))
    body_pr.set("anchor", "t")
    for side in ("lIns", "tIns", "rIns", "bIns"):
        body_pr.set(side, "0")
    for child in list(body_pr):
        body_pr.remove(child)
    etree.SubElement(body_pr, qn("a:normAutofit"))

    # Footer and slide number live on the layout, so they change in one place.
    tmp = prs.slides.add_slide(prs.slide_layouts[6])
    foot = text_box(tmp, 0.6, 7.03, 9.5, 0.28, FOOTER, size=10, color=MUTED)
    num = text_box(tmp, 11.73, 7.03, 1.0, 0.28, "", size=10, color=MUTED, align=PP_ALIGN.RIGHT)
    p = num.text_frame.paragraphs[0]._p
    fld = etree.SubElement(p, qn("a:fld"), id="{B6F15528-21DE-4FAA-801E-634DDDAF4B2B}", type="slidenum")
    etree.SubElement(fld, qn("a:rPr"), lang="en-US", sz="1000")
    etree.SubElement(fld, qn("a:t")).text = "‹#›"
    end = p.find(qn("a:endParaRPr"))
    if end is not None:
        p.remove(end)
        p.append(end)
    for shape in (foot, num):
        layout.shapes._spTree.append(shape._element)
    drop_slide(prs, tmp)
    return layout


def drop_slide(prs: Presentation, slide) -> None:
    sld_ids = prs.slides._sldIdLst
    for sld_id in list(sld_ids):
        if prs.part.related_part(sld_id.rId) is slide.part:
            prs.part.drop_rel(sld_id.rId)
            sld_ids.remove(sld_id)


# --------------------------------------------------------------------------- text primitives

def add_runs(paragraph, text: str, size: float, color: str = INK, bold: bool = False, font: str | None = None,
             italic: bool = False) -> None:
    """Write `text` into the paragraph; **double asterisks** mark bold spans."""
    for i, chunk in enumerate(re.split(r"\*\*", text)):
        if not chunk:
            continue
        run = paragraph.add_run()
        run.text = chunk
        run.font.size = Pt(size)
        run.font.bold = bold or (i % 2 == 1)
        run.font.italic = italic
        run.font.color.rgb = rgb(color)
        if font:
            run.font.name = font


def text_box(slide, x, y, w, h, text, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, font=None, italic=False, name=None, line_spacing=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if line_spacing:
            p.line_spacing = line_spacing
        add_runs(p, line, size, color, bold, font, italic)
    return tb


def bullets(slide, items: list[str], x=LEFT_X, y=TOP, w=LEFT_W, h=BOTTOM - TOP, size=18, gap=14,
            anchor=MSO_ANCHOR.MIDDLE):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = "Bullets"
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        p.line_spacing = 1.05
        add_runs(p, item, size)
        ppr = p._p.get_or_add_pPr()
        ppr.set("marL", str(Inches(0.27)))
        ppr.set("indent", str(-Inches(0.27)))
        bu_clr = etree.SubElement(ppr, qn("a:buClr"))
        etree.SubElement(bu_clr, qn("a:srgbClr"), val=ACCENT)
        etree.SubElement(ppr, qn("a:buFont"), typeface="Arial")
        etree.SubElement(ppr, qn("a:buChar"), char="•")
    return tb


def kicker_and_title(slide, kicker: str, title: str) -> None:
    k = text_box(slide, 0.6, 0.36, 9.0, 0.28, kicker.upper(), size=12, color=ACCENT, bold=True, name="Kicker")
    for r in k.text_frame.paragraphs[0].runs:
        r.font._rPr.set("spc", "120")
    slide.shapes.title.text = title
    for r in slide.shapes.title.text_frame.paragraphs[0].runs:
        r.font.color.rgb = rgb(TITLE)


def caption(slide, x, y, w, text, h=0.42, align=PP_ALIGN.LEFT):
    return text_box(slide, x, y, w, h, text, size=11, color=MUTED, italic=True, align=align, name="Caption")


# --------------------------------------------------------------------------- panels, figures, tables

def panel(slide, x=CARD_X, y=TOP, w=CARD_W, h=BOTTOM - TOP, fill=CARD, line=RULE, name="Panel"):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.name = name
    shp.adjustments[0] = 0.025
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    if line:
        shp.line.color.rgb = rgb(line)
        shp.line.width = Pt(0.75)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    shp.text_frame.text = ""
    return shp


def band_panel(slide, content_h: float, x=CARD_X, w=CARD_W, top=TOP, bottom=BOTTOM, name="Panel"):
    """A panel that hugs `content_h` of content, centred in the band; returns the inner (x, y, w)."""
    h = min(content_h + 2 * PAD, bottom - top)
    y = (top + bottom) / 2 - h / 2
    panel(slide, x, y, w, h, name=name)
    return x + PAD, y + PAD, w - 2 * PAD


def figure(slide, name: str, cap: str | None = None, x=CARD_X, w=CARD_W, top=TOP, bottom=BOTTOM, cap_h=0.4):
    """A panel that hugs a figure drawn at the panel's width (so its text keeps its size), caption below."""
    pad = 0.18
    img_w, img_h = Image.open(FIG / f"{name}.png").size
    max_h = bottom - top - 2 * pad - (cap_h + 0.06 if cap else 0)
    scale = min((w - 2 * pad) / img_w, max_h / img_h)
    fw, fh = img_w * scale, img_h * scale
    h = fh + 2 * pad + (cap_h + 0.06 if cap else 0)
    y = (top + bottom) / 2 - h / 2
    panel(slide, x, y, w, h, name=f"Panel: {name}")
    pic = slide.shapes.add_picture(str(FIG / f"{name}.png"), Inches(x + (w - fw) / 2), Inches(y + pad), Inches(fw), Inches(fh))
    pic.name = f"Figure: {name}"
    if cap:
        caption(slide, x + PAD, y + pad + fh + 0.06, w - 2 * PAD, cap, h=cap_h)
    return pic


def _cell_border(cell, color=RULE, width=9525):
    tc_pr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        ln = etree.SubElement(tc_pr, qn(tag), w=str(width), cap="flat", cmpd="sng", algn="ctr")
        fill = etree.SubElement(ln, qn("a:solidFill"))
        etree.SubElement(fill, qn("a:srgbClr"), val=color)
        etree.SubElement(ln, qn("a:prstDash"), val="solid")


def table(slide, x, y, col_w: list[float], rows: list[list[str]], size=14, row_h=0.36, header=True,
          align: list | None = None, name="Table", bold_rows: tuple = ()):
    n_r, n_c = len(rows), len(col_w)
    gf = slide.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(sum(col_w)), Inches(row_h * n_r))
    gf.name = name
    tbl = gf.table
    tbl_pr = tbl._tbl.tblPr
    tbl_pr.set("firstRow", "0")
    tbl_pr.set("bandRow", "0")
    style = tbl_pr.find(qn("a:tableStyleId"))
    if style is None:
        style = etree.SubElement(tbl_pr, qn("a:tableStyleId"))
    style.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"  # "No Style, No Grid": every cell styled below
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        is_head = header and i == 0
        for j in range(n_c):
            cell = tbl.cell(i, j)
            _cell_border(cell)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(HEAD if is_head else (ZEBRA if i % 2 == 0 else CARD))
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.035)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = (align[j] if align else PP_ALIGN.LEFT) if not is_head or j else PP_ALIGN.LEFT
            add_runs(p, row[j] if j < len(row) else "", size, TITLE if is_head else INK,
                     bold=is_head or i in bold_rows)
    return gf


def code_block(slide, x, y, w, h, code: str, size=12):
    shp = panel(slide, x, y, w, h, fill=CODE_BG, name="Code")
    tf = shp.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = Inches(0.15)
    tf.margin_top = tf.margin_bottom = Inches(0.1)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, line in enumerate(code.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        add_runs(p, line, size, MUTED if line.lstrip().startswith("#") else TITLE, font="Courier New")
    return shp


def stat_card(slide, x, y, w, h, label, value, sub, accent=False):
    panel(slide, x, y, w, h, fill=ACCENT_SOFT if accent else CARD, line=ACCENT if accent else RULE,
          name=f"Stat: {label}")
    text_box(slide, x + 0.2, y + 0.16, w - 0.4, 0.3, label, size=12, color=MUTED)
    text_box(slide, x + 0.2, y + 0.46, w - 0.4, 0.62, value, size=34, color=TITLE, bold=True, font="+mj-lt")
    text_box(slide, x + 0.2, y + h - 0.46, w - 0.4, 0.3, sub, size=12, color=INK)


# --------------------------------------------------------------------------- diagrams (native shapes)

def node(shapes, x, y, w, h, text, size=12, fill=CARD, line=MUTED, dashed=False, bold_first=True, name=None):
    shp = shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = 0.12
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    shp.line.color.rgb = rgb(line)
    shp.line.width = Pt(1.25 if line == ACCENT else 1.0)
    if dashed:
        shp.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    shp.shadow.inherit = False
    if name:
        shp.name = name
    tf = shp.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, line_txt in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        add_runs(p, line_txt, size, TITLE if i == 0 else INK, bold=bold_first and i == 0)
    return shp


def arrow(shapes, a, a_site, b, b_site, dashed=False, elbow=False, color=MUTED):
    """Connector glued to shape connection sites (0 top, 1 left, 2 bottom, 3 right) with an arrowhead."""
    kind = MSO_CONNECTOR.ELBOW if elbow else MSO_CONNECTOR.STRAIGHT
    c = shapes.add_connector(kind, 0, 0, 0, 0)
    c.begin_connect(a, a_site)
    c.end_connect(b, b_site)
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(1.25)
    if dashed:
        c.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    ln = c.line._get_or_add_ln()
    etree.SubElement(ln, qn("a:tailEnd"), type="triangle", w="med", len="med")
    return c


def polyline(shapes, points: list[tuple[float, float]], dashed=False, color=MUTED):
    """An open freeform line through `points` (inches) ending in an arrowhead: a right-angled route."""
    fb = shapes.build_freeform(Inches(points[0][0]), Inches(points[0][1]), scale=1.0)
    fb.add_line_segments([(Inches(px_), Inches(py_)) for px_, py_ in points[1:]], close=False)
    shp = fb.convert_to_shape()
    shp.fill.background()
    shp.line.color.rgb = rgb(color)
    shp.line.width = Pt(1.25)
    if dashed:
        shp.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    shp.shadow.inherit = False
    ln = shp.line._get_or_add_ln()
    etree.SubElement(ln, qn("a:tailEnd"), type="triangle", w="med", len="med")
    return shp


def edge_label(shapes, x, y, w, text, align=PP_ALIGN.CENTER, size=11):
    tb = shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.26))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = align
    add_runs(p, text, size, MUTED, italic=True)
    return tb


# --------------------------------------------------------------------------- native equations

def _mr(t: str, plain=False, size=2000) -> str:
    sty = '<m:rPr><m:sty m:val="p"/></m:rPr>' if plain else ""
    return (f'<m:r>{sty}<a:rPr lang="en-US" sz="{size}" i="{0 if plain else 1}"><a:solidFill><a:srgbClr val="{TITLE}"/>'
            f'</a:solidFill><a:latin typeface="Cambria Math" panose="02040503050406030204" pitchFamily="18" charset="0"/>'
            f'</a:rPr><m:t>{escape(t)}</m:t></m:r>')


def _sub(e, s):
    return f"<m:sSub><m:e>{e}</m:e><m:sub>{s}</m:sub></m:sSub>"


def _sup(e, s):
    return f"<m:sSup><m:e>{e}</m:e><m:sup>{s}</m:sup></m:sSup>"


def _subsup(e, b, t):
    return f"<m:sSubSup><m:e>{e}</m:e><m:sub>{b}</m:sub><m:sup>{t}</m:sup></m:sSubSup>"


def _frac(n, d):
    return f"<m:f><m:num>{n}</m:num><m:den>{d}</m:den></m:f>"


def _acc(e, ch):
    return f'<m:acc><m:accPr><m:chr m:val="{ch}"/></m:accPr><m:e>{e}</m:e></m:acc>'


def _delim(e, beg, end):
    return f'<m:d><m:dPr><m:begChr m:val="{beg}"/><m:endChr m:val="{end}"/></m:dPr><m:e>{e}</m:e></m:d>'


HAT, BAR = "\u0302", "\u0305"
P = lambda t: _mr(t, plain=True)  # noqa: E731  upright run (operators, digits, words)
I = _mr                           # italic run (variables)

EQUATIONS = {
    "maps": (
        I("x") + P("=") + _delim(I("q") + P(",") + I("Q") + P(",") + I("a"), "(", ")") + P(",   ")
        + I("f") + P(":") + I("\U0001D4B3") + P("→") + _sub(I("\U0001D4B4"), P("9")) + P(",   ")
        + I("g") + P(":") + _sub(I("\U0001D4B4"), P("9")) + P("→") + _sub(I("\U0001D4B4"), P("3")),
        r"$x=(q,Q,a),\quad f:\mathcal{X}\to\mathcal{Y}_9,\quad g:\mathcal{Y}_9\to\mathcal{Y}_3$",
    ),
    "metric": (
        P("macro-F1") + P("=") + _frac(
            _delim(_delim(_sub(_acc(I("y"), HAT), P("1")) + P(", …, ") + _sub(_acc(I("y"), HAT), I("N")), "{", "}"), "|", "|"),
            P("9")),
        r"$\mathrm{macro}$-$\mathrm{F1}=\dfrac{|\{\hat{y}_1,\ldots,\hat{y}_N\}|}{9}$",
    ),
    "rule": (
        _acc(I("y"), HAT) + _delim(I("x"), "(", ")") + P("=")
        + "<m:func><m:fName><m:limLow><m:e>" + P("arg max") + "</m:e><m:lim>" + I("c") + "</m:lim></m:limLow></m:fName><m:e>"
        + _frac(_acc(I("p"), BAR) + _delim(I("c") + P(" | ") + I("x"), "(", ")"), _subsup(I("π"), I("c"), I("τ")))
        + "</m:e></m:func>",
        r"$\hat{y}(x)=\arg\max_{c}\ \dfrac{\bar{p}(c\,|\,x)}{\pi_c^{\tau}}$",
    ),
}


def _fallback_png(tex: str, path: Path) -> tuple[float, float]:
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex, fontsize=20, color=f"#{TITLE}", math_fontfamily="stix")
    fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    w, h = Image.open(path).size
    return w / 300, h / 300


def equation(slide, key: str, x, y, w, h, tmp: Path):
    """An editable PowerPoint equation, with a rendered image as the fallback for viewers without OMML."""
    omml, tex = EQUATIONS[key]
    png = tmp / f"eq_{key}.png"
    fw, fh = _fallback_png(tex, png)
    scale = min(1.0, w / fw, h / fh)
    pic = slide.shapes.add_picture(str(png), Inches(x), Inches(y + (h - fh * scale) / 2), Inches(fw * scale), Inches(fh * scale))
    pic.name = f"Equation {key} (image fallback)"
    sp_tree = slide.shapes._spTree
    sp_tree.remove(pic._element)
    shape_id = pic.shape_id
    choice_sp = (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="Equation {key}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
        f'<p:spPr><a:xfrm><a:off x="{Emu(Inches(x))}" y="{Emu(Inches(y))}"/><a:ext cx="{Emu(Inches(w))}" cy="{Emu(Inches(h))}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>'
        f'<p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" anchor="ctr"><a:noAutofit/></a:bodyPr><a:lstStyle/>'
        f'<a:p><a:pPr algn="l"/><a14:m><m:oMathPara><m:oMathParaPr><m:jc m:val="left"/></m:oMathParaPr>'
        f'<m:oMath>{omml}</m:oMath></m:oMathPara></a14:m><a:endParaRPr lang="en-US" sz="2000"/></a:p></p:txBody></p:sp>'
    )
    decl = " ".join(f'xmlns:{k}="{v}"' for k, v in NS.items())
    alt = etree.fromstring(
        f'<mc:AlternateContent {decl}><mc:Choice Requires="a14">{choice_sp}</mc:Choice><mc:Fallback/></mc:AlternateContent>')
    alt.find(f"{{{NS['mc']}}}Fallback").append(pic._element)
    sp_tree.append(alt)


# --------------------------------------------------------------------------- slides

def content_slide(prs, layout, kicker, title, notes):
    s = prs.slides.add_slide(layout)
    kicker_and_title(s, kicker, title)
    s.notes_slide.notes_text_frame.text = notes
    return s


def build() -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    setup_theme(prs)
    layout = setup_master(prs)
    blank = prs.slide_layouts[6]
    tmp = Path(tempfile.mkdtemp())
    L, C, R = PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.RIGHT

    # 1. Title -------------------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    text_box(s, 0.8, 1.0, 7.2, 0.3, "SEMEVAL-2026 TASK 6  ·  CLARITY", size=13, color=ACCENT, bold=True)
    text_box(s, 0.8, 1.42, 7.45, 0.8, "Knowledge over Structure", size=42, color=TITLE, bold=True, font="+mj-lt",
             name="Title")
    text_box(s, 0.8, 2.32, 7.0, 0.9, "A controlled study of response clarity classification in political interviews",
             size=22, color=INK, font="+mj-lt")
    text_box(s, 0.8, 3.4, 7.0, 0.6, "Mid-evaluation progress: from a DeBERTa baseline to a LoRA-tuned 8B classifier",
             size=16, color=MUTED)
    text_box(s, 0.8, 4.45, 6.6, 1.6,
             "**Team Nier_ANLP**\nVidvathama R  (2024122002)\nSiddarth Gottumukkula  (2023102040)\n"
             "Sanjana Reddy Vonteri  (2026901007)\nShashikanta Sahoo  (2026900007)", size=14, color=INK)
    text_box(s, 0.8, 6.3, 7.0, 0.3, "International Institute of Information Technology, Hyderabad  ·  October 2026",
             size=12, color=MUTED)
    px0, py0 = 8.55, 1.5
    panel(s, px0, py0, 4.15, 4.3, name="Panel: task example")
    text_box(s, px0 + 0.3, py0 + 0.3, 3.55, 0.3, "ONE DEV ITEM", size=11, color=ACCENT, bold=True)
    text_box(s, px0 + 0.3, py0 + 0.72, 3.55, 0.85, "**Sub-question**\nWould you campaign against Senator Joe Lieberman "
                                                   "on Iraq?", size=14)
    text_box(s, px0 + 0.3, py0 + 1.72, 3.55, 0.6, "**Answer**\n“I’m going to stay out of Connecticut.”", size=14)
    text_box(s, px0 + 0.3, py0 + 2.5, 3.55, 0.6, "**Subtask 2** (9 evasion types)\nDodging, Dodging, Implicit", size=14)
    text_box(s, px0 + 0.3, py0 + 3.28, 3.55, 0.6, "**Subtask 1** (3 clarity levels)\nAmbivalent", size=14)
    s.notes_slide.notes_text_frame.text = (
        "Title. The task: given an interview question and the politician's answer, classify how the answer responds. "
        "The example on the right (dev item 17, three annotators) returns later as our annotated failure case. "
        "Source: Report/Report.pdf, title page and Figure 4.")

    # 2. Problem formulation -----------------------------------------------------------------
    s = content_slide(prs, layout, "Problem formulation", "Label how an answer responds to one sub-question",
                      "Each item is a triple: the sub-question q, the journalist's full turn Q, and the full answer a. "
                      "Subtask 2 asks for one of nine evasion types; Subtask 1's three clarity levels follow through the "
                      "fixed taxonomy g. The metric is multi-reference macro-F1: a prediction is a true positive if any "
                      "annotator gave that label; a miss charges a false negative to every label in the reference set. "
                      "If every prediction is acceptable, the score equals the number of distinct classes predicted over "
                      "nine, so naming every class matters. Sources: Report §2, Table 1; docs/01_scorer_geometry.md.")
    bullets(s, ["Input: sub-question **q** (15 tokens), full question **Q** (62), answer **a** (266).",
                "Subtask 2 predicts one of 9 evasion types; Subtask 1 follows through **g**.",
                "69% of train rows share an answer; 71.7% of those differ in label.",
                "Dev: 308 items, 3 annotators, only 125 unanimous. Test labels never released."])
    xi, yi, wi = band_panel(s, 4.55)
    text_box(s, xi, yi, wi, 0.3, "Input and label maps", size=12, color=MUTED)
    equation(s, "maps", xi, yi + 0.35, wi, 0.55, tmp)
    text_box(s, xi, yi + 1.2, wi, 0.3, "Multi-reference macro-F1, when every prediction is in its item's reference set",
             size=12, color=MUTED)
    equation(s, "metric", xi, yi + 1.55, wi, 0.95, tmp)
    text_box(s, xi, yi + 2.75, wi, 0.3, "Decision rule used by every system (τ fitted by nested CV)", size=12, color=MUTED)
    equation(s, "rule", xi, yi + 3.1, wi, 0.95, tmp)
    caption(s, xi, yi + 4.15, wi, "Clarification (4 dev reference sets) therefore weighs as much as Explicit (115): "
                                  "the score rewards landing in-set and naming every class.")

    # 3. Baselines and SOTA ------------------------------------------------------------------
    s = content_slide(prs, layout, "Baselines and state of the art", "Encoders plateau; the winners call an LLM per item",
                      "Fine-tuned encoders level off near 0.50 test S2 in the organisers' overview, however large or "
                      "ensembled. The top systems are multi-call LLM pipelines: TeleAI makes 1.94 calls per item with about "
                      "7,300 input tokens. Published systems change many components at once, so their gains cannot be "
                      "attributed. Our question is narrower: what limits a single-pass classifier? Floors: "
                      "docs/raw/mideval/trivial_baselines.csv. Published rows: each team's paper, dev set, own protocol.")
    bullets(s, ["Fine-tuned encoders plateau near 0.50 test S2, whatever their size or ensembling.",
                "Winners run multi-call LLM pipelines: 1.94 calls, ~7,300 input tokens per item.",
                "Published systems change several components at once; gains cannot be attributed.",
                "Our question: what limits a **single-pass** classifier on this task?"])
    xi, yi, wi = band_panel(s, 9 * 0.4 + 0.15 + 0.42)
    table(s, xi, yi, [3.5, 1.65, 0.84, 0.84], [
        ["System (dev set)", "Type", "S2", "S1"],
        ["Always “Explicit”", "floor", "0.060", "0.136"],
        ["Uniform random", "floor", "0.129", "0.299"],
        ["TF-IDF + logistic regression", "classical", "0.257", "0.445"],
        ["ChulaNLP, DeBERTa-large fine-tune", "encoder", "0.46", "0.65"],
        ["TeleAI, Qwen2.5-7B fine-tune", "single-pass LLM", "0.495", "–"],
        ["ChulaNLP, DeBERTa top-5 + Kimi-K2", "hybrid", "0.52", "–"],
        ["TeleAI, 3-stage DeepSeek-V3 (1st)", "multi-call LLM", "0.617", "0.812"],
        ["Human annotator vs. other two", "ceiling", "0.684", "–"],
    ], size=13, row_h=0.4, align=[L, L, R, R])
    caption(s, xi, yi + 9 * 0.4 + 0.15, wi, "Floors: docs/raw/mideval/trivial_baselines.csv. Published rows are dev scores "
                                            "from each team's paper, under their own protocols.")

    # 4. Architecture ------------------------------------------------------------------------
    s = content_slide(prs, layout, "Proposed architecture", "One classifier, one swappable backbone, one fitted scalar",
                      "Four explanations predict different things: label noise, the decision rule, missing knowledge, or "
                      "label meaning that cannot be learnt from 3,448 examples. The system is built so each can be tested "
                      "by one change. A flat 9-way head; Subtask 1 is read off through g. A system averages ten seeds and "
                      "applies logit adjustment, the only parameter fitted on dev. The backbone box is the one component "
                      "the E13 experiment swaps. Source: Report §1 and Figure 1.")
    bullets(s, ["Four candidate limits: label noise, decision rule, knowledge, or label meaning.",
                "One flat 9-way head; Subtask 1 derived through the fixed taxonomy **g**.",
                "System = 10-seed probability mean + one-parameter logit adjustment.",
                "Swappable backbone tests knowledge with everything else held fixed."])
    panel(s)
    g = s.shapes.add_group_shape()
    g.name = "Diagram: system (Report Figure 1)"
    sh = g.shapes
    x0, wfull, half = CARD_X + 0.35, CARD_W - 0.7, 3.2
    n_in = node(sh, x0, 1.82, wfull, 0.5, "Input  x = (q, Q, a)\nsub-question + full question + answer, ≤ 1,024 tokens", size=11)
    edge_label(sh, x0, 2.36, 2.0, "backbone (one of)", align=L, size=10)
    n_deb = node(sh, x0, 2.62, half, 0.62, "DeBERTa-v3-large\nfull fine-tune, [CLS] pooling", fill=ACCENT_SOFT, line=ACCENT)
    n_qw = node(sh, x0 + wfull - half, 2.62, half, 0.62, "Qwen3-8B-Base + LoRA\nr = 16, last-token pooling", fill=ACCENT_SOFT,
                line=ACCENT)
    n_head = node(sh, x0, 3.52, wfull, 0.45, "Linear head: 9 logits, softmax per seed", size=12)
    n_mean = node(sh, x0, 4.22, wfull, 0.45, "Mean of the 10 seeds' probabilities", size=12)
    n_la = node(sh, x0, 4.92, wfull, 0.45, "Logit adjustment: argmax of p / prior^τ   (τ by nested CV)", size=12)
    n_s2 = node(sh, x0, 5.62, half, 0.5, "Subtask 2\nevasion type (9)", size=11)
    n_s1 = node(sh, x0 + wfull - half, 5.62, half, 0.5, "Subtask 1\nclarity level (3)", size=11)
    for a, a_s, b, b_s in ((n_in, 2, n_deb, 0), (n_in, 2, n_qw, 0), (n_deb, 2, n_head, 0), (n_qw, 2, n_head, 0),
                           (n_head, 2, n_mean, 0), (n_mean, 2, n_la, 0), (n_s2, 3, n_s1, 1)):
        arrow(sh, a, a_s, b, b_s)
    polyline(sh, [(x0 + wfull / 2, 5.37), (x0 + wfull / 2, 5.495), (x0 + half / 2, 5.495), (x0 + half / 2, 5.62)])
    edge_label(sh, x0 + half, 5.6, wfull - 2 * half, "g", size=12)
    caption(s, x0, 6.3, wfull, "Redrawn from Report Figure 1. Accent boxes: the one component that differs between "
                               "the two tracks.", h=0.4)

    # 5. Protocol ----------------------------------------------------------------------------
    s = content_slide(prs, layout, "Evaluation protocol", "Every claim rests on ten paired seeds and a fixed bar",
                      "The epoch is chosen on a fixed 10% slice of train (345 rows); dev never selects anything. The only "
                      "parameter fitted on dev is tau, by 5-fold nested cross-validation, so reported scores are held out. "
                      "Seeds 0-9 fix head initialisation and data order, so they are paired across configurations. Each "
                      "variant is screened on three seeds against a bar fixed in advance (+0.015 with 2 of 3 seeds up). "
                      "Source: Report §2, docs/02_experiment_log.md.")
    bullets(s, ["Epoch chosen on a fixed 345-row train slice; dev never selects anything.",
                "Only dev-fitted parameter is **τ**, chosen by 5-fold nested cross-validation.",
                "Seeds 0–9 paired across configurations: same head initialisation and data order.",
                "Hypotheses and predicted outcomes logged **before** every run."])
    panel(s)
    g = s.shapes.add_group_shape()
    g.name = "Diagram: protocol"
    sh = g.shapes
    mx, mw = CARD_X + 0.35, 3.95
    b1 = node(sh, mx, 1.82, mw, 0.5, "Register hypothesis and predicted outcome", size=12)
    b2 = node(sh, mx, 2.57, mw, 0.5, "Change one component against a named control", size=12)
    b3 = node(sh, mx, 3.32, mw, 0.5, "Screen on 3 paired seeds", size=12, fill=ACCENT_SOFT, line=ACCENT)
    b4 = node(sh, mx, 4.3, mw, 0.5, "Extend to 10 paired seeds", size=12)
    b5 = node(sh, mx, 5.05, mw, 0.5, "System: 10-seed ensemble + logit adjustment", size=12)
    b6 = node(sh, mx, 5.8, mw, 0.55, "Report paired Δ, t, seeds up,\nbootstrap 95% CI, 2-annotator rescoring", size=11,
              bold_first=False)
    nx = mx + mw + 0.55
    neg = node(sh, nx, 3.24, CARD_X + CARD_W - 0.3 - nx, 0.66, "Log as negative,\nwith its mechanism", size=11, fill=CODE_BG,
               line=MUTED, bold_first=False)
    for a, b in ((b1, b2), (b2, b3), (b3, b4), (b4, b5), (b5, b6)):
        arrow(sh, a, 2, b, 0)
    arrow(sh, b3, 3, neg, 1)
    edge_label(sh, mx + mw / 2 + 0.1, 3.92, 2.3, "gain ≥ +0.015, 2 of 3 seeds up", align=L, size=10)
    edge_label(sh, mx + mw + 0.02, 3.3, 0.5, "below", size=10)
    caption(s, nx, 4.05, CARD_X + CARD_W - 0.3 - nx, "The bar is fixed before each run.", h=0.4)

    # 6. Iteration 1: E0 ---------------------------------------------------------------------
    s = content_slide(prs, layout, "Iteration 1  ·  Steps 0–2, E0", "A sound baseline took a library fix and three attempts",
                      "The first runs did not learn: transformers 5.x loads DeBERTa-v3 in its stored fp16 weights, "
                      "attention overflowed and the loss sat at 1.887 while the model predicted Explicit for everything. "
                      "Loading in fp32 fixed it. A 5-epoch rerun scored 0.223 because the cosine schedule spans the run: "
                      "a budget problem, not a bad idea, a pattern that recurs. The 8-epoch baseline is 0.337 +/- 0.044 on "
                      "five seeds. Sources: docs/02_experiment_log.md Steps 1-2 and E0; docs/raw/E0_analysis.txt.")
    bullets(s, ["DeBERTa-v3-large: sub-question + answer, 512 tokens, 8 epochs, plain cross-entropy.",
                "Library loaded fp16 weights; attention overflowed, loss stuck at 1.887.",
                "Fixed by an explicit fp32 load: baseline S2 **0.337 ± 0.044**, 5 seeds.",
                "A 5-epoch rerun scored 0.223: schedule truncation, not a bad idea."])
    xi, yi, wi = band_panel(s, 4.1)
    code_block(s, xi, yi, wi, 1.45,
               "# code/models/encoder.py:176\n# transformers 5.x loads DeBERTa-v3 in its stored fp16;\n"
               "# the model then predicts \"Explicit\" for every item.\n"
               "self.enc = AutoModel.from_pretrained(\n    name, dtype=torch.float32)", size=12)
    table(s, xi, yi + 1.75, [wi / 6] * 6, [
        ["Seed", "0", "1", "2", "3", "4"],
        ["Dev S2", "0.389", "0.339", "0.279", "0.311", "0.368"],
        ["Dev S1", "0.642", "0.580", "0.532", "0.596", "0.579"],
    ], size=14, row_h=0.4, align=[L] + [R] * 5)
    text_box(s, xi, yi + 3.1, wi, 0.4, "Mean  S2 **0.337 ± 0.044**   ·   S1 **0.586 ± 0.039**", size=15)
    caption(s, xi, yi + 3.65, wi, "E0c, docs/raw/E0_analysis.txt. A seed spread of 0.28–0.39 forces paired multi-seed "
                                  "comparisons from here on.")

    # 7. Iteration 2: decision rules ---------------------------------------------------------
    s = content_slide(prs, layout, "Iteration 2  ·  E2–E4", "The ranking is better than the decision",
                      "Averaging the five baseline seeds lifts S2 from 0.337 to 0.365. An acceptable label is in the top "
                      "three for 90% of items, an oracle of 0.745, so the decision is the weak point. One-scalar logit "
                      "adjustment reaches 0.438 held out and barely overfits (in-fold minus held-out 0.018). Nine per-class "
                      "weights, the metric-optimal rule in theory, overfit 308 items by 0.098. Logit adjustment joins every "
                      "later system. Source: docs/02_experiment_log.md E2-E4.")
    bullets(s, ["5-seed ensemble lifts S2 from 0.337 to 0.365.",
                "Acceptable label in the top 3 for 90% of items (oracle 0.745).",
                "One-scalar logit adjustment: **0.438** held out, overfit gap only 0.018.",
                "Nine per-class weights overfit 308 items: gap **+0.098**."])
    figure(s, "decision_rules", "E4 on the E0 5-seed ensemble; every score held out by nested CV. Hatched: the rule kept "
                                "for every later system.")

    # 8. Iteration 3: negatives --------------------------------------------------------------
    s = content_slide(prs, layout, "Iteration 3  ·  E5–E12", "Structure, rebalanced losses and soups do not help",
                      "Ten variants, each against its own named control; none clears the +0.015 screening bar. Three "
                      "hierarchies fail because a flat softmax already scores all nine labels jointly. The definition "
                      "re-ranker learns from the same 3,448 examples, so it has no knowledge the first model lacks. Weight "
                      "soups fail because seeds differ in head initialisation and data order. Balanced Softmax with focal "
                      "loss over-corrects (Explicit F1 0.679 to 0.319); note E8 also changed the input. "
                      "Source: docs/02_experiment_log.md E4-E12, Report §4.")
    bullets(s, ["Three hierarchies (routing, factorised head, gate + specialists): none beats flat softmax.",
                "Definition re-ranker 0.354 vs. 0.365: same 3,448 examples, no new knowledge.",
                "Weight soups fail: seeds differ in head initialisation and data order.",
                "Balanced Softmax + focal over-corrects: Explicit F1 falls 0.679 to 0.319."])
    figure(s, "negative_results", "Each variant against its own named control (docs/02_experiment_log.md); the comparison "
                                  "level is printed beside each bar.")

    # 9. Iteration 4: undertraining ----------------------------------------------------------
    s = content_slide(prs, layout, "Iteration 4  ·  E7–E10", "A negative result that was really undertraining",
                      "E7: ChulaNLP's comparable DeBERTa (0.46) sees the full journalist question, which ours did not. "
                      "Adding it at 8 epochs (E8b) scored below the baseline, 0.282 against 0.337. The run reports showed "
                      "why: final train loss 1.32-1.50 against 0.57-1.09, and low confidence (mean top probability 0.445 "
                      "vs 0.594). At 16 epochs (E10) the same input gives the best single model. E8's first diagnosis, the "
                      "loss, was revised after this one-change control. Source: docs/02_experiment_log.md E7-E10; per-seed "
                      "scores from runs/*/seed*/metrics.json.")
    bullets(s, ["E7: a comparable published DeBERTa (0.46) sees the full journalist question.",
                "E8b: full question at 8 epochs scored **below** baseline (0.282 vs. 0.337).",
                "Diagnosis: train loss 1.32–1.50, low confidence (top-p 0.445 vs. 0.594).",
                "E10: 16 epochs reverse it; full question gives the best single model."])
    figure(s, "undertraining", "Single-model dev S2, seeds 0–4, paired: E0 → E10 control (a), E8b → E10 (b).")

    # 10. Iteration 5: replication -----------------------------------------------------------
    s = content_slide(prs, layout, "Iteration 5  ·  E10–E11", "Per-model gains replicate; five-seed systems do not",
                      "E11 reruns all three configurations on seeds 5-9, giving ten paired seeds. Per model the two changes "
                      "add +0.069 S2 and +0.038 S1, each on 9 of 10 seeds. Systems are another matter: the baseline system "
                      "scored 0.438 on seeds 0-4 and 0.356 on seeds 5-9, and E10's 0.478 fell to 0.434. So all system claims "
                      "use ten seeds. The final DeBERTa system, chosen by a rule fixed in advance, is full question + 16 "
                      "epochs: 0.405 S2, 0.648 S1. Source: docs/raw/E11_replication.txt; Report Table 2, Figure 2.")
    bullets(s, ["E11 reruns all three configurations on seeds 5–9: ten paired seeds.",
                "Per model: S2 0.315 → **0.384**, S1 0.576 → 0.614, 9 of 10 seeds up.",
                "Five-seed systems swing 0.08; E10's 0.478 fell to 0.434 on new seeds.",
                "Final DeBERTa system, by pre-registered rule: **0.405 S2, 0.648 S1**."])
    figure(s, "deberta_seeds", "Report Figure 2. Open circles: one seed, paired across configurations; squares: mean ± 1 s.d. "
                               "over 10 seeds.")

    # 11. Iteration 6: backbone swap ---------------------------------------------------------
    s = content_slide(prs, layout, "Iteration 6  ·  E13", "Swapping only the backbone tests the knowledge hypothesis",
                      "With the decision layer saturated and the error analysis pointing at the model, two explanations "
                      "remain: knowledge or label meaning. A larger pretrained classifier tests knowledge with nothing else "
                      "changed: same 3,103 rows, input, 345-row slice, loss and selection rule. LoRA r=16, alpha 32, on all "
                      "attention and MLP projections; a new 9-way head on the last token; a padding self-check aborts any "
                      "run whose head reads a pad token. Predictions were registered before the run. Sources: "
                      "docs/02_experiment_log.md §E13, docs/raw/E13_final_analysis.txt, code/models/llm_classifier.py:124.")
    bullets(s, ["Remaining hypothesis: the 0.4B encoder lacks **knowledge**, not rule or labels.",
                "Swap DeBERTa for Qwen3-8B-Base with LoRA r = 16 on all linear layers.",
                "Same rows, input, slice, loss and selection; a padding self-check guards the head.",
                "Screen +0.081 (3/3), then ten seeds: S2 **+0.092**, S1 **+0.095**, 10/10."])
    xi, yi, wi = band_panel(s, 0.4 + 6 * 0.46 + 0.15 + 0.42)
    text_box(s, xi, yi, wi, 0.3, "Registered before the run, checked after", size=13, color=MUTED, bold=True)
    table(s, xi, yi + 0.4, [3.0, 2.6, 1.23], [
        ["Prediction", "Observed", "Verdict"],
        ["Passes the 3-seed screen", "+0.081, 3 of 3 seeds", "held"],
        ["Single-model S2 in 0.43–0.50", "0.476 ± 0.043", "held"],
        ["S1 gain of +0.02 to +0.05", "+0.095", "exceeded"],
        ["Gain on the commitment boundary", "largest on Non-Reply types", "**wrong**"],
        ["Larger gain on agreed items", "+0.076 agreed, +0.105 split", "**wrong**"],
    ], size=13, row_h=0.46, align=[L, L, L])
    caption(s, xi, yi + 0.4 + 6 * 0.46 + 0.15, wi, "Predictions: docs/02_experiment_log.md §E13. Outcomes: "
                                                   "docs/raw/E13_final_analysis.txt (in-set rate by annotator agreement).")

    # 12. Iteration 7: per-class -------------------------------------------------------------
    s = content_slide(prs, layout, "Iteration 7  ·  E13", "The 8B gain is broad, not on the expected boundary",
                      "Per-class F1 rises in 7 of 9 classes. The largest gains are on the Non-Reply types, Claims ignorance "
                      "+0.403 and Declining +0.220, which depend on recognising what kind of statement an answer is. The "
                      "commitment classes we expected to gain most moved least. Clarification falls, but it has only 4 dev "
                      "reference sets, and neither model ever predicts Partial/half-answer. Source: Report Table 4; "
                      "docs/raw/E13_final_analysis.txt section D.")
    bullets(s, ["Per-class F1 rises in 7 of 9 classes.",
                "Largest gains on Non-Reply types: Claims ignorance **+0.403**, Declining **+0.220**.",
                "Commitment classes move least: Implicit +0.040, Explicit +0.057.",
                "In-set gain similar on agreed (+0.076) and split items (+0.105)."])
    figure(s, "per_class_gain", "Report Table 4: single models averaged over the same 10 seeds; dashed lines separate the "
                                "clarity levels.")

    # 13. Iteration 8: follow-ups ------------------------------------------------------------
    s = content_slide(prs, layout, "Iteration 8  ·  E13b, E13c", "Longer training adopted; all-of-train unresolved",
                      "In the 3-epoch runs the last epoch was always selected, which under a cosine schedule proves little. "
                      "With 12 epochs (seeds 0-2) the slice selected epochs 8-11 and slice F1 rose +0.084 on all three, so "
                      "the registered rule adopted 12 epochs; dev S2 0.543 is reported, not used. Training on all 3,448 rows "
                      "gave +0.017 per model on 6 of 10 seeds; its system interval spans zero and S1 fell 0.746 to 0.711. "
                      "Mixing Qwen and DeBERTa probabilities adds nothing. Source: Report Table 3; docs/raw/E13_final_analysis.txt.")
    bullets(s, ["12 epochs: slice F1 **+0.084** on all three seeds; adopted by registered rule.",
                "Selected epochs 8–11; train loss near 0.02 with no slice decline.",
                "All of train: +0.017 per model (6/10); system interval spans zero.",
                "Mixing Qwen and DeBERTa probabilities adds nothing (0.543 vs. 0.549)."])
    xi, yi, wi = band_panel(s, 4 * 0.5 + 0.25 + 0.65 + 0.2 + 0.42)
    table(s, xi, yi, [2.5, 1.3, 1.3, 0.85, 0.88], [
        ["Follow-up (one change)", "Variant", "Qwen, 3 ep.", "Δ", "Seeds up"],
        ["12 epochs, single, S2 ‡", "0.543 ± 0.015", "0.462 ± 0.080", "+0.082", "†"],
        ["All of train, single, S2", "0.492 ± 0.045", "0.476 ± 0.043", "+0.017", "6/10"],
        ["All of train, system, S2", "0.575", "0.543", "+0.028", "–"],
    ], size=13, row_h=0.5, align=[L, R, R, R, C])
    text_box(s, xi, yi + 2.25, wi, 0.65, "All-of-train system difference: 95% CI **[−0.026, +0.092]**.\n"
                                         "As with the encoder, its 3-seed screen (+0.039) overstated the gain.", size=14)
    caption(s, xi, yi + 3.1, wi, "Report Table 3 (bottom). ‡ seeds 0–2. † reported only: the 12-epoch choice was made on the "
                                 "train slice, not on dev.")

    # 14. Benchmark: table -------------------------------------------------------------------
    s = content_slide(prs, layout, "Benchmark results", "The backbone swap outweighs the whole encoder track",
                      "Per model, Qwen adds +0.092 S2, more than the entire encoder track gained (+0.069 from 0.315 to "
                      "0.384). The 10-seed system gain is +0.136 with a bootstrap 95% CI of [+0.054, +0.224], and it holds on "
                      "the three two-annotator reference sets that mimic the test regime (0.524 vs 0.382). Logit adjustment "
                      "matters more for Qwen, whose raw predictions lean to frequent classes (General predicted for 24 items "
                      "but in 113 reference sets). System numbers here use the first nested-CV split. Sources: Report Table "
                      "3; docs/raw/E13_final_analysis.txt; README §6.")
    bullets(s, ["Backbone swap alone exceeds the encoder track's gain (+0.092 vs. +0.069).",
                "System S2 **+0.136**, 95% CI [+0.054, +0.224].",
                "Holds on 2-annotator reference sets: 0.524 vs. 0.382.",
                "Logit adjustment adds +0.062 per Qwen model, +0.005 for DeBERTa."])
    xi, yi, wi = band_panel(s, 8 * 0.46 + 0.15 + 0.42)
    table(s, xi, yi, [4.13, 1.35, 1.35], [
        ["System (dev)", "S2", "S1"],
        ["DeBERTa baseline, single model", "0.315 ± 0.040", "0.576 ± 0.030"],
        ["DeBERTa, full question + 16 ep., single", "0.384 ± 0.030", "0.614 ± 0.027"],
        ["DeBERTa, 10-seed system", "0.405", "0.648"],
        ["Qwen3-8B + LoRA, single model", "0.476 ± 0.043", "0.709 ± 0.031"],
        ["**Qwen3-8B + LoRA, 10-seed system**", "**0.543**", "**0.746**"],
        ["TeleAI, 1st place (multi-call)", "0.617", "0.812"],
        ["Human annotator vs. other two", "0.684", "–"],
    ], size=14, row_h=0.46, align=[L, R, R])
    caption(s, xi, yi + 8 * 0.46 + 0.15, wi, "Single model: mean ± s.d. over 10 seeds. System: 10-seed ensemble + logit "
                                             "adjustment, first nested-CV split.")

    # 15. Benchmark: figure (full width) -----------------------------------------------------
    s = content_slide(prs, layout, "Benchmark results", "Between published single-pass and multi-call systems",
                      "Panel (a): every Qwen seed beats its paired DeBERTa seed. Panel (b): the systems against published dev "
                      "results. Bars here are the mean over 10 nested-CV splits (0.412 and 0.549), slightly different from "
                      "the first-split numbers in the table. The Qwen system sits above TeleAI's fine-tuned Qwen2.5-7B and "
                      "ChulaNLP's hybrid, and 0.074 below TeleAI's winning pipeline, while making no LLM calls. Those systems "
                      "follow other protocols. Source: Report Figure 3.")
    figure(s, "qwen_vs_deberta", "Report Figure 3. (b) bars: mean over 10 nested-CV splits; horizontal lines: published dev "
                                 "results.", x=0.6, w=12.13, bottom=6.25)
    text_box(s, 0.6, 6.4, 12.13, 0.4, "Above TeleAI's fine-tuned 7B (0.495) and ChulaNLP's hybrid (0.52); **0.074 below the "
                                      "1.94-call winner, with zero LLM calls.**", size=16)

    # 16. Failure analysis -------------------------------------------------------------------
    s = content_slide(prs, layout, "Failure analysis", "The model fails, not the labels",
                      "Of the final DeBERTa ensemble's 308 predictions, 137 miss the reference set, and 53 of those are on "
                      "items all three annotators agreed on. Scored like the test set, each annotator against the other two "
                      "reaches 0.643-0.766 (mean 0.684), the ensemble 0.380. The biggest error pairs concern how committed "
                      "an answer is. Long inputs hurt: the 20 inputs still cut at 1,024 tokens land in-set 20% of the time. "
                      "The matrix uses majority gold; the error-pair counts use reference sets. Source: Report §6; "
                      "docs/raw/mideval/mideval_analysis.txt, confusion_final.csv.")
    bullets(s, ["137 of 308 predictions miss the reference set; 53 on unanimous items.",
                "Annotator vs. other two: 0.684; ensemble: 0.380.",
                "Top error pairs: Implicit→Explicit 25, General→Implicit 20, General→Explicit 20.",
                "Length hurts: the 20 inputs still truncated land in-set 20% vs. 58%."])
    figure(s, "confusion", "Final DeBERTa 10-seed ensemble, argmax, rows = majority gold. Bold: correct.")

    # 17. Failure example + limitations ------------------------------------------------------
    s = content_slide(prs, layout, "Failure example and limitations", "One answer that needs world knowledge",
                      "Dev item 17 was fixed in advance as the error to discuss. Reading 'I'm going to stay out of "
                      "Connecticut' as a near-answer needs to know Lieberman is Connecticut's senator; this is our reading, "
                      "not a measurement. Qwen ranks the acceptable Dodging first, but logit adjustment divides by the train "
                      "prior (0.205 for Dodging, 0.042 for Declining) and moves the decision to the rarer class. Limitations: "
                      "dev-only evaluation, moderate label agreement (Fleiss kappa 0.48 on nine types), a 12-epoch schedule "
                      "screened on 3 seeds, possible pretraining exposure to the public transcripts, and Partial/half-answer "
                      "never predicted. Source: Report Figure 4 and Limitations.")
    bullets(s, ["Dev-only evaluation: 308 items, so system differences carry wide intervals.",
                "Labels moderately reliable: Fleiss κ 0.48 on nine types; human ceiling 0.684.",
                "12-epoch schedule screened on three seeds only: adopted, not established.",
                "Partial/half-answer never predicted; 8B pretraining may include these transcripts."], size=17)
    xi, yi, wi = band_panel(s, 4 * 0.38 + 0.2 + 4 * 0.38 + 0.55 + 0.15 + 0.4)
    table(s, xi, yi, [1.5, wi - 1.5], [
        ["Dev item 17", ""],
        ["Sub-question", "Would you campaign against Senator Joe Lieberman on Iraq?"],
        ["Answer", "“I’m going to stay out of Connecticut.”"],
        ["Annotators", "Dodging, Dodging, Implicit  (S1: Ambivalent)"],
    ], size=13, row_h=0.38)
    y2 = yi + 4 * 0.38 + 0.2
    table(s, xi, y2, [1.75, (wi - 1.75) / 2, (wi - 1.75) / 2], [
        ["", "DeBERTa system", "Qwen system"],
        ["Mean p, top 3", "General .373, Implicit .282, Deflection .205", "Dodging .327, Implicit .189, Declining .170"],
        ["Ensemble argmax", "General (miss)", "**Dodging (in-set)**"],
        ["After logit adj.", "General (miss)", "Declining (miss)"],
        ["Subtask 1", "**Ambivalent (correct)**", "Clear Non-Reply (miss)"],
    ], size=12, row_h=0.38, align=[L, L, L])
    caption(s, xi, y2 + 4 * 0.38 + 0.55 + 0.15, wi, "Report Figure 4. The prior correction (0.205 vs. 0.042) moves Qwen's "
                                                    "correct top-1 to a rarer class.", h=0.4)

    # 18. Summary ----------------------------------------------------------------------------
    s = content_slide(prs, layout, "Summary and contributions", "What the mid-evaluation establishes",
                      "Four contributions. First, a controlled DeBERTa system whose two real improvements, longer training "
                      "and the full question, are separated from seven documented failures. Second, evidence that the "
                      "remaining gap is the model, not label noise or the decision rule. Third, a one-change test of the "
                      "knowledge explanation: the 8B backbone improves both subtasks on all ten paired seeds. Fourth, an "
                      "analysis showing our own registered prediction about where the gain falls was wrong. "
                      "Source: Report §1, §7.")
    bullets(s, ["Encoder limits isolated: only longer training and the full question helped.",
                "Seven documented negatives (hierarchies, losses, re-ranker, soups), each with a mechanism.",
                "Knowledge hypothesis supported: the 8B backbone wins on all ten paired seeds.",
                "Our registered “where” prediction was falsified: the gain is broad."])
    cw, chh, gap = (CARD_W - 0.3) / 2, 1.8, 0.3
    y0 = (TOP + BOTTOM) / 2 - (2 * chh + gap) / 2
    stat_card(s, CARD_X, y0, cw, chh, "DeBERTa-v3-large, 10-seed system", "0.405", "dev S2  ·  S1 0.648")
    stat_card(s, CARD_X + cw + gap, y0, cw, chh, "Qwen3-8B + LoRA, 10-seed system", "0.543", "dev S2  ·  S1 0.746", accent=True)
    stat_card(s, CARD_X, y0 + chh + gap, cw, chh, "System gain, Subtask 2", "+0.136", "95% CI [+0.054, +0.224]")
    stat_card(s, CARD_X + cw + gap, y0 + chh + gap, cw, chh, "Acceptable label in Qwen top 3", "93.1%", "top 5: 99.1%")

    # 19. Bridge: ranking vs decision --------------------------------------------------------
    s = content_slide(prs, layout, "From results to plan", "Qwen ranks well and knows when it is unsure",
                      "For Qwen (seeds 0-2) an acceptable label is first for 60.4% of items, in the top three for 93.1% and "
                      "the top five for 99.1%. The in-set rate climbs from 0.40 in the least confident fifth of dev to 0.87 "
                      "in the most confident. So most remaining errors are a choice among a few candidates the model already "
                      "ranks highly, and confidence tells us which items those are. That is the case for a cascade that "
                      "spends LLM calls only on uncertain items. Source: docs/raw/E13_Q8_topk_confidence.txt.")
    bullets(s, ["Acceptable label in Qwen's top 3 for **93.1%** of items; top 5, 99.1%.",
                "In-set rate rises from 0.40 to **0.87**, least to most confident fifth.",
                "Most errors are choices among a few highly ranked candidates.",
                "So an LLM is needed only on uncertain items, not on all."])
    figure(s, "ranking_vs_decision", "Qwen3-8B + LoRA, seeds 0–2, dev (docs/raw/E13_Q8_topk_confidence.txt). Exploratory; "
                                     "dev chose nothing.")

    # 20. Roadmap ----------------------------------------------------------------------------
    s = content_slide(prs, layout, "Roadmap", "A confidence-routed cascade by October 31",
                      "Four modules, each added as one change under the same protocol. M1: the 12-epoch Qwen classifier on "
                      "ten seeds returns a calibrated candidate set C(x) holding the gold label for 95% of held-out items, "
                      "plus an uncertainty score u(x). M3: a local open-weight LLM scores only the labels in C(x), with "
                      "definitions, a confusion guide and boundary examples. M2: route the fraction delta with the highest "
                      "u(x) to M3; sweep delta for an accuracy-versus-calls curve. M4: distil M3 back into M1 and run 4-bit. "
                      "Target: at least 0.60 dev S2 with at most 0.3 LLM calls per item, a sixth of the winner's. "
                      "Source: Report §7, Figure 5, Table 5.")
    bullets(s, ["**Oct 5–11, M1:** 12-epoch Qwen on ten seeds; calibrated candidate set C(x).",
                "**Oct 12–18, M3:** local LLM scores only C(x), with definitions and boundary examples.",
                "**Oct 19–25, M2:** uncertainty router; sweep deferral δ for accuracy vs. calls.",
                "**Oct 26–31, M4:** distil into M1, 4-bit inference; freeze and final scores."],
            y=TOP, h=4.0, size=17, gap=12)
    tgt = panel(s, LEFT_X, 5.75, LEFT_W, 0.95, fill=SAGE_SOFT, line=SAGE, name="Target")
    tf = tgt.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.2)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    add_runs(tf.paragraphs[0], "**Target:** at least 0.60 dev S2 with at most 0.3 LLM calls per item.", 16, TITLE)
    panel(s)
    g = s.shapes.add_group_shape()
    g.name = "Diagram: planned cascade (Report Figure 5)"
    sh = g.shapes
    lx, lw = CARD_X + 0.3, 1.8             # left column
    cx, cw2 = lx + lw + 0.4, 2.35          # centre column
    rx = cx + cw2 + 0.45                   # right column
    rw = CARD_X + CARD_W - 0.3 - rx
    n_in = node(sh, cx, 1.8, cw2, 0.48, "Input (q, Q, a)", size=12)
    m1 = node(sh, cx, 2.6, cw2, 0.95, "M1 Candidate generator\nQwen3-8B + LoRA, K seeds,\nlogit adj. → C(x), u(x)", size=11,
              fill=ACCENT_SOFT, line=ACCENT)
    m2 = node(sh, cx, 3.95, cw2, 0.62, "M2 Router\ndefer the δ most uncertain", size=11, dashed=True)
    out = node(sh, cx, 5.55, cw2, 0.55, "Subtask 2 → Subtask 1", size=12)
    m3 = node(sh, rx, 3.75, rw, 1.05, "M3 LLM adjudicator\nscores only C(x):\ndefinitions, boundary\nexamples", size=10,
              dashed=True)
    m4 = node(sh, lx, 2.5, lw, 1.15, "M4 Distillation\nM3's train decisions\nas soft targets;\n4-bit inference", size=10,
              dashed=True)
    arrow(sh, n_in, 2, m1, 0)
    arrow(sh, m1, 2, m2, 0)
    arrow(sh, m2, 2, out, 0)
    arrow(sh, m2, 3, m3, 1)
    arrow(sh, m4, 3, m1, 1, dashed=True)
    m3c, out_y, m1_y = rx + rw / 2, 5.55 + 0.275, 2.6 + 0.475
    polyline(sh, [(m3c, 4.8), (m3c, out_y), (cx + cw2, out_y)])
    polyline(sh, [(m3c, 3.75), (m3c, m1_y), (cx + cw2, m1_y)], dashed=True)
    edge_label(sh, cx + cw2 + 0.05, m1_y - 0.3, rw + 0.3, "soft targets (train)", align=C, size=10)
    edge_label(sh, cx + cw2 / 2 - 1.3, 4.95, 1.2, "keep, 1 − δ", align=R, size=10)
    edge_label(sh, cx + cw2 + 0.02, 4.0, 0.4, "δ", size=11)
    caption(s, lx, 6.3, CARD_W - 0.6, "Redrawn from Report Figure 5. Dashed: planned, none run yet. LLM calls per item = δ.",
            h=0.4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    print("wrote", build())
