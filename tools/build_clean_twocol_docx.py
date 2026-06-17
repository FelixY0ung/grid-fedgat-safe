#!/usr/bin/env python3
"""Create a cleaner two-column ICPICN DOCX candidate from Pandoc output.

Pandoc preserves useful Word math, but the ICPICN template's automatic heading
numbering leaks orphan Roman numerals when a blind two-column section is forced.
This post-process keeps the front matter one-column, starts the body in two
columns, maps Pandoc's undefined paragraph styles to template styles, and writes
manual section numbers so LibreOffice/Word cannot emit stray numbering.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "w14": "http://schemas.microsoft.com/office/word/2010/wordml",
}

for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)

W = "{%s}" % NS["w"]
M = "{%s}" % NS["m"]
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


def w_attr(name: str) -> str:
    return W + name


def child(parent: ET.Element, tag: str) -> ET.Element | None:
    return parent.find("w:" + tag, NS)


def para_text(p: ET.Element) -> str:
    return "".join(t.text or "" for t in p.findall(".//w:t", NS)).strip()


def set_para_text(p: ET.Element, text: str) -> None:
    for r in list(p.findall("w:r", NS)):
        p.remove(r)
    r = ET.SubElement(p, W + "r")
    t = ET.SubElement(r, W + "t")
    t.set(XML_SPACE, "preserve")
    t.text = text


def get_ppr(p: ET.Element) -> ET.Element:
    ppr = child(p, "pPr")
    if ppr is None:
        ppr = ET.Element(W + "pPr")
        p.insert(0, ppr)
    return ppr


def set_pstyle(p: ET.Element, style: str) -> None:
    ppr = get_ppr(p)
    pstyle = child(ppr, "pStyle")
    if pstyle is None:
        pstyle = ET.Element(W + "pStyle")
        ppr.insert(0, pstyle)
    pstyle.set(w_attr("val"), style)


def set_page_break_before(p: ET.Element) -> None:
    ppr = get_ppr(p)
    if child(ppr, "pageBreakBefore") is None:
        ppr.append(ET.Element(W + "pageBreakBefore"))


def is_display_equation_paragraph(p: ET.Element) -> bool:
    children = [elem for elem in list(p) if elem.tag != W + "pPr"]
    return len(children) == 1 and children[0].tag == M + "oMathPara"


def column_width_from_section(sect: ET.Element) -> int:
    pg_sz = child(sect, "pgSz")
    pg_mar = child(sect, "pgMar")
    cols = child(sect, "cols")
    if pg_sz is None or pg_mar is None:
        raise RuntimeError("Section properties missing page size or margins")

    page_width = int(pg_sz.get(w_attr("w")))
    left_margin = int(pg_mar.get(w_attr("left"), "0"))
    right_margin = int(pg_mar.get(w_attr("right"), "0"))
    usable_width = page_width - left_margin - right_margin
    if cols is None:
        return usable_width

    num_cols = int(cols.get(w_attr("num"), "1"))
    if num_cols <= 1:
        return usable_width
    col_space = int(cols.get(w_attr("space"), "0"))
    return (usable_width - col_space * (num_cols - 1)) // num_cols


def set_right_tab(p: ET.Element, position: int) -> None:
    ppr = get_ppr(p)
    tabs = child(ppr, "tabs")
    if tabs is None:
        tabs = ET.Element(W + "tabs")
        ppr.append(tabs)
    tab = ET.SubElement(tabs, W + "tab")
    tab.set(w_attr("val"), "right")
    tab.set(w_attr("pos"), str(position))


def run_text(text: str) -> ET.Element:
    r = ET.Element(W + "r")
    t = ET.SubElement(r, W + "t")
    t.set(XML_SPACE, "preserve")
    t.text = text
    return r


def run_tab() -> ET.Element:
    r = ET.Element(W + "r")
    ET.SubElement(r, W + "tab")
    return r


def run_field_char(kind: str) -> ET.Element:
    r = ET.Element(W + "r")
    fld = ET.SubElement(r, W + "fldChar")
    fld.set(w_attr("fldCharType"), kind)
    return r


def run_instr(text: str) -> ET.Element:
    r = ET.Element(W + "r")
    instr = ET.SubElement(r, W + "instrText")
    instr.set(XML_SPACE, "preserve")
    instr.text = text
    return r


def add_equation_number(p: ET.Element, fallback_number: int, tab_position: int) -> None:
    set_right_tab(p, tab_position)
    p.extend(
        [
            run_tab(),
            run_text("("),
            run_field_char("begin"),
            run_instr(" SEQ Equation \\* ARABIC "),
            run_field_char("separate"),
            run_text(str(fallback_number)),
            run_field_char("end"),
            run_text(")"),
        ]
    )


def clone_section(
    base_sect: ET.Element,
    cols: dict[str, str] | None = None,
    section_type: str = "continuous",
) -> ET.Element:
    sect = ET.Element(W + "sectPr")
    typ = ET.SubElement(sect, W + "type")
    typ.set(w_attr("val"), section_type)

    for tag in ["pgSz", "pgMar", "pgNumType", "formProt", "textDirection", "docGrid"]:
        elem = child(base_sect, tag)
        if elem is not None:
            sect.append(deepcopy(elem))

    if cols is not None:
        cols_elem = ET.Element(W + "cols")
        for key, value in cols.items():
            cols_elem.set(w_attr(key), value)
        insert_at = 0
        for i, elem in enumerate(list(sect)):
            if elem.tag in {W + "type", W + "pgSz", W + "pgMar", W + "pgNumType"}:
                insert_at = i + 1
        sect.insert(insert_at, cols_elem)

    return sect


def remove_template_autonumbering(styles_root: ET.Element, numbering_root: ET.Element) -> None:
    manual_styles = {"Heading1", "Heading2", "Heading3", "Heading4", "tablehead"}
    for style in styles_root.findall("w:style", NS):
        sid = style.get(w_attr("styleId"))
        if sid in manual_styles:
            ppr = child(style, "pPr")
            if ppr is not None:
                for numpr in list(ppr.findall("w:numPr", NS)):
                    ppr.remove(numpr)

    for lvl in numbering_root.findall(".//w:lvl", NS):
        for pstyle in list(lvl.findall("w:pStyle", NS)):
            if pstyle.get(w_attr("val")) in manual_styles:
                lvl.remove(pstyle)


def normalize_paragraphs(body: ET.Element) -> None:
    paragraphs = list(body.findall("w:p", NS))
    for i, p in enumerate(paragraphs[:-1]):
        text = para_text(p)
        if text.startswith("Abstract") and len(text) <= 12:
            nxt = paragraphs[i + 1]
            set_pstyle(p, "Abstract")
            set_para_text(p, "Abstract\u2014" + para_text(nxt))
            body.remove(nxt)
            break

    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    section_idx = 0
    for p in body.findall("w:p", NS):
        ppr = child(p, "pPr")
        pstyle = child(ppr, "pStyle") if ppr is not None else None
        style = pstyle.get(w_attr("val")) if pstyle is not None else None
        text = para_text(p)

        if style == "Title":
            pstyle.set(w_attr("val"), "papertitle")
        elif style == "FirstParagraph":
            pstyle.set(w_attr("val"), "BodyText")
        elif style == "Heading1":
            if text.lower() == "references":
                pstyle.set(w_attr("val"), "Heading5")
                set_para_text(p, "References")
            else:
                prefix = roman[section_idx] if section_idx < len(roman) else str(section_idx + 1)
                clean = text
                for old_prefix in roman:
                    marker = old_prefix + ". "
                    if clean.startswith(marker):
                        clean = clean[len(marker) :]
                        break
                set_para_text(p, f"{prefix}. {clean}")
                section_idx += 1
        elif text.startswith("Keywords"):
            set_pstyle(p, "Keywords")
        elif text.startswith(("Table I.", "Table II.", "Table III.")):
            set_pstyle(p, "tablehead")
            if text.startswith("Table III."):
                set_page_break_before(p)


def normalize_tables(body: ET.Element) -> None:
    for tbl in body.findall("w:tbl", NS):
        rows = tbl.findall("w:tr", NS)
        for row_index, tr in enumerate(rows):
            style = "tablecolhead" if row_index == 0 else "tablecopy"
            for p in tr.findall(".//w:p", NS):
                set_pstyle(p, style)


def number_display_equations(body: ET.Element) -> None:
    final_sect = child(body, "sectPr")
    if final_sect is None:
        raise RuntimeError("No final sectPr found in document body")
    column_width = column_width_from_section(final_sect)
    tab_position = max(3000, column_width - 120)

    equation_index = 0
    for p in body.findall("w:p", NS):
        if is_display_equation_paragraph(p):
            equation_index += 1
            add_equation_number(p, equation_index, tab_position)


def add_two_column_body_section(body: ET.Element) -> None:
    final_sect = child(body, "sectPr")
    if final_sect is None:
        raise RuntimeError("No final sectPr found in document body")

    keyword_para = None
    for p in body.findall("w:p", NS):
        if para_text(p).startswith("Keywords"):
            keyword_para = p
            break
    if keyword_para is None:
        raise RuntimeError("No keywords paragraph found")

    front_sect = clone_section(final_sect, cols=None, section_type="continuous")
    body_sect = clone_section(
        final_sect,
        cols={"num": "2", "space": "360", "equalWidth": "true", "sep": "false"},
        section_type="continuous",
    )

    keyword_ppr = get_ppr(keyword_para)
    for old in list(keyword_ppr.findall("w:sectPr", NS)):
        keyword_ppr.remove(old)
    keyword_ppr.append(front_sect)

    body.remove(final_sect)
    body.append(body_sect)


def build(src: Path, dst: Path) -> None:
    with zipfile.ZipFile(src, "r") as zin:
        doc_root = ET.fromstring(zin.read("word/document.xml"))
        styles_root = ET.fromstring(zin.read("word/styles.xml"))
        numbering_root = ET.fromstring(zin.read("word/numbering.xml"))

        body = child(doc_root, "body")
        if body is None:
            raise RuntimeError("No document body found")

        remove_template_autonumbering(styles_root, numbering_root)
        normalize_paragraphs(body)
        normalize_tables(body)
        add_two_column_body_section(body)
        number_display_equations(body)

        replacements = {
            "word/document.xml": ET.tostring(doc_root, encoding="utf-8", xml_declaration=True),
            "word/styles.xml": ET.tostring(styles_root, encoding="utf-8", xml_declaration=True),
            "word/numbering.xml": ET.tostring(numbering_root, encoding="utf-8", xml_declaration=True),
        }

        with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = replacements.get(item.filename)
                if data is None:
                    data = zin.read(item.filename)
                zout.writestr(item, data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    args = parser.parse_args()
    build(args.src, args.dst)
    print(args.dst)


if __name__ == "__main__":
    main()
