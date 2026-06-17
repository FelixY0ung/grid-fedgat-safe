#!/usr/bin/env python3
"""Check the generated ICPICN submission candidate for hard readiness gates."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET


DOCX_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


PLACEHOLDER_PATTERNS = [
    re.compile(r"First Author Name", re.IGNORECASE),
    re.compile(r"Second Author Name", re.IGNORECASE),
    re.compile(r"Third Author Name", re.IGNORECASE),
    re.compile(r"Author \d+ Name", re.IGNORECASE),
    re.compile(r"email@example\.com", re.IGNORECASE),
    re.compile(r"Affiliation, City, Country", re.IGNORECASE),
]

META_COMMENTARY_PATTERNS = [
    re.compile(r"original project plan", re.IGNORECASE),
    re.compile(r"derivation audit", re.IGNORECASE),
    re.compile(r"not defensible", re.IGNORECASE),
    re.compile(r"in this environment", re.IGNORECASE),
    re.compile(r"API requires a token", re.IGNORECASE),
    re.compile(r"manuscript should not", re.IGNORECASE),
    re.compile(r"strong ICPICN fit", re.IGNORECASE),
    re.compile(r"weaker result", re.IGNORECASE),
]

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
DEFAULT_REQUIRED_TEXT = ["Lewei Yang", "ylw_yang@mail.ustc.edu.cn"]
EQUATION_RE = re.compile(r"\((\d+)\)")


def run_text(cmd: list[str]) -> str:
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"missing required command: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.strip() or f"command failed: {' '.join(cmd)}") from exc
    return completed.stdout


def parse_pdfinfo(text: str) -> tuple[int, int]:
    pages = None
    size = None
    for line in text.splitlines():
        if line.startswith("Pages:"):
            pages = int(line.split(":", 1)[1].strip())
        elif line.startswith("File size:"):
            size = int(line.split(":", 1)[1].strip().split()[0])
    if pages is None or size is None:
        raise RuntimeError("pdfinfo output did not include Pages and File size")
    return pages, size


def contains_placeholder(text: str) -> list[str]:
    return [pattern.pattern for pattern in PLACEHOLDER_PATTERNS if pattern.search(text)]


def contains_meta_commentary(text: str) -> list[str]:
    return [pattern.pattern for pattern in META_COMMENTARY_PATTERNS if pattern.search(text)]


def docx_equation_summary(path: Path) -> tuple[int, int, int]:
    try:
        with zipfile.ZipFile(path) as docx:
            document_xml = docx.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"could not read DOCX document XML: {path}") from exc

    root = ET.fromstring(document_xml)
    body = root.find(".//w:body", DOCX_NS)
    if body is None:
        raise RuntimeError("DOCX document has no body")

    math_paragraphs = 0
    seq_fields = 0
    seq_fields_on_math_paragraphs = 0

    for paragraph in body.findall("w:p", DOCX_NS):
        has_math_para = paragraph.find("m:oMathPara", DOCX_NS) is not None
        if has_math_para:
            math_paragraphs += 1

        instr_text = "".join(
            node.text or "" for node in paragraph.findall(".//w:instrText", DOCX_NS)
        )
        if "SEQ Equation" in instr_text:
            seq_fields += 1
            if has_math_para:
                seq_fields_on_math_paragraphs += 1

    return math_paragraphs, seq_fields, seq_fields_on_math_paragraphs


def check(args: argparse.Namespace) -> int:
    failures: list[str] = []
    warnings: list[str] = []

    source = args.source.read_text(encoding="utf-8")
    source_placeholders = contains_placeholder(source)
    if source_placeholders:
        failures.append(
            "source still contains author placeholders: " + ", ".join(source_placeholders)
        )
    source_meta = contains_meta_commentary(source)
    if source_meta:
        failures.append(
            "source still contains meta-commentary phrases: " + ", ".join(source_meta)
        )
    if not EMAIL_RE.search(source):
        failures.append("source does not contain an email-like author address")
    for required in args.require_text:
        if required not in source:
            failures.append(f"source is missing required text: {required}")

    if not args.pdf.exists():
        failures.append(f"PDF candidate is missing: {args.pdf}")
    else:
        pages, size = parse_pdfinfo(run_text(["pdfinfo", str(args.pdf)]))
        if pages < args.min_pages:
            failures.append(f"PDF has {pages} pages; expected at least {args.min_pages}")
        if pages > args.max_pages:
            failures.append(f"PDF has {pages} pages; expected at most {args.max_pages}")
        if size > args.max_bytes:
            failures.append(f"PDF is {size} bytes; expected at most {args.max_bytes}")

        pdf_text = run_text(["pdftotext", "-layout", str(args.pdf), "-"])
        pdf_placeholders = contains_placeholder(pdf_text)
        if pdf_placeholders:
            failures.append(
                "PDF text still contains author placeholders: " + ", ".join(pdf_placeholders)
            )
        pdf_meta = contains_meta_commentary(pdf_text)
        if pdf_meta:
            failures.append(
                "PDF text still contains meta-commentary phrases: " + ", ".join(pdf_meta)
            )
        if "arXiv" in source or "arxiv.org" in source.lower():
            failures.append("source still contains arXiv reference text")
        if "arXiv" in pdf_text or "arxiv.org" in pdf_text.lower():
            failures.append("PDF text still contains arXiv reference text")
        if not EMAIL_RE.search(pdf_text):
            failures.append("PDF text does not contain an email-like author address")
        for required in args.require_text:
            if required not in pdf_text:
                failures.append(f"PDF text is missing required text: {required}")

        pdf_equation_numbers = sorted({int(match) for match in EQUATION_RE.findall(pdf_text)})
        if len(pdf_equation_numbers) < 7:
            failures.append(
                "PDF text does not appear to contain all seven equation numbers: "
                + str(pdf_equation_numbers)
            )

        for label in [
            "I. INTRODUCTION",
            "II. RELATED WORK AND GAP",
            "III. MODEL AND METHOD",
            "VI. RESULTS",
            "TABLE I. MAIN SERVICE AND SAFETY EVIDENCE",
            "TABLE II. AC REPLAY AND FEASIBILITY-FILTER EVIDENCE",
            "TABLE III. DEADLINE-RESERVE SENSITIVITY",
            "REFERENCES",
        ]:
            if label not in pdf_text:
                failures.append(f"PDF text is missing expected label: {label}")

        page_number_lines = re.findall(r"(?mi)^\s*Page\s+\d+\s*$", pdf_text)
        if page_number_lines:
            warnings.append("PDF text contains explicit page-number lines")

        print(f"PDF pages: {pages}")
        print(f"PDF bytes: {size}")

    if args.docx and not args.docx.exists():
        failures.append(f"DOCX candidate is missing: {args.docx}")
    elif args.docx:
        try:
            math_count, seq_count, inline_seq_count = docx_equation_summary(args.docx)
        except RuntimeError as exc:
            failures.append(str(exc))
        else:
            if math_count != 7:
                failures.append(f"DOCX has {math_count} display equation paragraphs; expected 7")
            if seq_count != 7:
                failures.append(f"DOCX has {seq_count} equation SEQ fields; expected 7")
            if inline_seq_count != 7:
                failures.append(
                    "DOCX equation numbering is not in the same paragraph as all equations: "
                    f"{inline_seq_count}/7 inline fields"
                )
            print(
                "DOCX display equations: "
                f"{math_count}; inline SEQ Equation fields: {inline_seq_count}"
            )

    if failures:
        print("Submission readiness: FAIL")
        for failure in failures:
            print(f"- {failure}")
    else:
        print("Submission readiness: PASS")

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")

    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "ICPICN2026_paper_compact.md")
    parser.add_argument("--docx", type=Path, default=PROJECT_ROOT / "build/ICPICN2026_paper_compact_twocol_clean.docx")
    parser.add_argument("--pdf", type=Path, default=PROJECT_ROOT / "build/ICPICN2026_paper_compact_twocol_clean.pdf")
    parser.add_argument("--min-pages", type=int, default=5)
    parser.add_argument("--max-pages", type=int, default=6)
    parser.add_argument("--max-bytes", type=int, default=5_000_000)
    parser.add_argument("--require-text", action="append", default=DEFAULT_REQUIRED_TEXT)
    args = parser.parse_args()
    sys.exit(check(args))


if __name__ == "__main__":
    main()
