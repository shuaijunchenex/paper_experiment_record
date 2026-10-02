"""Small shared helpers for data-backed booktabs tables and PDF previews."""
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

DATASETS = ("cifar10", "mnli", "qqp")
LABELS = {"cifar10": "CIFAR10", "mnli": "MNLI", "qqp": "QQP"}


def verify_sources(root):
    provenance = json.loads((root / "sources/provenance.json").read_text(encoding="utf-8"))
    for item in provenance["files"]:
        assert hashlib.sha256((root / "sources" / item["name"]).read_bytes()).hexdigest() == item["sha256"]
    return provenance


def value_text(value):
    if 0 < value < .005:
        return "<0.01"
    return f"{value:.2f}"


def render_pdf(root, stem, title, header, rows, widths, notes, *, spans, separators=(), bold_cells=()):
    font_dir = Path("C:/Windows/Fonts")
    pdfmetrics.registerFont(TTFont("TableTimes", str(font_dir / "times.ttf")))
    pdfmetrics.registerFont(TTFont("TableTimesBold", str(font_dir / "timesbd.ttf")))
    pdfmetrics.registerFontFamily("TableTimes", normal="TableTimes", bold="TableTimesBold")
    title_style = ParagraphStyle("title", fontName="TableTimesBold", fontSize=11.5, leading=14)
    note_style = ParagraphStyle("notes", fontName="TableTimes", fontSize=8.5, leading=10.7)
    title_paragraph = Paragraph(escape(title), title_style)
    note_paragraph = Paragraph(escape(notes), note_style)
    width = sum(widths)
    _, title_height = title_paragraph.wrap(width, 1000)
    _, note_height = note_paragraph.wrap(width, 1000)
    data = header + rows
    table = Table(data, colWidths=widths)
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "TableTimes"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("LEADING", (0, 0), (-1, -1), 11.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (2, 0), (-1, 1), "CENTER"),
        ("ALIGN", (2, 2), (-1, -1), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEABOVE", (0, 0), (-1, 0), .8, colors.black),
        ("LINEBELOW", (0, 1), (-1, 1), .5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), .8, colors.black),
    ]
    for start, end in spans:
        commands.extend([("SPAN", (start, 0), (end, 0)),
                         ("LINEBELOW", (start, 0), (end, 0), .35, colors.black)])
    for index in separators:
        commands.append(("LINEABOVE", (0, index + 2), (-1, index + 2), .35, colors.black))
    for column, index in bold_cells:
        commands.append(("FONTNAME", (column, index + 2), (column, index + 2), "TableTimesBold"))
    table.setStyle(TableStyle(commands))
    _, table_height = table.wrap(width, 2000)
    height = 16 + title_height + 9 + table_height + 9 + note_height + 16
    pdf = canvas.Canvas(str(root / f"{stem}.pdf"), pagesize=(width + 36, height))
    pdf.setTitle(title)
    pdf.setAuthor("FLEX experiment records")
    cursor = height - 16 - title_height
    title_paragraph.drawOn(pdf, 18, cursor)
    cursor -= 9 + table_height
    table.drawOn(pdf, 18, cursor)
    cursor -= 9 + note_height
    note_paragraph.drawOn(pdf, 18, cursor)
    pdf.showPage()
    pdf.save()


def write_metadata(root, provenance, details):
    details.update({"source_batch": provenance["batch"], "source_files": provenance["files"],
                    "source_hashes_verified": True, "new_experiments_run": False,
                    "pdf_renderer": "ReportLab using the same formatted cells as the LaTeX table",
                    "latex_compiled_locally": False,
                    "latex_dependencies": ["booktabs"]})
    (root / "table_metadata.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
