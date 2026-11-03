import re, sys
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

SRC = sys.argv[1] if len(sys.argv) > 1 else "novelty_check.md"
DST = sys.argv[2] if len(sys.argv) > 2 else SRC.rsplit(".", 1)[0] + ".docx"

doc = Document()
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(11)
from docx.oxml.ns import qn
style.element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")

def add_runs(p, text):
    # tokenize **bold**, *italic*, `code`, [label](url)
    token = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`|\[[^\]]+\]\([^)]+\))")
    for part in token.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = p.add_run(part[2:-2]); r.bold = True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            r = p.add_run(part[1:-1]); r.italic = True
        elif part.startswith("`") and part.endswith("`"):
            r = p.add_run(part[1:-1]); r.font.name = "Menlo"; r.font.size = Pt(10)
        else:
            m = re.match(r"\[([^\]]+)\]\(([^)]+)\)", part)
            if m:
                r = p.add_run(m.group(1)); r.font.color.rgb = RGBColor(0x1a, 0x56, 0xb0); r.underline = True
            else:
                p.add_run(part)

lines = open(SRC).read().splitlines()
i = 0
while i < len(lines):
    line = lines[i]
    if line.startswith("# "):
        h = doc.add_heading(re.sub(r"`", "", line[2:]), level=0)
    elif line.startswith("## "):
        doc.add_heading(line[3:], level=1)
    elif line.startswith("### "):
        doc.add_heading(line[4:], level=2)
    elif line.startswith("|"):
        # collect table block
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            cells = [c.strip() for c in lines[i].strip("|").split("|")]
            if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                rows.append(cells)
            i += 1
        i -= 1
        t = doc.add_table(rows=len(rows), cols=len(rows[0]))
        t.style = "Light Grid Accent 1"
        for ri, row in enumerate(rows):
            for ci, cell in enumerate(row[:len(t.columns)]):
                c = t.cell(ri, ci)
                c.text = ""
                add_runs(c.paragraphs[0], re.sub(r"\*\*(.+?)\*\*", r"\1", cell) if ri == 0 else cell)
                if ri == 0:
                    for r in c.paragraphs[0].runs:
                        r.bold = True
    elif re.match(r"^\d+\. ", line):
        p = doc.add_paragraph(style="List Number")
        add_runs(p, re.sub(r"^\d+\. ", "", line))
    elif line.startswith("- "):
        p = doc.add_paragraph(style="List Bullet")
        add_runs(p, line[2:])
    elif line.strip() == "":
        pass
    else:
        p = doc.add_paragraph()
        add_runs(p, line)
    i += 1

doc.save(DST)
print("saved", DST)
