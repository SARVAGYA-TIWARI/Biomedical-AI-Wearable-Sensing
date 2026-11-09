#!/usr/bin/env python3
"""
Convert BTP Presentation Script & QA Dossier to a beautifully formatted .docx
Student: Sarvagya Tiwari (230122047)
Supervisor: Prof. Ritika Jain, Department of Chemistry, IIT Guwahati
"""

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

DOC_PATH = r"d:\BTP\BTP_Presentation_Script_and_QA.docx"

doc = docx.Document()

# Set standard 1-inch margins
for section in doc.sections:
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

# Colors
C_NAVY = RGBColor(0x00, 0x3B, 0x6F)
C_BLUE = RGBColor(0x00, 0x7B, 0xC0)
C_GREEN = RGBColor(0x00, 0x80, 0x40)
C_DARK = RGBColor(0x22, 0x22, 0x22)
C_GRAY = RGBColor(0x55, 0x55, 0x55)

def style_heading_1(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(18)
    run.font.bold = True
    run.font.color.rgb = C_NAVY
    return p

def style_heading_2(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(14)
    run.font.bold = True
    run.font.color.rgb = C_BLUE
    return p

def style_heading_3(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(12)
    run.font.bold = True
    run.font.color.rgb = C_DARK
    return p

def add_body(text, bold_prefix="", italic=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_b = p.add_run(bold_prefix)
        r_b.font.name = "Calibri"
        r_b.font.size = Pt(11)
        r_b.font.bold = True
        r_b.font.color.rgb = C_DARK
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(11)
    run.font.italic = italic
    run.font.color.rgb = C_DARK
    return p

def add_spoken_quote(text, visual_text=""):
    if visual_text:
        pv = doc.add_paragraph()
        pv.paragraph_format.space_before = Pt(4)
        pv.paragraph_format.space_after = Pt(2)
        rv = pv.add_run(f"Visual Cue: {visual_text}")
        rv.font.name = "Calibri"
        rv.font.size = Pt(10)
        rv.font.italic = True
        rv.font.color.rgb = C_GRAY

    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    
    # Border & shading
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(r'<w:shd {} w:fill="F4F8FA"/>'.format(nsdecls('w')))
    tcPr.append(shd)
    
    # Left border
    borders = parse_xml(
        r'<w:tcBorders {}><w:top w:val="none"/><w:left w:val="single" w:sz="24" w:space="0" w:color="007BC0"/><w:bottom w:val="none"/><w:right w:val="none"/></w:tcBorders>'.format(nsdecls('w'))
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.left_indent = Inches(0.15)
    p.paragraph_format.right_indent = Inches(0.15)
    p.paragraph_format.line_spacing = 1.2
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(10.5)
    run.font.color.rgb = C_DARK

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

# Title Block
p_title = doc.add_paragraph()
p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_title.paragraph_format.space_before = Pt(0)
p_title.paragraph_format.space_after = Pt(2)
r_t = p_title.add_run("INDIAN INSTITUTE OF TECHNOLOGY GUWAHATI")
r_t.font.name = "Calibri"
r_t.font.size = Pt(15)
r_t.font.bold = True
r_t.font.color.rgb = C_NAVY

p_sub = doc.add_paragraph()
p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_sub.paragraph_format.space_after = Pt(12)
r_s = p_sub.add_run("DEPARTMENT OF CHEMISTRY | CH 499 B.TECH PROJECT DEFENSE")
r_s.font.name = "Calibri"
r_s.font.size = Pt(11)
r_s.font.bold = True
r_s.font.color.rgb = C_BLUE

p_main = doc.add_paragraph()
p_main.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_main.paragraph_format.space_after = Pt(6)
r_m = p_main.add_run("Spoken Presentation Script & Professor Q&A Defense Dossier")
r_m.font.name = "Calibri"
r_m.font.size = Pt(20)
r_m.font.bold = True
r_m.font.color.rgb = C_NAVY

p_info = doc.add_paragraph()
p_info.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_info.paragraph_format.space_after = Pt(16)
r_i = p_info.add_run("Candidate: Sarvagya Tiwari (Roll No. 230122047)\nSupervisor: Prof. Ritika Jain | Target Presentation Duration: 7–8 Minutes\nSlide Deck: IITG_BTP_Presentation_Sarvagya_Tiwari.pptx (15 Slides)")
r_i.font.name = "Calibri"
r_i.font.size = Pt(11)
r_i.font.color.rgb = C_GRAY

doc.add_paragraph().paragraph_format.space_after = Pt(8)

# Read the Markdown file and parse structured parts
with open(r"d:\BTP\BTP_Presentation_Script_and_QA.md", "r", encoding="utf-8") as f:
    lines = f.readlines()

in_spoken = False
current_slide_title = ""
current_slide_time = ""
current_visual = ""
current_spoken_text = []

in_qa = False
current_q = ""
current_a = []

for line in lines:
    line_s = line.strip()
    
    if line_s.startswith("# PART 1:"):
        style_heading_1("PART 1: 7–8 MINUTE VERBAL PRESENTATION SCRIPT")
        continue
    elif line_s.startswith("# PART 2:"):
        # Flush any pending spoken slide
        if current_slide_title and current_spoken_text:
            style_heading_2(f"{current_slide_title}  [{current_slide_time}]")
            add_spoken_quote(" ".join(current_spoken_text), current_visual)
            current_slide_title = ""
            current_spoken_text = []
        style_heading_1("PART 2: COMPREHENSIVE PROFESSOR DEFENSE Q&A DOSSIER")
        in_qa = True
        continue
    elif line_s.startswith("# SUMMARY CHEAT SHEET"):
        # Flush any pending Q
        if current_q and current_a:
            style_heading_2(current_q)
            for par in current_a:
                add_body(par)
            current_q = ""
            current_a = []
        style_heading_1("DEFENSE QUICK REFERENCE CHEAT SHEET")
        continue

    if not in_qa:
        if line_s.startswith("### Slide "):
            # Flush previous slide
            if current_slide_title and current_spoken_text:
                style_heading_2(f"{current_slide_title}  [{current_slide_time}]")
                add_spoken_quote(" ".join(current_spoken_text), current_visual)
                current_slide_title = ""
                current_spoken_text = []
                current_visual = ""

            parts = line_s.replace("### ", "").split(" [Time: ")
            current_slide_title = parts[0]
            current_slide_time = parts[1].replace("]", "") if len(parts) > 1 else ""
        elif line_s.startswith("**Visual:**"):
            current_visual = line_s.replace("**Visual:**", "").strip("* ")
        elif line_s.startswith("**Spoken Script:**"):
            pass
        elif line_s.startswith(">"):
            text = line_s.lstrip("> ").strip()
            if text:
                current_spoken_text.append(text)
    else:
        if line_s.startswith("### Question "):
            if current_q and current_a:
                style_heading_2(current_q)
                for par in current_a:
                    add_body(par)
                current_a = []
            current_q = line_s.replace("### ", "").strip('"')
        elif line_s.startswith("**Answer:**"):
            pass
        elif line_s.startswith(">"):
            text = line_s.lstrip("> ").strip()
            if text:
                current_a.append(text)
        elif line_s.startswith("|"):
            # Table in cheat sheet
            pass

# Flush final
if current_q and current_a:
    style_heading_2(current_q)
    for par in current_a:
        add_body(par)

# Add Summary Table at end
style_heading_2("Multi-Cohort Benchmark Matrix")
t = doc.add_table(rows=10, cols=4)
t.alignment = WD_TABLE_ALIGNMENT.CENTER
t.autofit = False

headers = ["Metric / Dimension", "CDC NHANES (N=3,292)", "Google Health (N=1,165)", "MESA Sleep (N=2,237)"]
rows = [
    ["Study Population", "General US Adults (18–65y)", "Consumer Adults (20–75y)", "Multi-Ethnic Elderly (54–93y)"],
    ["Sensors", "ActiGraph GT3X+ Accelerometer", "Optical PPG (HR/HRV) + Accel", "Actiwatch Spectrum Actigraphy"],
    ["Zero-Blood Tier 1 Log R²", "0.4044 (Pearson r = 0.6364)", "0.3335 (Pearson r = 0.5790)", "0.2927 (Pearson r = 0.5410)"],
    ["Zero-Blood Tier 1 AUROC", "0.7766", "0.7812", "0.7502"],
    ["Sensitivity in Obese (BMI ≥ 30)", "85.2%", "89.0%", "92.0%"],
    ["Routine Blood Labs Alone R²", "N/A", "0.5193", "0.4183"],
    ["Full Multimodal Log R² (Tier 2)", "0.5120 (Pearson r = 0.7155)", "0.5882 (Pearson r = 0.7670)", "0.4474 (Pearson r = 0.6689)"],
    ["Full Multimodal AUROC (Tier 2)", "0.8650", "0.8845", "0.8410"],
    ["Zero-Shot Transfer Retention", "Training Source", "—", "97.5% R² / 100.3% AUROC"],
]

for col_idx, col in enumerate(t.columns):
    col.width = Inches(2.2) if col_idx == 0 else Inches(1.4)

for j, h in enumerate(headers):
    cell = t.cell(0, j)
    cell.text = h
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(r'<w:shd {} w:fill="003B6F"/>'.format(nsdecls('w')))
    tcPr.append(shd)
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.runs[0]
    run.font.name = "Calibri"
    run.font.size = Pt(9.5)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

for i, row in enumerate(rows):
    for j, val in enumerate(row):
        cell = t.cell(i + 1, j)
        cell.text = val
        tcPr = cell._element.get_or_add_tcPr()
        fill_color = "F0F4F8" if i % 2 == 0 else "FFFFFF"
        shd = parse_xml(r'<w:shd {} w:fill="{}"/>'.format(nsdecls('w'), fill_color))
        tcPr.append(shd)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
        run = p.runs[0]
        run.font.name = "Calibri"
        run.font.size = Pt(9)
        if j == 0:
            run.font.bold = True
            run.font.color.rgb = C_NAVY
        else:
            run.font.color.rgb = C_DARK

doc.save(DOC_PATH)
print(f"✅ Successfully created document at: {DOC_PATH}")
