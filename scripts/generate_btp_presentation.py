#!/usr/bin/env python3
"""
Generate BTP Presentation: Wearable AI for Insulin Resistance Screening
Student: Sarvagya Tiwari (230122047), IIT Guwahati
Supervisor: Prof. Ritika Jain, Department of Chemistry
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

# ─── Configuration ───
OUTPUT_PATH = r"d:\BTP\IITG_BTP_Presentation_Sarvagya_Tiwari.pptx"
FIGURES_DIR = r"d:\BTP\figures"

# IIT Guwahati brand colors
IITG_BLUE = RGBColor(0x00, 0x3B, 0x6F)       # Dark navy blue
IITG_ACCENT = RGBColor(0x00, 0x7B, 0xC0)      # Medium blue
IITG_LIGHT_BLUE = RGBColor(0xD0, 0xE4, 0xF0)  # Light background
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x00, 0x00, 0x00)
DARK_GRAY = RGBColor(0x33, 0x33, 0x33)
MED_GRAY = RGBColor(0x66, 0x66, 0x66)
LIGHT_GRAY = RGBColor(0xF0, 0xF0, 0xF0)
GREEN = RGBColor(0x00, 0x80, 0x40)
RED_ACCENT = RGBColor(0xC0, 0x00, 0x00)
GOLD = RGBColor(0xCC, 0x99, 0x00)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def add_background(slide, color=WHITE):
    """Set solid background color for a slide."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_shape(slide, left, top, width, height, fill_color, border_color=None):
    """Add a rectangle shape."""
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape


def add_rounded_rect(slide, left, top, width, height, fill_color, border_color=None):
    """Add a rounded rectangle shape."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.5)
    else:
        shape.line.fill.background()
    return shape


def set_text(shape, text, font_size=18, color=BLACK, bold=False, alignment=PP_ALIGN.LEFT, font_name="Calibri"):
    """Set text in a shape's text frame."""
    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return tf


def add_text_para(tf, text, font_size=16, color=BLACK, bold=False, alignment=PP_ALIGN.LEFT,
                  space_before=Pt(4), space_after=Pt(2), font_name="Calibri", italic=False):
    """Add a new paragraph to a text frame."""
    p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.italic = italic
    p.font.name = font_name
    p.alignment = alignment
    if space_before:
        p.space_before = space_before
    if space_after:
        p.space_after = space_after
    return p


def add_header_bar(slide, title_text):
    """Add a professional header bar at the top."""
    bar = add_shape(slide, Inches(0), Inches(0), SLIDE_W, Inches(1.0), IITG_BLUE)
    tf = set_text(bar, title_text, font_size=28, color=WHITE, bold=True, alignment=PP_ALIGN.LEFT, font_name="Calibri")
    bar.text_frame.margin_left = Inches(0.6)
    bar.text_frame.margin_top = Inches(0.15)
    # Accent line
    add_shape(slide, Inches(0), Inches(1.0), SLIDE_W, Inches(0.04), IITG_ACCENT)
    return bar


def add_slide_number(slide, num, total=14):
    """Add slide number in bottom right."""
    txBox = slide.shapes.add_textbox(Inches(12.0), Inches(7.05), Inches(1.2), Inches(0.4))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = f"{num}/{total}"
    p.font.size = Pt(11)
    p.font.color.rgb = MED_GRAY
    p.font.name = "Calibri"
    p.alignment = PP_ALIGN.RIGHT


def add_footer(slide, text="Department of Chemistry | IIT Guwahati"):
    """Add footer text."""
    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(6), Inches(0.4))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(10)
    p.font.color.rgb = MED_GRAY
    p.font.name = "Calibri"
    p.alignment = PP_ALIGN.LEFT


def try_add_image(slide, filename, left, top, width=None, height=None):
    """Try to add an image, skip if not found."""
    path = os.path.join(FIGURES_DIR, filename)
    if os.path.exists(path):
        kwargs = {"image_file": path, "left": left, "top": top}
        if width:
            kwargs["width"] = width
        if height:
            kwargs["height"] = height
        slide.shapes.add_picture(**kwargs)
        return True
    else:
        print(f"  [SKIP] Image not found: {filename}")
        return False


# ═══════════════════════════════════════════════════════════════
# SLIDE 1: TITLE SLIDE
# ═══════════════════════════════════════════════════════════════
print("Creating Slide 1: Title...")
slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank layout
add_background(slide, WHITE)

# Top blue bar
add_shape(slide, Inches(0), Inches(0), SLIDE_W, Inches(2.2), IITG_BLUE)
add_shape(slide, Inches(0), Inches(2.2), SLIDE_W, Inches(0.06), GOLD)

# IIT Guwahati logo
logo_path = os.path.join(FIGURES_DIR, "iitg_logo.png")
if os.path.exists(logo_path):
    slide.shapes.add_picture(logo_path, Inches(0.5), Inches(0.3), height=Inches(1.5))

# Institute name
txBox = slide.shapes.add_textbox(Inches(2.2), Inches(0.3), Inches(10), Inches(0.7))
tf = txBox.text_frame
p = tf.paragraphs[0]
p.text = "INDIAN INSTITUTE OF TECHNOLOGY GUWAHATI"
p.font.size = Pt(26)
p.font.color.rgb = WHITE
p.font.bold = True
p.font.name = "Calibri"
p.alignment = PP_ALIGN.LEFT

add_text_para(tf, "Department of Chemistry", font_size=18, color=RGBColor(0xBB, 0xDD, 0xFF),
              bold=False, alignment=PP_ALIGN.LEFT, space_before=Pt(4))

# Course label
txBox2 = slide.shapes.add_textbox(Inches(9), Inches(1.5), Inches(4), Inches(0.5))
tf2 = txBox2.text_frame
p2 = tf2.paragraphs[0]
p2.text = "CH 499: B.Tech Project - I"
p2.font.size = Pt(16)
p2.font.color.rgb = RGBColor(0xBB, 0xDD, 0xFF)
p2.font.bold = True
p2.font.name = "Calibri"
p2.alignment = PP_ALIGN.RIGHT

# Title
txBox3 = slide.shapes.add_textbox(Inches(1.0), Inches(2.8), Inches(11.3), Inches(2.0))
tf3 = txBox3.text_frame
tf3.word_wrap = True
p3 = tf3.paragraphs[0]
p3.text = "Wearable AI for Insulin Resistance Screening"
p3.font.size = Pt(38)
p3.font.color.rgb = IITG_BLUE
p3.font.bold = True
p3.font.name = "Calibri"
p3.alignment = PP_ALIGN.CENTER

add_text_para(tf3, "Non-Invasive Metabolic Health Monitoring Using Consumer Wearables & Machine Learning",
              font_size=20, color=IITG_ACCENT, bold=False, alignment=PP_ALIGN.CENTER, space_before=Pt(12))

# Student and Supervisor info
info_box = add_rounded_rect(slide, Inches(2.5), Inches(5.2), Inches(8.3), Inches(1.7),
                            IITG_LIGHT_BLUE, border_color=IITG_ACCENT)
tf4 = info_box.text_frame
tf4.word_wrap = True
tf4.margin_left = Inches(0.3)
tf4.margin_top = Inches(0.15)

p4 = tf4.paragraphs[0]
p4.text = "Presented by:  Sarvagya Tiwari  |  Roll No. 230122047"
p4.font.size = Pt(18)
p4.font.color.rgb = IITG_BLUE
p4.font.bold = True
p4.font.name = "Calibri"
p4.alignment = PP_ALIGN.CENTER

add_text_para(tf4, "Supervisor:  Prof. Ritika Jain  |  Department of Chemistry, IIT Guwahati",
              font_size=16, color=DARK_GRAY, bold=False, alignment=PP_ALIGN.CENTER, space_before=Pt(8))
add_text_para(tf4, "November 2026",
              font_size=14, color=MED_GRAY, bold=False, alignment=PP_ALIGN.CENTER, space_before=Pt(6))


# ═══════════════════════════════════════════════════════════════
# SLIDE 2: OUTLINE / AGENDA
# ═══════════════════════════════════════════════════════════════
print("Creating Slide 2: Outline...")
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_header_bar(slide, "Presentation Outline")
add_slide_number(slide, 2)
add_footer(slide)

items = [
    ("01", "Background & Problem Statement", "Clinical burden of insulin resistance and limitations of invasive screening"),
    ("02", "Research Objectives", "Six core goals guiding the experimental framework"),
    ("03", "Datasets & Cohort Architecture", "Three-cohort design: NHANES, Google Health, MESA (N = 6,698)"),
    ("04", "Methodology", "Feature engineering, ML/DL architectures, and validation protocols"),
    ("05", "Results: Regression & Classification", "Continuous HOMA-IR prediction and IR risk stratification"),
    ("06", "Results: Interpretability & Transfer", "SHAP feature attribution and zero-shot cross-cohort generalization"),
    ("07", "Production Architecture & Conclusion", "Smartwatch deployment blueprint and key takeaways"),
]

y_start = Inches(1.5)
for i, (num, title, desc) in enumerate(items):
    y = y_start + Inches(i * 0.78)
    # Number circle
    circle = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.8), y, Inches(0.5), Inches(0.5))
    circle.fill.solid()
    circle.fill.fore_color.rgb = IITG_BLUE
    circle.line.fill.background()
    set_text(circle, num, font_size=14, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    circle.text_frame.margin_left = Inches(0)
    circle.text_frame.margin_top = Inches(0.05)

    # Title
    txBox = slide.shapes.add_textbox(Inches(1.6), y - Inches(0.02), Inches(5), Inches(0.35))
    set_text(txBox, title, font_size=18, color=IITG_BLUE, bold=True)

    # Description
    txBox2 = slide.shapes.add_textbox(Inches(1.6), y + Inches(0.3), Inches(10), Inches(0.35))
    set_text(txBox2, desc, font_size=14, color=MED_GRAY)


# ═══════════════════════════════════════════════════════════════
# SLIDE 3: BACKGROUND & PROBLEM STATEMENT
# ═══════════════════════════════════════════════════════════════
print("Creating Slide 3: Background...")
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_header_bar(slide, "Background & Problem Statement")
add_slide_number(slide, 3)
add_footer(slide)

# Left panel - The Problem
left_box = add_rounded_rect(slide, Inches(0.5), Inches(1.3), Inches(6.0), Inches(5.5),
                            LIGHT_GRAY, border_color=RGBColor(0xDD, 0xDD, 0xDD))
tf = left_box.text_frame
tf.word_wrap = True
tf.margin_left = Inches(0.2)
tf.margin_top = Inches(0.15)

p = tf.paragraphs[0]
p.text = "The Clinical Challenge"
p.font.size = Pt(22)
p.font.color.rgb = RED_ACCENT
p.font.bold = True
p.font.name = "Calibri"

bullets = [
    "Insulin resistance (IR) is the root driver of Type 2 Diabetes, cardiovascular disease, and metabolic syndrome",
    "Over 40% of adults globally have impaired insulin sensitivity",
    "More than 80% remain undiagnosed until irreversible organ damage occurs",
    "Gold standard HOMA-IR requires fasting venous blood draws — invasive, episodic, costly",
    "Single-time-point blood tests cannot capture continuous metabolic dynamics",
    "No existing population-level screening tool for early-stage IR detection"
]
for b in bullets:
    add_text_para(tf, f"▸  {b}", font_size=14, color=DARK_GRAY, space_before=Pt(8), space_after=Pt(2))

# Right panel - Our Approach
right_box = add_rounded_rect(slide, Inches(6.8), Inches(1.3), Inches(6.0), Inches(5.5),
                             RGBColor(0xE8, 0xF4, 0xE8), border_color=GREEN)
tf2 = right_box.text_frame
tf2.word_wrap = True
tf2.margin_left = Inches(0.2)
tf2.margin_top = Inches(0.15)

p2 = tf2.paragraphs[0]
p2.text = "Our Approach: Wearable AI"
p2.font.size = Pt(22)
p2.font.color.rgb = GREEN
p2.font.bold = True

bullets2 = [
    "Passive consumer wearables (smartwatch PPG + accelerometry) as non-invasive metabolic sensors",
    "Machine learning models predict continuous HOMA-IR from wearable signals + basic anthropometrics",
    "Two-tier architecture: daily zero-blood screening + annual lab-enhanced calibration",
    "Multi-cohort validation across 3 independent datasets (N = 6,698 participants)",
    "Explainable AI (SHAP) for clinically interpretable feature attribution",
    "Zero-shot cross-cohort generalization without fine-tuning"
]
for b in bullets2:
    add_text_para(tf2, f"▸  {b}", font_size=14, color=DARK_GRAY, space_before=Pt(8), space_after=Pt(2))


# ═══════════════════════════════════════════════════════════════
# SLIDE 4: RESEARCH OBJECTIVES
# ═══════════════════════════════════════════════════════════════
print("Creating Slide 4: Objectives...")
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_header_bar(slide, "Research Objectives")
add_slide_number(slide, 4)
add_footer(slide)

objectives = [
    ("Continuous HOMA-IR Regression", "Predict continuous insulin sensitivity scores from non-invasive wearable signals without blood draws"),
    ("IR Risk Classification", "Classify individuals into Normal, Pre-Resistant, and Insulin-Resistant risk categories (binary and 3-class)"),
    ("Multi-Cohort Validation", "Validate across three independent clinical cohorts spanning diverse demographics and sensor modalities"),
    ("Feature Interpretability", "Apply 5-fold stratified TreeSHAP to identify and rank the most predictive non-invasive biomarkers"),
    ("Cross-Cohort Generalization", "Test zero-shot transfer of trained models to entirely unseen populations without fine-tuning"),
    ("Production Architecture", "Design an end-to-end smartwatch deployment blueprint with on-device and cloud-based inference tiers"),
]

for i, (title, desc) in enumerate(objectives):
    row = i // 2
    col = i % 2
    x = Inches(0.5 + col * 6.4)
    y = Inches(1.4 + row * 1.9)

    # Objective box
    box = add_rounded_rect(slide, x, y, Inches(6.0), Inches(1.6),
                           IITG_LIGHT_BLUE if col == 0 else RGBColor(0xF0, 0xF5, 0xFA),
                           border_color=IITG_ACCENT)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.2)
    tf.margin_top = Inches(0.1)

    # Number badge
    num_circle = slide.shapes.add_shape(MSO_SHAPE.OVAL, x + Inches(0.1), y + Inches(0.1),
                                        Inches(0.4), Inches(0.4))
    num_circle.fill.solid()
    num_circle.fill.fore_color.rgb = IITG_BLUE
    num_circle.line.fill.background()
    set_text(num_circle, str(i + 1), font_size=14, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    num_circle.text_frame.margin_left = Inches(0)
    num_circle.text_frame.margin_top = Inches(0.04)

    p = tf.paragraphs[0]
    p.text = f"     {title}"
    p.font.size = Pt(17)
    p.font.color.rgb = IITG_BLUE
    p.font.bold = True
    p.font.name = "Calibri"

    add_text_para(tf, f"     {desc}", font_size=13, color=DARK_GRAY, space_before=Pt(6))


# ═══════════════════════════════════════════════════════════════
# SLIDE 5: DATASETS & COHORT ARCHITECTURE
# ═══════════════════════════════════════════════════════════════
print("Creating Slide 5: Datasets...")
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_header_bar(slide, "Multi-Cohort Dataset Architecture  (N = 6,698)")
add_slide_number(slide, 5)
add_footer(slide)

cohorts = [
    ("CDC NHANES 2011-2014", "N = 3,292", "18 - 65 yrs", "ActiGraph GT3X+",
     "Triaxial Actigraphy", "National epidemiological\ncohort, 7-day continuous\nwear, full HOMA-IR labs",
     IITG_BLUE),
    ("Google Health / Nature 2026", "N = 1,165", "20 - 75 yrs", "Consumer Smartwatch",
     "Optical PPG + Accel", "Multi-sensor wearable trial,\nHR/HRV, comprehensive\nmetabolic phenotyping",
     IITG_ACCENT),
    ("MESA Sleep Ancillary", "N = 2,237", "54 - 93 yrs", "Actiwatch Spectrum",
     "Wrist Actigraphy", "Multi-ethnic elderly cohort,\nexternal validation,\nzero-shot transfer target",
     RGBColor(0x00, 0x60, 0x60)),
]

for i, (name, n, age, device, sensor, desc, color) in enumerate(cohorts):
    x = Inches(0.5 + i * 4.2)
    # Header box
    hbox = add_rounded_rect(slide, x, Inches(1.3), Inches(3.9), Inches(0.9), color)
    tf = hbox.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.15)
    tf.margin_top = Inches(0.08)
    p = tf.paragraphs[0]
    p.text = name
    p.font.size = Pt(16)
    p.font.color.rgb = WHITE
    p.font.bold = True
    p.font.name = "Calibri"
    p.alignment = PP_ALIGN.CENTER
    add_text_para(tf, n, font_size=20, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER, space_before=Pt(2))

    # Details box
    dbox = add_rounded_rect(slide, x, Inches(2.3), Inches(3.9), Inches(4.3),
                            LIGHT_GRAY, border_color=color)
    tf2 = dbox.text_frame
    tf2.word_wrap = True
    tf2.margin_left = Inches(0.15)
    tf2.margin_top = Inches(0.15)

    details = [
        ("Age Range:", age),
        ("Device:", device),
        ("Sensors:", sensor),
        ("Role:", desc),
    ]
    first = True
    for label, val in details:
        if first:
            p = tf2.paragraphs[0]
            p.text = label
            p.font.size = Pt(13)
            p.font.color.rgb = color
            p.font.bold = True
            p.font.name = "Calibri"
            first = False
        else:
            add_text_para(tf2, label, font_size=13, color=color, bold=True, space_before=Pt(10))
        add_text_para(tf2, val, font_size=13, color=DARK_GRAY, space_before=Pt(2))

# Total box
total_box = add_rounded_rect(slide, Inches(4.0), Inches(6.7), Inches(5.3), Inches(0.5),
                             IITG_BLUE)
set_text(total_box, "Total Validated Population:  N = 6,698 Multi-Ethnic Participants",
         font_size=16, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
total_box.text_frame.margin_top = Inches(0.05)


# ═══════════════════════════════════════════════════════════════
# SLIDE 6: METHODOLOGY - FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
print("Creating Slide 6: Methodology - Features...")
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_header_bar(slide, "Methodology: Feature Engineering & Signal Processing")
add_slide_number(slide, 6)
add_footer(slide)

# Feature categories
categories = [
    ("Anthropometric Priors", IITG_BLUE, [
        "BMI = Weight / Height^2",
        "Waist-to-Height Ratio (WHtR)",
        "Age, Sex (binary encoding)",
    ]),
    ("Autonomic PPG Metrics", RGBColor(0x00, 0x80, 0x40), [
        "Resting Heart Rate (nadir)",
        "Heart Rate Variability (RMSSD)",
        "Nocturnal HR dip percentage",
        "Autonomic Stress Ratio (RHR/HRV)",
    ]),
    ("Actigraphy & Activity", RGBColor(0xCC, 0x66, 0x00), [
        "Daily step count / MVPA minutes",
        "Sedentary bout duration & fragmentation",
        "Physical activity volume (ENMO)",
        "Active-to-sedentary ratio",
    ]),
    ("Circadian & Sleep", RGBColor(0x66, 0x00, 0x99), [
        "Sleep duration & efficiency",
        "WASO (wake after sleep onset)",
        "Interdaily Stability (IS)",
        "Intradaily Variability (IV)",
        "Cosinor amplitude & acrophase",
    ]),
]

for i, (cat_name, color, features) in enumerate(categories):
    x = Inches(0.4 + i * 3.2)
    # Category header
    hdr = add_rounded_rect(slide, x, Inches(1.3), Inches(3.0), Inches(0.6), color)
    set_text(hdr, cat_name, font_size=15, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    hdr.text_frame.margin_top = Inches(0.08)

    # Feature list
    fbox = add_rounded_rect(slide, x, Inches(2.0), Inches(3.0), Inches(3.0),
                            LIGHT_GRAY, border_color=color)
    tf = fbox.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.15)
    tf.margin_top = Inches(0.1)
    for j, feat in enumerate(features):
        if j == 0:
            p = tf.paragraphs[0]
            p.text = f"▸ {feat}"
            p.font.size = Pt(13)
            p.font.color.rgb = DARK_GRAY
            p.font.name = "Calibri"
        else:
            add_text_para(tf, f"▸ {feat}", font_size=13, color=DARK_GRAY, space_before=Pt(8))

# Log-transform formula box
formula_box = add_rounded_rect(slide, Inches(1.0), Inches(5.3), Inches(11.3), Inches(1.5),
                                RGBColor(0xFD, 0xF0, 0xE0), border_color=GOLD)
tf = formula_box.text_frame
tf.word_wrap = True
tf.margin_left = Inches(0.3)
tf.margin_top = Inches(0.1)
p = tf.paragraphs[0]
p.text = "Key Mathematical Insight: Log-Scale Homoscedastic Target Transformation"
p.font.size = Pt(16)
p.font.color.rgb = IITG_BLUE
p.font.bold = True
p.font.name = "Calibri"
add_text_para(tf, "Target = ln(HOMA-IR) = ln( Insulin x Glucose / 405 )", font_size=16,
              color=DARK_GRAY, bold=True, space_before=Pt(8))
add_text_para(tf, "Eliminates severe skewness (kurtosis > 15), restores Gaussian residuals, and boosts R^2 by +0.08 to +0.14 across all cohorts",
              font_size=13, color=MED_GRAY, space_before=Pt(4))


# ═══════════════════════════════════════════════════════════════
# SLIDE 7: METHODOLOGY - ML ARCHITECTURE
# ═══════════════════════════════════════════════════════════════
print("Creating Slide 7: ML Architecture...")
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_header_bar(slide, "Methodology: Machine Learning Architecture")
add_slide_number(slide, 7)
add_footer(slide)

# Tier 1 box
t1_box = add_rounded_rect(slide, Inches(0.5), Inches(1.3), Inches(6.0), Inches(5.5),
                           RGBColor(0xE8, 0xF4, 0xE8), border_color=GREEN)
tf1 = t1_box.text_frame
tf1.word_wrap = True
tf1.margin_left = Inches(0.2)
tf1.margin_top = Inches(0.15)

# Tier 1 header
p = tf1.paragraphs[0]
p.text = "TIER 1: On-Device Zero-Blood Screening"
p.font.size = Pt(20)
p.font.color.rgb = GREEN
p.font.bold = True
p.font.name = "Calibri"

t1_details = [
    "▸ Inputs: Wearable signals + basic anthropometrics (no blood)",
    "▸ Blended Ensemble: LightGBM (80%) + Ridge Regression (20%)",
    "▸ Out-of-Fold cross-validated stacking (5-fold stratified)",
    "▸ Quantized TFLite INT8 model: < 3.5 MB on-device",
    "▸ Battery impact: < 1.5% per day",
    "",
    "Performance (Zero-Blood Mode):",
    "  • Log R^2: 0.29 - 0.40  |  Pearson r: 0.54 - 0.64",
    "  • Binary AUROC: 0.75 - 0.78",
    "  • High-Risk Sensitivity: 85% - 92% (BMI >= 30)",
    "",
    "Always-on passive screening: no fingersticks, no clinic visits",
]
for d in t1_details:
    is_bold = d.startswith("Performance") or d.startswith("Always-on")
    color = GREEN if is_bold else DARK_GRAY
    add_text_para(tf1, d, font_size=13, color=color, bold=is_bold, space_before=Pt(5))

# Tier 2 box
t2_box = add_rounded_rect(slide, Inches(6.8), Inches(1.3), Inches(6.0), Inches(5.5),
                           RGBColor(0xE8, 0xEC, 0xF8), border_color=IITG_ACCENT)
tf2 = t2_box.text_frame
tf2.word_wrap = True
tf2.margin_left = Inches(0.2)
tf2.margin_top = Inches(0.15)

p = tf2.paragraphs[0]
p.text = "TIER 2: Cloud Multimodal Fusion"
p.font.size = Pt(20)
p.font.color.rgb = IITG_ACCENT
p.font.bold = True

t2_details = [
    "▸ Inputs: Wearable signals + anthropometrics + routine lab panel",
    "▸ Deep Multimodal Residual Latent Space Encoder (z in R^32)",
    "▸ Multi-head attention for cross-modality feature fusion",
    "▸ Isotonic regression for probability calibration",
    "▸ Annual sync with Health Connect / EHR systems",
    "",
    "Performance (Full Multimodal):",
    "  • Log R^2: 0.45 - 0.59  |  Pearson r: 0.67 - 0.77",
    "  • Binary AUROC: 0.83 - 0.88",
    "  • 3-Class F1 (Macro): 0.72 - 0.78",
    "",
    "Triggered by Tier 1 risk escalation (MHI Score >= 65)",
]
for d in t2_details:
    is_bold = d.startswith("Performance") or d.startswith("Triggered")
    color = IITG_ACCENT if is_bold else DARK_GRAY
    add_text_para(tf2, d, font_size=13, color=color, bold=is_bold, space_before=Pt(5))


# ═══════════════════════════════════════════════════════════════
# SLIDE 8: RESULTS - CONTINUOUS REGRESSION BENCHMARK
# ═══════════════════════════════════════════════════════════════
print("Creating Slide 8: Regression Results...")
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_header_bar(slide, "Results: Continuous HOMA-IR Regression Benchmark")
add_slide_number(slide, 8)
add_footer(slide)

# Table
from pptx.util import Emu
table_shape = slide.shapes.add_table(6, 4, Inches(0.5), Inches(1.3), Inches(7.5), Inches(3.5))
table = table_shape.table

# Set column widths
table.columns[0].width = Inches(2.5)
table.columns[1].width = Inches(1.7)
table.columns[2].width = Inches(1.7)
table.columns[3].width = Inches(1.7)

headers = ["Modality Tier", "CDC NHANES", "Google Health", "MESA Sleep"]
rows_data = [
    ["Wearables Only (Log R^2)", "0.1245", "0.1383", "0.0851"],
    ["Zero-Blood Tier 1 (Log R^2)", "0.4044", "0.3335", "0.2927"],
    ["Zero-Blood Tier 1 (Pearson r)", "0.6364", "0.5790", "0.5410"],
    ["Routine Blood Labs (Log R^2)", "N/A", "0.5193", "0.4183"],
    ["Full Multimodal (Log R^2)", "0.5120", "0.5882", "0.4474"],
]

# Style header row
for j, h in enumerate(headers):
    cell = table.cell(0, j)
    cell.text = h
    p = cell.text_frame.paragraphs[0]
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.font.name = "Calibri"
    p.alignment = PP_ALIGN.CENTER
    cell.fill.solid()
    cell.fill.fore_color.rgb = IITG_BLUE

# Fill data
for i, row in enumerate(rows_data):
    for j, val in enumerate(row):
        cell = table.cell(i + 1, j)
        cell.text = val
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(12)
        p.font.name = "Calibri"
        p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT
        if j == 0:
            p.font.bold = True
            p.font.color.rgb = IITG_BLUE
        else:
            p.font.color.rgb = DARK_GRAY
        # Alternate row colors
        cell.fill.solid()
        cell.fill.fore_color.rgb = LIGHT_GRAY if i % 2 == 0 else WHITE

# Add figure
try_add_image(slide, "mesa_phase1_scatter_hexbin.png",
              Inches(8.3), Inches(1.3), width=Inches(4.5))

# Key insight box
insight_box = add_rounded_rect(slide, Inches(0.5), Inches(5.1), Inches(12.3), Inches(1.5),
                                RGBColor(0xFD, 0xF0, 0xE0), border_color=GOLD)
tf = insight_box.text_frame
tf.word_wrap = True
tf.margin_left = Inches(0.2)
tf.margin_top = Inches(0.1)
p = tf.paragraphs[0]
p.text = "Key Finding"
p.font.size = Pt(16)
p.font.color.rgb = IITG_BLUE
p.font.bold = True
add_text_para(tf, "Zero-blood wearable + anthropometric screening explains 29% - 40% of variance in continuous insulin sensitivity, achieving Pearson correlations of r = 0.54 to 0.64 — comparable to clinical lipid panel accuracy but without any blood draw.",
              font_size=13, color=DARK_GRAY, space_before=Pt(6))


# ═══════════════════════════════════════════════════════════════
# SLIDE 9: RESULTS - CLASSIFICATION PERFORMANCE
# ═══════════════════════════════════════════════════════════════
print("Creating Slide 9: Classification Results...")
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_header_bar(slide, "Results: Insulin Resistance Risk Classification")
add_slide_number(slide, 9)
add_footer(slide)

# Classification table
table_shape2 = slide.shapes.add_table(6, 4, Inches(0.5), Inches(1.3), Inches(7.5), Inches(3.3))
table2 = table_shape2.table
table2.columns[0].width = Inches(2.5)
table2.columns[1].width = Inches(1.7)
table2.columns[2].width = Inches(1.7)
table2.columns[3].width = Inches(1.7)

c_headers = ["Classification Metric", "CDC NHANES", "Google Health", "MESA Sleep"]
c_rows = [
    ["Wearables Only AUROC", "0.6620", "0.6651", "0.6411"],
    ["Zero-Blood Tier 1 AUROC", "0.7766", "0.7812", "0.7502"],
    ["Full Multimodal AUROC", "0.8650", "0.8845", "0.8410"],
    ["High-Risk Sensitivity (BMI>=30)", "85.2%", "89.0%", "92.0%"],
    ["Diagnostic Specificity (Tier 1)", "72.4%", "75.1%", "88.3%"],
]

for j, h in enumerate(c_headers):
    cell = table2.cell(0, j)
    cell.text = h
    p = cell.text_frame.paragraphs[0]
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.font.name = "Calibri"
    p.alignment = PP_ALIGN.CENTER
    cell.fill.solid()
    cell.fill.fore_color.rgb = IITG_BLUE

for i, row in enumerate(c_rows):
    for j, val in enumerate(row):
        cell = table2.cell(i + 1, j)
        cell.text = val
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(12)
        p.font.name = "Calibri"
        p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT
        if j == 0:
            p.font.bold = True
            p.font.color.rgb = IITG_BLUE
        else:
            p.font.color.rgb = DARK_GRAY
        cell.fill.solid()
        cell.fill.fore_color.rgb = LIGHT_GRAY if i % 2 == 0 else WHITE

# Add ROC curve figure
try_add_image(slide, "mesa_phase2_roc_pr_curves.png",
              Inches(8.3), Inches(1.3), width=Inches(4.5))

# Insight
insight_box2 = add_rounded_rect(slide, Inches(0.5), Inches(4.9), Inches(12.3), Inches(1.8),
                                 RGBColor(0xE8, 0xF4, 0xE8), border_color=GREEN)
tf = insight_box2.text_frame
tf.word_wrap = True
tf.margin_left = Inches(0.2)
tf.margin_top = Inches(0.1)
p = tf.paragraphs[0]
p.text = "Clinical Significance"
p.font.size = Pt(16)
p.font.color.rgb = GREEN
p.font.bold = True
add_text_para(tf, "In high-risk obese populations (BMI >= 30), our zero-blood Tier 1 model detects 85% - 92% of insulin-resistant individuals. This sensitivity is critical for a population-level triage tool — identifying 9 out of 10 at-risk individuals before they develop irreversible diabetic complications.",
              font_size=13, color=DARK_GRAY, space_before=Pt(6))
add_text_para(tf, "The hierarchical escalation to Tier 2 (with routine labs) further boosts AUROC to 0.84 - 0.88, approaching clinical diagnostic accuracy.",
              font_size=13, color=DARK_GRAY, space_before=Pt(4))


# ═══════════════════════════════════════════════════════════════
# SLIDE 10: SHAP FEATURE INTERPRETABILITY
# ═══════════════════════════════════════════════════════════════
print("Creating Slide 10: SHAP Interpretability...")
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_header_bar(slide, "Model Interpretability: SHAP Feature Attribution")
add_slide_number(slide, 10)
add_footer(slide)

# SHAP Rankings table
shap_features = [
    ("1", "Waist Circumference / WHtR", "0.766", "Visceral adiposity"),
    ("2", "BMI", "0.515", "Body mass index"),
    ("3", "Resting Heart Rate", "0.309", "Autonomic tone"),
    ("4", "Age", "0.203", "Metabolic aging"),
    ("5", "Daily Sedentary Duration", "0.151", "Physical inactivity"),
    ("6", "Daily Activity Volume", "0.114", "Energy expenditure"),
    ("7", "Circadian Stability (IS)", "0.111", "Rhythm regularity"),
    ("8", "Sleep Duration", "0.086", "Nocturnal recovery"),
]

table_shape3 = slide.shapes.add_table(len(shap_features) + 1, 4, Inches(0.5), Inches(1.3),
                                       Inches(6.0), Inches(4.0))
table3 = table_shape3.table
table3.columns[0].width = Inches(0.5)
table3.columns[1].width = Inches(2.5)
table3.columns[2].width = Inches(1.2)
table3.columns[3].width = Inches(1.8)

s_headers = ["#", "Feature", "Mean |SHAP|", "Signal Domain"]
for j, h in enumerate(s_headers):
    cell = table3.cell(0, j)
    cell.text = h
    p = cell.text_frame.paragraphs[0]
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.font.name = "Calibri"
    p.alignment = PP_ALIGN.CENTER
    cell.fill.solid()
    cell.fill.fore_color.rgb = IITG_BLUE

for i, (rank, feat, shap_val, domain) in enumerate(shap_features):
    vals = [rank, feat, shap_val, domain]
    for j, val in enumerate(vals):
        cell = table3.cell(i + 1, j)
        cell.text = val
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(11)
        p.font.name = "Calibri"
        p.alignment = PP_ALIGN.CENTER
        p.font.color.rgb = DARK_GRAY
        cell.fill.solid()
        cell.fill.fore_color.rgb = LIGHT_GRAY if i % 2 == 0 else WHITE

# SHAP beeswarm figure
try_add_image(slide, "shap_global_beeswarm_homa.png",
              Inches(6.8), Inches(1.3), width=Inches(6.0))

# Clinical insight
shap_insight = add_rounded_rect(slide, Inches(0.5), Inches(5.6), Inches(12.3), Inches(1.2),
                                 RGBColor(0xFD, 0xF0, 0xE0), border_color=GOLD)
tf = shap_insight.text_frame
tf.word_wrap = True
tf.margin_left = Inches(0.2)
tf.margin_top = Inches(0.1)
p = tf.paragraphs[0]
p.text = "Interpretability Insight: Static-Dynamic Synergy"
p.font.size = Pt(15)
p.font.color.rgb = IITG_BLUE
p.font.bold = True
add_text_para(tf, "Anthropometric features (WHtR, BMI) serve as the static metabolic prior, while wearable features (RHR, sedentary time, circadian stability) capture the dynamic physiological variance. Neither alone is sufficient — their fusion creates the clinically viable Tier 1 screener.",
              font_size=12, color=DARK_GRAY, space_before=Pt(4))


# ═══════════════════════════════════════════════════════════════
# SLIDE 11: ZERO-SHOT CROSS-COHORT TRANSFER
# ═══════════════════════════════════════════════════════════════
print("Creating Slide 11: Cross-Cohort Transfer...")
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_header_bar(slide, "Zero-Shot Cross-Cohort Generalization")
add_slide_number(slide, 11)
add_footer(slide)

# Transfer experiment description
desc_box = add_rounded_rect(slide, Inches(0.5), Inches(1.3), Inches(6.0), Inches(2.5),
                             IITG_LIGHT_BLUE, border_color=IITG_ACCENT)
tf = desc_box.text_frame
tf.word_wrap = True
tf.margin_left = Inches(0.2)
tf.margin_top = Inches(0.15)
p = tf.paragraphs[0]
p.text = "Experiment Design"
p.font.size = Pt(18)
p.font.color.rgb = IITG_BLUE
p.font.bold = True

transfer_details = [
    "▸ Train exclusively on CDC NHANES (N = 3,292)",
    "    - Population: general US adults, mean age 40.8 yrs",
    "    - Sensor: hip-worn ActiGraph GT3X+ accelerometer",
    "▸ Test zero-shot on MESA Sleep (N = 2,237)",
    "    - Population: multi-ethnic elderly, mean age 68.6 yrs",
    "    - Sensor: wrist-worn Actiwatch Spectrum",
    "▸ Zero fine-tuning, zero domain adaptation",
    "    - 28-year mean age gap, different hardware, different wear site",
]
for d in transfer_details:
    add_text_para(tf, d, font_size=12, color=DARK_GRAY, space_before=Pt(3))

# Results box
res_box = add_rounded_rect(slide, Inches(0.5), Inches(4.0), Inches(6.0), Inches(2.7),
                            RGBColor(0xE8, 0xF4, 0xE8), border_color=GREEN)
tf2 = res_box.text_frame
tf2.word_wrap = True
tf2.margin_left = Inches(0.2)
tf2.margin_top = Inches(0.15)
p = tf2.paragraphs[0]
p.text = "Transfer Results"
p.font.size = Pt(18)
p.font.color.rgb = GREEN
p.font.bold = True

transfer_results = [
    ("R^2 Retention:", "97.5%  (0.2584 vs 0.2649 internal)"),
    ("AUROC Retention:", "100.3%  (0.7485 vs 0.7466 internal)"),
    ("Pearson r:", "0.5083 (transfer) vs 0.5147 (internal)"),
]
for label, val in transfer_results:
    add_text_para(tf2, f"  {label}", font_size=14, color=GREEN, bold=True, space_before=Pt(10))
    add_text_para(tf2, f"    {val}", font_size=14, color=DARK_GRAY, space_before=Pt(2))

add_text_para(tf2, "Physiological relationships learned from one cohort generalize to an entirely unseen population without any retraining.",
              font_size=12, color=MED_GRAY, italic=True, space_before=Pt(10))

# Cross-cohort figure
try_add_image(slide, "cross_cohort_transfer_generalization.png",
              Inches(6.8), Inches(1.3), width=Inches(6.0))


# ═══════════════════════════════════════════════════════════════
# SLIDE 12: ADDITIONAL RESULTS FIGURES
# ═══════════════════════════════════════════════════════════════
print("Creating Slide 12: Additional Results...")
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_header_bar(slide, "Detailed Empirical Visualizations")
add_slide_number(slide, 12)
add_footer(slide)

# Four figures in a 2x2 grid
fig_positions = [
    ("mesa_phase1_bland_altman.png", Inches(0.5), Inches(1.3), Inches(6.0), "Bland-Altman Agreement Analysis"),
    ("mesa_phase2_confusion_matrices.png", Inches(6.8), Inches(1.3), Inches(6.0), "3-Class Confusion Matrices"),
    ("shap_clinical_feature_interactions.png", Inches(0.5), Inches(4.3), Inches(6.0), "SHAP Feature Interaction Plots"),
    ("mesa_phase3_latent_space_pca.png", Inches(6.8), Inches(4.3), Inches(6.0), "Multimodal Latent Space PCA"),
]

for fname, x, y, w, caption in fig_positions:
    try_add_image(slide, fname, x, y + Inches(0.3), width=w)
    # Caption
    cap_box = slide.shapes.add_textbox(x, y - Inches(0.05), w, Inches(0.35))
    set_text(cap_box, caption, font_size=13, color=IITG_BLUE, bold=True, alignment=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════
# SLIDE 13: PRODUCTION ARCHITECTURE
# ═══════════════════════════════════════════════════════════════
print("Creating Slide 13: Production Architecture...")
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_header_bar(slide, "Production Architecture: MetaSense AI")
add_slide_number(slide, 13)
add_footer(slide)

# Architecture diagram as boxes
components = [
    ("Smartwatch / Band", "Green/Red/IR PPG (25Hz)\nTriaxial IMU (50Hz)\nSkin Temperature",
     Inches(0.5), Inches(1.5), Inches(3.0), Inches(2.0), IITG_BLUE),
    ("Smartphone Gateway", "Edge Feature Extraction\nTFLite Inference Runtime\nTier 1 Daily Risk Score\nSQLCipher Encrypted DB",
     Inches(4.5), Inches(1.5), Inches(3.5), Inches(2.0), IITG_ACCENT),
    ("Cloud Platform", "Deep Multimodal Encoder\nFHIR Data Pipeline\nTier 2 Annual Recalibration\nModel Registry & A/B Testing",
     Inches(9.0), Inches(1.5), Inches(3.8), Inches(2.0), RGBColor(0x00, 0x60, 0x60)),
]

for name, desc, x, y, w, h, color in components:
    box = add_rounded_rect(slide, x, y, w, h, color)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.15)
    tf.margin_top = Inches(0.1)
    p = tf.paragraphs[0]
    p.text = name
    p.font.size = Pt(16)
    p.font.color.rgb = WHITE
    p.font.bold = True
    p.font.name = "Calibri"
    p.alignment = PP_ALIGN.CENTER
    for line in desc.split("\n"):
        add_text_para(tf, line, font_size=11, color=RGBColor(0xDD, 0xEE, 0xFF),
                      alignment=PP_ALIGN.CENTER, space_before=Pt(3))

# Arrows between components
arrow1 = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(3.6), Inches(2.2),
                                 Inches(0.8), Inches(0.4))
arrow1.fill.solid()
arrow1.fill.fore_color.rgb = GOLD
arrow1.line.fill.background()
set_text(arrow1, "BLE 5.2", font_size=9, color=DARK_GRAY, alignment=PP_ALIGN.CENTER)
arrow1.text_frame.margin_top = Inches(0.02)

arrow2 = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(8.1), Inches(2.2),
                                 Inches(0.8), Inches(0.4))
arrow2.fill.solid()
arrow2.fill.fore_color.rgb = GOLD
arrow2.line.fill.background()
set_text(arrow2, "TLS 1.3", font_size=9, color=DARK_GRAY, alignment=PP_ALIGN.CENTER)
arrow2.text_frame.margin_top = Inches(0.02)

# Key specs boxes
specs = [
    ("On-Device Model", "LightGBM (80%) + Ridge (20%)\nINT8 Quantized | < 3.5 MB\n< 1.5% Battery / Day"),
    ("Privacy & Compliance", "HIPAA / GDPR Compliant\nISO 11073 Personal Health\nEnd-to-End Encryption"),
    ("User Experience", "Daily Metabolic Health Index (0-100)\nSHAP Actionable Lifestyle Nudges\n1-Tap Physician PDF Export"),
]

for i, (title, desc) in enumerate(specs):
    x = Inches(0.5 + i * 4.3)
    box = add_rounded_rect(slide, x, Inches(4.2), Inches(4.0), Inches(2.5),
                           LIGHT_GRAY, border_color=IITG_ACCENT)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.15)
    tf.margin_top = Inches(0.1)
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(15)
    p.font.color.rgb = IITG_BLUE
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER
    for line in desc.split("\n"):
        add_text_para(tf, line, font_size=12, color=DARK_GRAY, alignment=PP_ALIGN.CENTER, space_before=Pt(5))


# ═══════════════════════════════════════════════════════════════
# SLIDE 14: CONCLUSION & KEY TAKEAWAYS
# ═══════════════════════════════════════════════════════════════
print("Creating Slide 14: Conclusion...")
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_header_bar(slide, "Conclusion & Key Contributions")
add_slide_number(slide, 14)
add_footer(slide)

# Main contributions
contributions = [
    ("Non-Invasive IR Screening is Clinically Viable",
     "Zero-blood wearable + anthropometric models achieve R^2 = 0.29-0.40 and AUROC = 0.75-0.78 across three independent cohorts, establishing passive metabolic surveillance as a feasible population-level screening paradigm."),
    ("Multi-Cohort Generalization Proven",
     "Zero-shot transfer from NHANES to MESA retains 97.5% of R^2 and 100.3% of AUROC — proving that learned physiological relationships are invariant to sensor hardware, wear location, and demographic shifts."),
    ("Clinically Interpretable Feature Rankings",
     "TreeSHAP analysis reveals that waist circumference (WHtR), BMI, resting heart rate, and sedentary duration are the top non-invasive predictors — all measurable by commodity smartwatches without any clinical infrastructure."),
    ("Production-Ready Architecture",
     "The dual-tier MetaSense AI system enables always-on passive screening (Tier 1) with escalation to lab-enhanced diagnostics (Tier 2), achieving 85-92% sensitivity in high-risk populations."),
]

for i, (title, desc) in enumerate(contributions):
    y = Inches(1.3 + i * 1.45)
    box = add_rounded_rect(slide, Inches(0.5), y, Inches(12.3), Inches(1.3),
                           IITG_LIGHT_BLUE if i % 2 == 0 else LIGHT_GRAY,
                           border_color=IITG_ACCENT)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.6)
    tf.margin_top = Inches(0.1)

    # Number badge
    badge = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.6), y + Inches(0.15),
                                    Inches(0.4), Inches(0.4))
    badge.fill.solid()
    badge.fill.fore_color.rgb = IITG_BLUE
    badge.line.fill.background()
    set_text(badge, str(i + 1), font_size=14, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    badge.text_frame.margin_left = Inches(0)
    badge.text_frame.margin_top = Inches(0.04)

    p = tf.paragraphs[0]
    p.text = f"     {title}"
    p.font.size = Pt(16)
    p.font.color.rgb = IITG_BLUE
    p.font.bold = True
    p.font.name = "Calibri"
    add_text_para(tf, f"     {desc}", font_size=12, color=DARK_GRAY, space_before=Pt(4))

# Future work banner
future_box = add_rounded_rect(slide, Inches(0.5), Inches(6.3), Inches(12.3), Inches(0.9),
                               RGBColor(0xFD, 0xF0, 0xE0), border_color=GOLD)
tf = future_box.text_frame
tf.word_wrap = True
tf.margin_left = Inches(0.2)
tf.margin_top = Inches(0.05)
p = tf.paragraphs[0]
p.text = "Future Directions"
p.font.size = Pt(14)
p.font.color.rgb = IITG_BLUE
p.font.bold = True
add_text_para(tf, "Multi-wavelength PPG (SpO2 estimation)  |  CGM co-training for continuous glucose integration  |  Foundation models for wearable time-series  |  Prospective clinical validation (RCT)  |  Indian population study",
              font_size=12, color=DARK_GRAY, space_before=Pt(4))


# ═══════════════════════════════════════════════════════════════
# SLIDE 15: THANK YOU / Q&A (BONUS)
# ═══════════════════════════════════════════════════════════════
print("Creating Slide 15: Thank You...")
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, IITG_BLUE)

# Logo
if os.path.exists(logo_path):
    slide.shapes.add_picture(logo_path, Inches(5.5), Inches(0.8), height=Inches(2.0))

# Thank you text
txBox = slide.shapes.add_textbox(Inches(2.0), Inches(3.0), Inches(9.3), Inches(1.5))
tf = txBox.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Thank You"
p.font.size = Pt(48)
p.font.color.rgb = WHITE
p.font.bold = True
p.font.name = "Calibri"
p.alignment = PP_ALIGN.CENTER

add_text_para(tf, "Questions & Discussion", font_size=28, color=RGBColor(0xBB, 0xDD, 0xFF),
              bold=False, alignment=PP_ALIGN.CENTER, space_before=Pt(12))

# Contact info box
contact_box = add_rounded_rect(slide, Inches(3.0), Inches(5.0), Inches(7.3), Inches(1.8),
                                RGBColor(0x00, 0x50, 0x90))
tf2 = contact_box.text_frame
tf2.word_wrap = True
tf2.margin_left = Inches(0.3)
tf2.margin_top = Inches(0.15)
p = tf2.paragraphs[0]
p.text = "Sarvagya Tiwari  |  Roll No. 230122047"
p.font.size = Pt(18)
p.font.color.rgb = WHITE
p.font.bold = True
p.font.name = "Calibri"
p.alignment = PP_ALIGN.CENTER

add_text_para(tf2, "Department of Chemistry, IIT Guwahati", font_size=15,
              color=RGBColor(0xBB, 0xDD, 0xFF), alignment=PP_ALIGN.CENTER, space_before=Pt(6))
add_text_para(tf2, "Supervisor: Prof. Ritika Jain  |  rjain1992@iitg.ac.in", font_size=14,
              color=RGBColor(0xBB, 0xDD, 0xFF), alignment=PP_ALIGN.CENTER, space_before=Pt(4))
add_text_para(tf2, "github.com/SARVAGYA-TIWARI/Biomedical-AI-Wearable-Sensing", font_size=12,
              color=RGBColor(0x99, 0xCC, 0xEE), alignment=PP_ALIGN.CENTER, space_before=Pt(4))


# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════
prs.save(OUTPUT_PATH)
print(f"\n✅ Presentation saved to: {OUTPUT_PATH}")
print(f"   Total slides: {len(prs.slides)}")

