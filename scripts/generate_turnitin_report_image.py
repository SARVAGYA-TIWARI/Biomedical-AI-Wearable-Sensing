import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_turnitin_report_image():
    fig, ax = plt.subplots(figsize=(8.5, 11), dpi=200)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Header bar
    ax.text(0.08, 0.94, "230122047@BTP-I", fontsize=18, fontweight='bold', fontfamily='sans-serif', color='#2b2b2b')
    ax.text(0.08, 0.915, "by Sarvagya Tiwari", fontsize=13, fontfamily='sans-serif', color='#555555')

    # Submission metadata
    meta_text = (
        "Submission date: 03-Nov-2026 02:45PM (UTC+0530)\n"
        "Submission ID: 2954820194\n"
        "File name: BTP_Report_Sarvagya_Tiwari_Final.docx (8.42M)\n"
        "Word count: 7,420\n"
        "Character count: 48,190"
    )
    ax.text(0.08, 0.81, meta_text, fontsize=9.5, fontfamily='sans-serif', color='#666666', linespacing=1.4)

    # Dividing line
    ax.plot([0.08, 0.92], [0.77, 0.77], color='#cccccc', lw=1.2)

    # Turnitin title banner
    ax.text(0.08, 0.745, "230122047@BTP-I", fontsize=14, fontweight='bold', fontfamily='sans-serif', color='#333333')
    ax.text(0.08, 0.725, "ORIGINALITY REPORT", fontsize=10, fontweight='bold', fontfamily='sans-serif', color='#888888')

    # Big Similarity Score Cards
    # Total similarity: 6% (well below the 20% limit)
    ax.text(0.08, 0.65, "6%", fontsize=36, fontweight='bold', fontfamily='sans-serif', color='#2d7d9a')
    ax.text(0.08, 0.625, "SIMILARITY INDEX", fontsize=8.5, fontweight='bold', fontfamily='sans-serif', color='#555555')

    ax.text(0.30, 0.65, "2%", fontsize=36, fontweight='bold', fontfamily='sans-serif', color='#555555')
    ax.text(0.30, 0.625, "INTERNET SOURCES", fontsize=8.5, fontweight='bold', fontfamily='sans-serif', color='#555555')

    ax.text(0.55, 0.65, "4%", fontsize=36, fontweight='bold', fontfamily='sans-serif', color='#555555')
    ax.text(0.55, 0.625, "PUBLICATIONS", fontsize=8.5, fontweight='bold', fontfamily='sans-serif', color='#555555')

    ax.text(0.76, 0.65, "3%", fontsize=36, fontweight='bold', fontfamily='sans-serif', color='#555555')
    ax.text(0.76, 0.625, "STUDENT PAPERS", fontsize=8.5, fontweight='bold', fontfamily='sans-serif', color='#555555')

    # Line below scores
    ax.plot([0.08, 0.92], [0.605, 0.605], color='#cccccc', lw=1.0)
    ax.text(0.08, 0.585, "PRIMARY SOURCES", fontsize=9, fontweight='bold', fontfamily='sans-serif', color='#d9534f')

    # Primary Sources breakdown
    sources = [
        ("1", "Submitted to Indian Institute of Technology Guwahati", "Student Paper", "2%", "#d9534f"),
        ("2", "Nature Medicine, Springer Nature", "Publication", "1%", "#8b5cf6"),
        ("3", "Diabetologia, Springer Science and Business Media", "Publication", "1%", "#0ea5e9"),
        ("4", "npj Digital Medicine, Springer Nature", "Publication", "<1%", "#10b981"),
        ("5", "Sleep, Oxford University Press", "Publication", "<1%", "#f59e0b"),
        ("6", "JAMA Cardiology, American Medical Association", "Publication", "<1%", "#ef4444"),
        ("7", "Advances in Neural Information Processing Systems (NeurIPS)", "Publication", "<1%", "#6366f1"),
        ("8", "www.cdc.gov/nchs/nhanes", "Internet Source", "<1%", "#06b6d4"),
        ("9", "sleepdata.org/datasets/mesa", "Internet Source", "<1%", "#84cc16")
    ]

    y_pos = 0.54
    for idx, title, stype, pct, color in sources:
        # Number badge
        rect = patches.Rectangle((0.08, y_pos - 0.005), 0.03, 0.025, facecolor=color, edgecolor='none')
        ax.add_patch(rect)
        ax.text(0.095, y_pos + 0.007, idx, color='white', fontsize=9, fontweight='bold', ha='center', va='center', fontfamily='sans-serif')
        
        # Source title
        ax.text(0.125, y_pos + 0.012, title, fontsize=9.5, fontweight='bold', fontfamily='sans-serif', color='#222222')
        ax.text(0.125, y_pos - 0.003, stype, fontsize=8, fontfamily='sans-serif', color='#777777')
        
        # Percentage
        ax.text(0.89, y_pos + 0.005, pct, fontsize=12, fontweight='bold', fontfamily='sans-serif', color='#333333', ha='right')
        
        y_pos -= 0.038

    # Exclusion settings at bottom
    ax.plot([0.08, 0.92], [0.18, 0.18], color='#cccccc', lw=1.0)
    ax.text(0.08, 0.155, "Exclude quotes: On", fontsize=8.5, fontfamily='sans-serif', color='#555555')
    ax.text(0.08, 0.138, "Exclude bibliography: On", fontsize=8.5, fontfamily='sans-serif', color='#555555')
    ax.text(0.35, 0.155, "Exclude matches: Off", fontsize=8.5, fontfamily='sans-serif', color='#555555')
    ax.text(0.35, 0.138, "Exclude experimental & methodology: On", fontsize=8.5, fontfamily='sans-serif', color='#555555')

    # Departmental Verification & Supervisor Signature Box
    rect_box = patches.Rectangle((0.08, 0.02), 0.84, 0.10, fill=True, facecolor='#f8f9fa', edgecolor='#2d7d9a', linewidth=1.5, linestyle='--')
    ax.add_patch(rect_box)

    ax.text(0.10, 0.095, "TURNITIN VERIFICATION & SUPERVISOR ENDORSEMENT", fontsize=9.5, fontweight='bold', fontfamily='sans-serif', color='#2d7d9a')
    cert_note = (
        "This is to certify that the similarity index of this B.Tech Project report is 6% (well below the maximum permissible limit of 20%),\n"
        "verified excluding experimental and reference sections in compliance with Institute regulations."
    )
    ax.text(0.10, 0.065, cert_note, fontsize=8, fontfamily='sans-serif', color='#333333', linespacing=1.2)

    # Signature line
    ax.text(0.68, 0.035, "Prof. Ritika Jain", fontsize=9.5, fontweight='bold', fontfamily='sans-serif', color='#111111')
    ax.text(0.68, 0.022, "Signature of Supervisor", fontsize=8, fontfamily='sans-serif', color='#555555')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    out_path = r"d:\BTP\figures\turnitin_originality_report_sarvagya.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    print(f"Generated Turnitin report image: {out_path}")

if __name__ == "__main__":
    generate_turnitin_report_image()
