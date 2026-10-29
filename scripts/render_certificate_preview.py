import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import textwrap

fig, ax = plt.subplots(figsize=(8.5, 11), dpi=200)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Double outer border like IITG official certificates
rect1 = patches.Rectangle((0.04, 0.04), 0.92, 0.92, fill=False, edgecolor='#222222', linewidth=2.0)
rect2 = patches.Rectangle((0.046, 0.046), 0.908, 0.908, fill=False, edgecolor='#222222', linewidth=1.0)
ax.add_patch(rect1)
ax.add_patch(rect2)

# Place IITG Logo
logo = Image.open(r'd:\BTP\iitg_logo.png')
ax.imshow(logo, extent=[0.41, 0.59, 0.77, 0.89], aspect='auto', zorder=3)

# Header
ax.text(0.5, 0.92, 'Indian Institute of Technology Guwahati', ha='center', va='center', fontsize=15, fontweight='bold', fontfamily='serif')

# Certificate Title
ax.text(0.5, 0.72, 'Certificate', ha='center', va='center', fontsize=22, fontweight='bold', fontfamily='serif')

# Body Text with proper wrapping
body1 = "This is to certify that Mr. Sarvagya Tiwari (Roll No. 230122047) has been working under my supervision as a Project Student in my research group since July 2025."
wrapped1 = textwrap.fill(body1, width=74)
ax.text(0.09, 0.64, wrapped1, ha='left', va='top', fontsize=11, fontfamily='serif', linespacing=1.4)

body2 = 'His project, entitled "Non-Invasive Insulin Resistance Screening Using Multimodal Wearable Sensing and Explainable AI", is being forwarded for submission. I hereby certify that he has satisfactorily completed all the requirements in accordance with the rules and regulations of the Institute pertaining to the conduct of such investigations.'
wrapped2 = textwrap.fill(body2, width=74)
ax.text(0.09, 0.54, wrapped2, ha='left', va='top', fontsize=11, fontfamily='serif', linespacing=1.4)

body3 = "To the best of my knowledge, the results presented in this project report have not been submitted elsewhere for the award of any degree."
wrapped3 = textwrap.fill(body3, width=74)
ax.text(0.09, 0.40, wrapped3, ha='left', va='top', fontsize=11, fontfamily='serif', linespacing=1.4)

# Supervisor info
supervisor_text = "Prof. Ritika Jain\nIndian Institute of Technology Guwahati\ne-mail: rjain1992@iitg.ac.in\n\nDate: May 2026"
ax.text(0.09, 0.24, supervisor_text, ha='left', va='top', fontsize=11, fontfamily='serif', linespacing=1.3)

# Signature line
ax.plot([0.58, 0.91], [0.13, 0.13], color='black', lw=1.2)
ax.text(0.745, 0.10, 'Signature of Supervisor', ha='center', va='center', fontsize=11, fontfamily='serif', fontweight='bold')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
out_png = r'd:\BTP\IITG_BTP_Certificate_Sarvagya_Tiwari_Preview.png'
plt.savefig(out_png, bbox_inches='tight', dpi=200)
print(f'Successfully generated certificate preview image at: {out_png}')
