"""
Generate high-res preview image for the enlarged, editable slide.
Saves to figures/slide_concept_extraction_mesh_normalization.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

root = Path(__file__).resolve().parent.parent
fig_dir = root / "figures"
fig_dir.mkdir(exist_ok=True)
out_path = fig_dir / "slide_concept_extraction_mesh_normalization.png"

fig = plt.figure(figsize=(16, 9), dpi=200, facecolor='#fcfbf9')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# Top accent bar (#B54C34)
ax.add_patch(patches.Rectangle((0, 8.9), 16, 0.1, facecolor='#B54C34', edgecolor='none'))

# Slide Title & Subtitle
ax.text(0.8, 8.2, "BIOMEDICAL CONCEPT EXTRACTION & MeSH NORMALIZATION",
        fontsize=21, fontweight='bold', color='#282825', family='sans-serif')
ax.text(0.8, 7.75, "Bridging Discovered Topics to Standardized Clinical Entities for Research Gap Detection",
        fontsize=13, color='#5a5a55', style='italic', family='sans-serif')

# ==============================================================
# LEFT CARD: STAGE 4 — BIOMEDICAL CONCEPT EXTRACTION
# ==============================================================
ax.add_patch(patches.FancyBboxPatch((0.75, 0.75), 6.95, 6.7,
                                   boxstyle="round,pad=0.2,rounding_size=0.15",
                                   facecolor='#ffffff', edgecolor='#dce0e6', linewidth=1.5))
ax.add_patch(patches.Rectangle((0.75, 0.75), 0.16, 6.7, facecolor='#B54C34', edgecolor='none'))

# Stage 4 Header
ax.text(1.2, 7.05, "STAGE 4 — BIOMEDICAL CONCEPT EXTRACTION",
        fontsize=15, fontweight='bold', color='#B54C34')
ax.text(1.2, 6.65, "Discover important medical ideas present inside research topics",
        fontsize=11.5, color='#5a5a55')

# Sources Box
ax.add_patch(patches.FancyBboxPatch((1.15, 5.6), 6.25, 0.85,
                                   boxstyle="round,pad=0.1,rounding_size=0.08",
                                   facecolor='#fef2ed', edgecolor='#ffccbc', linewidth=1.2))
ax.text(1.3, 6.15, "Extraction Sources across 12,642 Papers:",
        fontsize=11, fontweight='bold', color='#bf360c')
ax.text(1.3, 5.75, "• MeSH Descriptors   • Author Keywords   • Medical Noun Phrases",
        fontsize=11.5, fontweight='bold', color='#282825')

# Quantitative 3-Step Box
ax.add_patch(patches.FancyBboxPatch((1.15, 3.85), 6.25, 1.5,
                                   boxstyle="round,pad=0.1,rounding_size=0.08",
                                   facecolor='#f8fafc', edgecolor='#cfd8dc', linewidth=1.2))

# Big numbers
ax.text(1.8, 4.65, "19,689", fontsize=22, fontweight='bold', color='#475569', ha='center')
ax.text(1.8, 4.15, "Raw Concepts", fontsize=10.5, fontweight='bold', color='#64748b', ha='center')

ax.text(3.3, 4.6, "→", fontsize=22, fontweight='bold', color='#94a3b8', ha='center')

ax.text(4.4, 4.65, "Freq ≥ 5", fontsize=16, fontweight='bold', color='#B54C34', ha='center')
ax.text(4.4, 4.15, "Occurrence Filter", fontsize=10, fontweight='bold', color='#64748b', ha='center')

ax.text(5.5, 4.6, "→", fontsize=22, fontweight='bold', color='#94a3b8', ha='center')

ax.text(6.5, 4.65, "2,790", fontsize=22, fontweight='bold', color='#2e7d32', ha='center')
ax.text(6.5, 4.15, "Retained Concepts", fontsize=10.5, fontweight='bold', color='#2e7d32', ha='center')

# Bullets
ax.text(1.2, 3.35, "• Topics identify broad research themes; concepts reveal exact medical ideas.",
        fontsize=11.5, color='#282825')
ax.text(1.2, 2.95, "• Extracts specific clinical entities, technologies, symptoms, and therapies.",
        fontsize=11.5, color='#282825')

# Easy Memory Box
ax.add_patch(patches.FancyBboxPatch((1.15, 1.05), 6.25, 1.6,
                                   boxstyle="round,pad=0.1,rounding_size=0.08",
                                   facecolor='#fef9e7', edgecolor='#f59e0b', linewidth=1.5))
ax.text(1.35, 2.25, "EASY MEMORY KEY:", fontsize=11.5, fontweight='bold', color='#b45309')
ax.text(1.35, 1.75, "★  Topics  =  Research Area  (e.g., Topic 1: HIV Care)", fontsize=12.5, fontweight='bold', color='#1f2937')
ax.text(1.35, 1.30, "★  Concepts  =  Important Medical Ideas  (e.g., medication adherence)", fontsize=12.5, fontweight='bold', color='#1f2937')


# ==============================================================
# RIGHT CARD: STAGE 5 — MeSH NORMALIZATION
# ==============================================================
ax.add_patch(patches.FancyBboxPatch((8.3, 0.75), 6.95, 6.7,
                                   boxstyle="round,pad=0.2,rounding_size=0.15",
                                   facecolor='#ffffff', edgecolor='#dce0e6', linewidth=1.5))
ax.add_patch(patches.Rectangle((8.3, 0.75), 0.16, 6.7, facecolor='#2e7d32', edgecolor='none'))

# Stage 5 Header
ax.text(8.75, 7.05, "STAGE 5 — MeSH NORMALIZATION",
        fontsize=15, fontweight='bold', color='#2e7d32')
ax.text(8.75, 6.65, "Standardize concept names and eliminate duplicate lexical forms",
        fontsize=11.5, color='#5a5a55')

# Purpose Box
ax.add_patch(patches.FancyBboxPatch((8.7, 5.6), 6.25, 0.85,
                                   boxstyle="round,pad=0.1,rounding_size=0.08",
                                   facecolor='#f0fdf4', edgecolor='#bbf7d0', linewidth=1.2))
ax.text(8.85, 6.15, "Purpose: Ensure Consistency & Merge Synonyms",
        fontsize=11, fontweight='bold', color='#166534')
ax.text(8.85, 5.75, "• Merges plural forms, hyphens, and variants (e.g., 'cell phone' → 'smartphone').",
        fontsize=11.5, fontweight='bold', color='#282825')

# Normalization 3-Step Box
ax.add_patch(patches.FancyBboxPatch((8.7, 3.85), 6.25, 1.5,
                                   boxstyle="round,pad=0.1,rounding_size=0.08",
                                   facecolor='#f8fafc', edgecolor='#cfd8dc', linewidth=1.2))

# Big numbers
ax.text(9.35, 4.65, "2,790", fontsize=22, fontweight='bold', color='#475569', ha='center')
ax.text(9.35, 4.15, "Extracted Concepts", fontsize=10.5, fontweight='bold', color='#64748b', ha='center')

ax.text(10.5, 4.6, "→", fontsize=22, fontweight='bold', color='#94a3b8', ha='center')

ax.text(11.6, 4.65, "2,786", fontsize=22, fontweight='bold', color='#2563eb', ha='center')
ax.text(11.6, 4.15, "4 Duplicates Merged", fontsize=10, fontweight='bold', color='#2563eb', ha='center')

ax.text(12.7, 4.6, "→", fontsize=22, fontweight='bold', color='#94a3b8', ha='center')

ax.text(13.9, 4.65, "21,439", fontsize=22, fontweight='bold', color='#2e7d32', ha='center')
ax.text(13.9, 4.15, "Topic-Concept Pairs", fontsize=10.5, fontweight='bold', color='#2e7d32', ha='center')

# Bullets
ax.text(8.75, 3.35, "• Establishes formal bipartite connections linking 63 topics to normalized MeSH terms.",
        fontsize=11.5, color='#282825')
ax.text(8.75, 2.95, "• Avoids fragmented metrics by grouping morphological and synonym variations.",
        fontsize=11.5, color='#282825')

# Pipeline Bridge Box
ax.add_patch(patches.FancyBboxPatch((8.7, 1.05), 6.25, 1.6,
                                   boxstyle="round,pad=0.1,rounding_size=0.08",
                                   facecolor='#eff6ff', edgecolor='#3b82f6', linewidth=1.5))
ax.text(8.85, 2.25, "STRUCTURED PIPELINE BRIDGE:", fontsize=11.5, fontweight='bold', color='#1d4ed8')
ax.text(8.85, 1.75, "Papers   ◄────────►   Topics   ◄────────►   Biomedical Concepts",
        fontsize=12.5, fontweight='bold', color='#1e40af')
ax.text(8.85, 1.30, "Direct input for semantic relationship analysis and candidate research gap scoring.",
        fontsize=11, color='#374151')

# Slide Number Text
ax.text(15.2, 0.35, "10b", fontsize=11, color='#9e9e9e', ha='right', family='sans-serif')

plt.savefig(out_path, bbox_inches='tight', facecolor='#fcfbf9')
plt.close(fig)
print(f"[SUCCESS] Preview image regenerated -> {out_path}")
