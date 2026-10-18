"""
Generate presentation-ready figure for MeSH Concept Normalization and Topic Grounding.
Saves to figures/mesh_concept_normalization.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
root = Path(__file__).resolve().parent.parent
fig_dir = root / "figures"
fig_dir.mkdir(exist_ok=True)
out_path = fig_dir / "mesh_concept_normalization.png"

# Setup Figure
fig = plt.figure(figsize=(16, 8.5), dpi=180, facecolor='#ffffff')
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.35], height_ratios=[1, 1.1], hspace=0.32, wspace=0.28)

# -------------------------------------------------------------
# Panel A: Variant Normalization Engine (Top Left)
# -------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('#ffffff')
ax1.set_title('(A) Morphological & Synonym Normalization Rules', fontsize=12, fontweight='bold', pad=12, color='#0d47a1')

rules = [
    ('cell phone / smart phone / cell phones', 'smartphone', 'Hardware'),
    ('mobile application / mobile app', 'mobile applications', 'Software'),
    ('tele-medicine / tele health', 'telemedicine', 'Delivery'),
    ('self-care / self management', 'self care', 'Behavioral'),
    ('follow-up / follow up study', 'follow-up studies', 'Methodology')
]

ax1.set_ylim(-0.5, len(rules) - 0.5)
ax1.set_xlim(0, 12)
ax1.axis('off')

for i, (orig, norm, cat) in enumerate(reversed(rules)):
    # Box for original raw variants
    ax1.text(0.1, i, f'"{orig}"', fontsize=8.0, va='center', ha='left',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='#f8f9fa', edgecolor='#cfd8dc', lw=1.2))
    # Flow arrow
    ax1.annotate('', xy=(6.2, i), xytext=(5.2, i),
                 arrowprops=dict(arrowstyle='->', lw=2, color='#0288d1'))
    # Box for canonical MeSH heading
    ax1.text(6.4, i, norm, fontsize=8.8, fontweight='bold', va='center', ha='left',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='#e1f5fe', edgecolor='#0288d1', lw=1.5), color='#01579b')
    # Domain Tag
    ax1.text(11.8, i, f'[{cat}]', fontsize=8, va='center', ha='right', color='#546e7a', style='italic')

# -------------------------------------------------------------
# Panel B: Concept Quality Filtering Funnel (Bottom Left)
# -------------------------------------------------------------
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('#ffffff')
ax2.set_title('(B) Vocabulary Quality & Clinical Filtering Funnel', fontsize=12, fontweight='bold', pad=12, color='#0d47a1')

counts = [2790, 175, 1878, 737]
labels = [
    '2,790 Total Raw Concepts',
    '175 Non-Clinical Demographic Stops',
    '1,878 Sparsity Pruned (< 5 docs)',
    '737 High-Signal Clinical Concepts'
]
colors = ['#1976d2', '#e53935', '#78909c', '#2e7d32']

bars = ax2.barh([3, 2, 1, 0], counts, color=colors, height=0.55, edgecolor='white', lw=1.5)
ax2.set_yticks([3, 2, 1, 0])
ax2.set_yticklabels([
    'Raw Vocabulary',
    'Demographic Filter\n(humans, male, etc.)',
    'Rare Singletons',
    'Retained Concepts'
], fontsize=9.5)
ax2.set_xlabel('Concept Count', fontsize=10, fontweight='bold', color='#37474f')
ax2.set_xlim(0, 3600)

for bar, label in zip(bars, labels):
    ax2.text(bar.get_width() + 65, bar.get_y() + bar.get_height()/2, label,
             va='center', fontsize=8.8, fontweight='bold', color='#263238')
ax2.grid(axis='x', linestyle='--', alpha=0.5)

# -------------------------------------------------------------
# Panel C: Topic-to-MeSH Concept Grounding (Right - Spanning Both Rows)
# -------------------------------------------------------------
ax3 = fig.add_subplot(gs[:, 1])
ax3.set_facecolor('#ffffff')
ax3.set_title('(C) Discovered Topics Grounded in Canonical MeSH Concepts', fontsize=12, fontweight='bold', pad=12, color='#0d47a1')

grounding_data = [
    ('Topic 9: Cardiac Telemetry', [('arrhythmias, cardiac', 62), ('heart rate', 85), ('electrocardiography', 104)]),
    ('Topic 5: Smoking Cessation', [('substance-related disorders', 94), ('smoking', 112), ('tobacco use cessation', 168)]),
    ('Topic 4: Oncology & Cancer Support', [('patient outcome assessment', 76), ('cancer survivors', 89), ('breast neoplasms', 124)]),
    ('Topic 1: HIV Prevention & Care', [('medication adherence', 118), ('anti-retroviral agents', 142), ('hiv infections', 355)]),
    ('Topic 0: Maternal & Community Care', [('community health workers', 100), ('infant', 103), ('pregnancy', 182)])
]

y_ticks = []
y_labels = []
bar_y = 0
colors_bars = ['#26a69a', '#00acc1', '#00897b']

topic_positions = []

for t_title, concepts in grounding_data:
    start_y = bar_y
    for idx, (c_name, c_cnt) in enumerate(concepts):
        bar = ax3.barh(bar_y, c_cnt, color=colors_bars[idx % 3], height=0.62, edgecolor='white', lw=1.2)
        ax3.text(c_cnt + 6, bar_y, f'{c_cnt} docs', va='center', fontsize=8.5, fontweight='bold', color='#37474f')
        y_ticks.append(bar_y)
        y_labels.append(c_name)
        bar_y += 1
    mid_y = (start_y + bar_y - 1) / 2
    topic_positions.append((t_title, mid_y))
    # Separator
    ax3.axhline(bar_y - 0.5, color='#cfd8dc', linestyle='-', lw=1.2)
    bar_y += 0.5

ax3.set_yticks(y_ticks)
ax3.set_yticklabels(y_labels, fontsize=9.5)
ax3.set_xlabel('Document Grounding Frequency (Number of Papers)', fontsize=10, fontweight='bold', color='#37474f')
ax3.set_xlim(0, 440)
ax3.grid(axis='x', linestyle='--', alpha=0.5)

# Place badges correctly matching mid_y of each topic group
for t_title, mid_y in topic_positions:
    ax3.text(430, mid_y, t_title, ha='right', va='center', fontsize=9.2, fontweight='bold',
             color='#0d47a1', bbox=dict(boxstyle='round,pad=0.35', facecolor='#e3f2fd', edgecolor='#90caf9', lw=1.2))

fig.suptitle('Biomedical Concept Extraction & MeSH Normalization Architecture', fontsize=15, fontweight='bold', y=0.98, color='#0d47a1')

plt.savefig(out_path, bbox_inches='tight', facecolor='#ffffff')
plt.close(fig)
print(f"[SUCCESS] Figure finalized at {out_path}")
