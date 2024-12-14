
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

from PIL import Image

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Define the steps and their positions
steps = [
    "Start with small vortex corelines",
    "Perform distance-based connectivity check",
    "Group connected lines into groups",
    "Merge groups(twice)",
    "Finalize connected vortex corelines"
]

positions = [
    (0.5, 5),
    (0.5, 4),
    (0.5, 3),
    (0.5, 2),
    (0.5, 1),
    (0.5, 0)
]

# Draw the steps as rectangles and add text
for pos, step in zip(positions, steps):
    rect = mpatches.Rectangle((pos[0]-0.26, pos[1]-0.2), 0.515, 0.4, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    ax.text(pos[0], pos[1], step, ha='center', va='center', fontsize=12.5, wrap=True)

# Draw arrows between steps
for i in range(len(positions) - 2):
    x_start, y_start = positions[i]
    x_end, y_end = positions[i+1]
    ax.annotate("", xy=(x_end, y_end + 0.15), xytext=(x_start, y_start - 0.17),
                                arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

# Set limits and hide axes
ax.set_xlim(0, 1)
ax.set_ylim(-0.5, 5.5)
ax.axis('off')


output_path = "diagrams/flowchart_vortex_corelines_group.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
fig.savefig(output_path, bbox_inches='tight')

with Image.open(output_path) as img:
    cropped = img.crop((150, 20, img.width-150, img.height-100))
    cropped.save(output_path)

