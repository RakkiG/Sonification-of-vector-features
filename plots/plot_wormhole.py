import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a figure and axis
fig, ax = plt.subplots(figsize=(20, 15))

# Add the boxes (rectangles) and labels
# Box 1: "wormhole"
ax.add_patch(patches.Rectangle((0, 0), 0.5, 0.25, edgecolor='black', facecolor='none', lw=2))
ax.text(0.25, 0.125, "$wormhole$", ha='center', va='center', fontsize=12)

# Box 2: "$S_{new} = r*S_{ori}$" and "$f_{new} = r*f_{ori}$"
ax.add_patch(patches.Rectangle((2, 0), 0.7, 0.3, edgecolor='black', facecolor='none', lw=2))
ax.text(2.4, 0.2, "$S_{new} = r \cdot S_{ori}$", ha='center', va='center', fontsize=12)
ax.text(2.4, 0.1, "$f_{new} = r \cdot f_{ori}$", ha='center', va='center', fontsize=12)

# Box 3: "$i$"
ax.add_patch(patches.Rectangle((0, 1), 0.2, 0.125, edgecolor='black', facecolor='none', lw=2))
ax.text(0.1, 1.0625, "$i$", ha='center', va='center', fontsize=12)

# Box 4: "$r$"
ax.add_patch(patches.Rectangle((2, 1), 0.2, 0.125, edgecolor='black', facecolor='none', lw=2))
ax.text(2.1, 1.0625, "$r$", ha='center', va='center', fontsize=12)

# Add arrows
ax.annotate("", xy=(2, 1.0625), xytext=(0.2, 1.0625), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.text(1, 1.075, "$mapping$", fontsize=12)

ax.annotate("", xy=(1, 0.25), xytext=(2, 0.25), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.text(1.4, 0.28, "$resampling(\\ )$", fontsize=12)

ax.annotate("", xy=(2.25, 1), xytext=(1.57, 0.27), arrowprops=dict(arrowstyle="->", lw=1.5))

# Set limits and hide axes
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 2)
ax.axis('off')

# Show the plot
plt.show()
