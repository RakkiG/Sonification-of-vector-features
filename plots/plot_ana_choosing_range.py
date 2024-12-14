
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 2))

# Draw the horizontal line
dash_x = np.linspace(-1, 9, 100)
dash_y = np.full_like(dash_x, 0.5)
ax.plot(dash_x, dash_y, color="black")

# Define and draw the blocks
block1 = [[0, 0], [8, 0], [8, 1], [0, 1], [0, 0]]
block2 = [[2, 0], [2, 1], [5.6, 1], [5.6, 0], [2, 0]]
block1 = np.array(block1)
block2 = np.array(block2)
ax.fill(block1[:, 0], block1[:, 1], color="black", alpha=0.3, label="Block 1")
ax.fill(block2[:, 0], block2[:, 1], color="blue", alpha=0.5, label="Block 2")

shift_two_arrows = 1

# Draw arrows
ax.annotate('', xy=(4-shift_two_arrows, 1.3), xytext=(4.7-shift_two_arrows, 1.3),
            arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='->'))
ax.annotate('', xy=(5.6-shift_two_arrows, 1.3), xytext=(4.9-shift_two_arrows, 1.3),
            arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='->'))


# Add labels and connecting lines
# Label and line for the blue block
ax.text(4.2, -0.5, "$x_{nat}$", fontsize=12, color="blue")
ax.annotate('', xy=(3.5, 0.3), xytext=(4.2, -0.4),
            arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='-'))

# Label and line for the gray block
ax.text(-0.5, -0.5, "$x_{ana\_crange}$", fontsize=15, color="black")
ax.annotate('', xy=(0, -0.4), xytext=(0.3, 0.3),
            arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='-'))

# Set axis limits
ax.set_xlim(-1, 9)
ax.set_ylim(-0.5, 2)

# Hide axes
ax.axis('off')

# Show the plot
plt.savefig("block_and_arrow_plot.png", dpi=300, bbox_inches='tight', pad_inches=0)
with Image.open("../diagrams/block_and_arrow_plot.png") as img:
    cropped = img.crop((20, 110, img.width, img.height) )  #
    cropped.save("block_and_arrow_plot.png")
plt.show()
