import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle, FancyArrow
import os

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 7))

# Helper function to add blocks
def add_block(x, y, width, height, text):
    rect = Rectangle((x, y), height, width, edgecolor="black", facecolor="white", lw=1.5)  # Swap width and height
    ax.add_patch(rect)
    ax.text(x + height / 2, y + width / 2, text, ha="center", va="center", fontsize=12, fontweight="bold")

# Helper function to add arrows
def add_arrow(x1, y1, x2, y2):
    arrow = FancyArrow(x1, y1, x2 - x1, y2 - y1, width=0.01, head_width=0.06, head_length=0.2,
                       length_includes_head=True, color="black")
    ax.add_patch(arrow)

# Add blocks in horizontal layout
add_block(0, 0, 0.6, 2, r"$\mathbf{n_p}$")
add_block(3, 0, 0.6, 2.5, r"$\mathbf{n_1, n_2, \dots, n_n}$")
add_block(6.5, 0, 0.6, 2, r"$\mathbf{C_{n,i}}$")
add_block(9.5, 0, 0.6, 2, r"$\mathbf{P_{n,i}}$")
add_block(12.5, 0, 0.6, 2.73, r"$\mathbf{P = \sum_{i=1}^{n} C_{n,i}^{0} P_{n,i}}$")

# Add arrows connecting the blocks
add_arrow(2, 0.3, 3, 0.3)  # Arrow from block 1 to block 2
add_arrow(5.5, 0.3, 6.5, 0.3)  # Arrow from block 2 to block 3
add_arrow(8.5, 0.3, 9.5, 0.3)  # Arrow from block 3 to block 4
add_arrow(11.5, 0.3, 12.5, 0.3)  # Arrow from block 4 to block 5

# Set axis limits and hide axes
ax.set_xlim(-1, 15.5)
ax.set_ylim(0, 6)
ax.axis("off")

# Create the 'diagrams' directory if it does not exist
output_dir = "../diagrams"
os.makedirs(output_dir, exist_ok=True)

# Save the plot
output_path = os.path.join(output_dir, "horizontal_method_ori.png")
plt.savefig(output_path, bbox_inches="tight")
plt.show()
with Image.open(output_path) as img:
    cropped = img.crop((55, 490, img.width-20, img.height) )
    cropped.save(output_path)
print(f"Plot saved to {output_path}")




