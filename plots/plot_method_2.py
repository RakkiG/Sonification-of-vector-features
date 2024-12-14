#
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle, FancyArrow
#
# def draw_box(ax, xy, width, height, text, fontsize=10):
#     """Draw a rectangle with text centered in it."""
#     ax.add_patch(Rectangle(xy, width, height, edgecolor='black', facecolor='none', lw=1))
#     ax.text(xy[0] + width / 2, xy[1] + height / 2, text, fontsize=fontsize,
#             ha='center', va='center')
#
# def draw_arrow(ax, start, end):
#     """Draw an arrow from start to end."""
#     ax.add_patch(FancyArrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
#                             width=0.02, head_width=0.15, head_length=0.3, color='black'))
#
# # Create the figure and axis
# fig, ax = plt.subplots(figsize=(12, 10))
# ax.set_xlim(-6, 6)
# ax.set_ylim(-4, 6)
# ax.axis('off')
#
# # Node positions
# positions = {
#     "root": (0, 5),
#     "lessThan0": (-5, 3),
#     "equals0": (0, 3),
#     "greaterThan0": (5, 3),
#     "nRange1": (-5, 1),
#     "nEquals16": (0, 1),
#     "nRange2": (5, 1),
#     "formula": (-5, -1),
#     "result": (-1.7, -3)
# }
#
# # Draw boxes
# nodes = {
#     "root": "$r_i$",
#     "lessThan0": "$<0$",
#     "equals0": "$==0$",
#     "greaterThan0": "$>0$",
#     "nRange1": "$n: [19, 25]$",
#     "nEquals16": "$n: 16$",
#     "nRange2": "$n: [8, 14]$",
#     "formula": "$f_n = \\\\sqrt{\\\\frac{1}{4\\\\pi^2} (n-1)(n+1)(n+2) \\\\frac{\\\\sigma}{\\\\rho r_0^3}}$",
#     "result": "$p_1 + p_2 + p_3$"
# }
#
# box_sizes = {
#     "root": (2, 1),
#     "lessThan0": (2, 1),
#     "equals0": (2, 1),
#     "greaterThan0": (2, 1),
#     "nRange1": (3, 1),
#     "nEquals16": (3, 1),
#     "nRange2": (3, 1),
#     "formula": (12, 1),
#     "result": (5, 1)
# }
#
# for key, pos in positions.items():
#     draw_box(ax, pos, *box_sizes[key], nodes[key], fontsize=10)
#
# # Draw arrows
# arrows = [
#     ("root", "lessThan0"),
#     ("root", "equals0"),
#     ("root", "greaterThan0"),
#     ("lessThan0", "nRange1"),
#     ("equals0", "nEquals16"),
#     ("greaterThan0", "nRange2"),
#     ("formula", "result")
# ]
#
# for start, end in arrows:
#     draw_arrow(ax, (positions[start][0] + box_sizes[start][0] / 2, positions[start][1]),
#                (positions[end][0] + box_sizes[end][0] / 2, positions[end][1] + box_sizes[end][1]))
#
# # Add LM labels
# ax.text(-4, 2, "$LM$", fontsize=10, ha='center', va='center')
# ax.text(0, 2, "$LM$", fontsize=10, ha='center', va='center')
# ax.text(4, 2, "$LM$", fontsize=10, ha='center', va='center')
#
# plt.show()


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

def draw_box(ax, xy, width, height, text, fontsize=10):
    """Draw a rectangle with text centered in it."""
    ax.add_patch(Rectangle(xy, width, height, edgecolor='black', facecolor='none', lw=1))
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, fontsize=fontsize,
            ha='center', va='center')

def draw_arrow(ax, start, end):
    """Draw an arrow from start to end."""
    ax.add_patch(FancyArrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                            width=0.02, head_width=0.15, head_length=0.3, color='black'))

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(-6, 6)
ax.set_ylim(-4, 6)
ax.axis('off')

# Node positions
positions = {
    "root": (0, 5),
    "lessThan0": (-5, 3),
    "equals0": (0, 3),
    "greaterThan0": (5, 3),
    "nRange1": (-5, 1),
    "nEquals16": (0, 1),
    "nRange2": (5, 1),
    "formula": (-5, -1),
    "result": (-1.7, -3)
}

# Draw boxes
nodes = {
    "root": "$r_i$",
    "lessThan0": "$<0$",
    "equals0": "$==0$",
    "greaterThan0": "$>0$",
    "nRange1": "$n: [19, 25]$",
    "nEquals16": "$n: 16$",
    "nRange2": "$n: [8, 14]$",
    "formula": r"$f_n = \sqrt{\frac{1}{4\pi^2} \cdot (n-1)(n+1)(n+2) \cdot \frac{\sigma}{\rho r_0^3}}$",
    "result": "$p_1 + p_2 + p_3$"
}

box_sizes = {
    "root": (2, 1),
    "lessThan0": (2, 1),
    "equals0": (2, 1),
    "greaterThan0": (2, 1),
    "nRange1": (3, 1),
    "nEquals16": (3, 1),
    "nRange2": (3, 1),
    "formula": (12, 1),
    "result": (5, 1)
}

for key, pos in positions.items():
    draw_box(ax, pos, *box_sizes[key], nodes[key], fontsize=10)

# Draw arrows
arrows = [
    ("root", "lessThan0"),
    ("root", "equals0"),
    ("root", "greaterThan0"),
    ("lessThan0", "nRange1"),
    ("equals0", "nEquals16"),
    ("greaterThan0", "nRange2"),
    ("formula", "result")
]

for start, end in arrows:
    draw_arrow(ax, (positions[start][0] + box_sizes[start][0] / 2, positions[start][1]),
               (positions[end][0] + box_sizes[end][0] / 2, positions[end][1] + box_sizes[end][1]))

# Add LM labels
ax.text(-4, 2, "$LM$", fontsize=10, ha='center', va='center')
ax.text(0, 2, "$LM$", fontsize=10, ha='center', va='center')
ax.text(4, 2, "$LM$", fontsize=10, ha='center', va='center')

plt.show()
