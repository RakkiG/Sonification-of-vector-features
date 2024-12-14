

import numpy as np
import matplotlib.pyplot as plt
import os

fig, ax = plt.subplots(figsize=(6, 6))

x = np.array([0, 0])
y = np.array([0, 4])
ax.plot(x, y, color='black', linewidth=2)

divide_length = 1
segment_length = 0.4
num_segments = 5


for i in range(num_segments):
    start_y = i * divide_length
    ax.plot([-0.2, 0.2], [start_y, start_y], color='blue', linewidth=3)


for i in range(0, num_segments - 1):
    mid_y = (i * divide_length + (i + 1) * divide_length) / 2  # 计算中点
    ax.plot(0, mid_y, 'ro', markersize=6)  # 绘制红色小球


ax.set_xlim(-0.4, 0.4)
ax.set_ylim(0, 4)
ax.set_aspect('equal', adjustable='box')


ax.set_xticks([])
ax.set_yticks([])


ax.plot([-0.2, -0.1], [1, 1], color='green', linewidth=3)
ax.plot([-0.2, -0.1], [2, 2], color='green', linewidth=3)
ax.annotate('', xy=(-0.15, 2), xytext=(-0.15, 1.7),
            arrowprops=dict(facecolor='green', edgecolor='green', shrinkA=0, shrinkB=0, width=1, headwidth=6))
ax.annotate('', xy=(-0.15, 1), xytext=(-0.15, 1.3),
            arrowprops=dict(facecolor='green', edgecolor='green', shrinkA=0, shrinkB=0, width=1, headwidth=6))
ax.text(-0.15, 1.5, r"$l$", ha='center', va='center', fontsize=25, color='green')

# shift length marker
shift_height = 2
ax.plot([-0.2, -0.1], [1 + shift_height, 1 + shift_height], color='green', linewidth=3)
ax.plot([-0.2, -0.1], [2 + shift_height, 2 + shift_height], color='green', linewidth=3)
ax.annotate('', xy=(-0.15, 2 + shift_height), xytext=(-0.15, 1.7 + shift_height),
            arrowprops=dict(facecolor='green', edgecolor='green', shrinkA=0, shrinkB=0, width=1, headwidth=6))
ax.annotate('', xy=(-0.15, 1 + shift_height), xytext=(-0.15, 1.3 + shift_height),
            arrowprops=dict(facecolor='green', edgecolor='green', shrinkA=0, shrinkB=0, width=1, headwidth=6))
ax.text(-0.15, 1.5 + shift_height, r"$a$", ha='center', va='center', fontsize=25, color='green')


for spine in ax.spines.values():
    spine.set_visible(False)


output_dir = "../diagrams"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


output_file = os.path.join(output_dir, "line_with_segments_no_border.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0)


plt.show()


print(f"save to {output_file}")
