

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import os

output_dir = "../diagrams"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

x = np.array([0, 1])
y = np.array([0, 1])
z = np.array([0, 1])
X, Y, Z = np.meshgrid(x, y, z)


positions = np.array([[0, 0, 0],    # 0
                      [1, 0, 0],    # 1
                      [0, 1, 0],    # 2
                      [1, 1, 0],    # 3
                      [0, 0, 1],    # 4
                      [1, 0, 1],    # 5
                      [0, 1, 1],    # 6
                      [1, 1, 1]])   # 7


edges = [
    [positions[0], positions[1]], [positions[0], positions[2]], [positions[0], positions[4]],
    [positions[1], positions[3]], [positions[1], positions[5]], [positions[2], positions[3]],
    [positions[2], positions[6]], [positions[3], positions[7]], [positions[4], positions[5]],
    [positions[4], positions[6]], [positions[5], positions[7]], [positions[6], positions[7]]
]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')


for edge in edges:
    edge = np.array(edge)
    ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], color='black', linewidth=1)

labels_positions = [
    (positions[0], "7", (-0.06, 0.02, 0.0)),
    (positions[1], "6", (0.02, -0.02, -0.03)),
    (positions[2], "4", (-0.02, 0.03, -0.13)),
    (positions[3], "5", (0.06, -0.02, 0.02)),  #
    (positions[4], "3", (-0.06, 0.02, -0.02)),
    (positions[5], "2", (0.01, -0.02, 0.1)),
    (positions[6], "0", (-0.02, 0.02, 0.03)),
    (positions[7], "1", (0.02, 0.02, 0.03))
]

for pos, label, offset in labels_positions:
    ax.text(pos[0] + offset[0],
            pos[1] + offset[1],
            pos[2] + offset[2],
            label, color='black', fontsize=20, ha='center', va='center', fontweight='bold')

line_pairs = [
    (positions[0], positions[6]),  # (0, 6)
    (positions[1], positions[3]),  # (1, 3)
    (positions[1], positions[4]),  # (1, 4)
    (positions[1], positions[6]),  # (1, 6)
    (positions[3], positions[6])   # (3, 6)
]

for pair in line_pairs:
    pair = np.array(pair)
    ax.plot(pair[:, 0], pair[:, 1], pair[:, 2], color='blue', linewidth=2)

ax.set_xlim([-0.05, 1.05])
ax.set_ylim([-0.05, 1.05])
ax.set_zlim([-0.05, 1.05])
ax.set_xticks([])  #  X
ax.set_yticks([])  #  Y
ax.set_zticks([])  #  Z

ax.legend()

ax.grid(False)

output_file = os.path.join(output_dir, "3D_Critical_Point_Extraction_Labeled_Bold_Lines.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f" {output_file}")


plt.show()
