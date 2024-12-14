

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import os

# 创建 diagrams 文件夹
output_dir = "../diagrams"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 创建 3D 网格
x = np.array([0, 1])
y = np.array([0, 1])
z = np.array([0, 1])
X, Y, Z = np.meshgrid(x, y, z)

# 定义顶点坐标
positions = np.array([[0, 0, 0],    # 0
                      [1, 0, 0],    # 1
                      [0, 1, 0],    # 2
                      [1, 1, 0],    # 3
                      [0, 0, 1],    # 4
                      [1, 0, 1],    # 5
                      [0, 1, 1],    # 6
                      [1, 1, 1]])   # 7

# 定义长方体的边线
edges = [
    [positions[0], positions[1]], [positions[0], positions[2]], [positions[0], positions[4]],
    [positions[1], positions[3]], [positions[1], positions[5]], [positions[2], positions[3]],
    [positions[2], positions[6]], [positions[3], positions[7]], [positions[4], positions[5]],
    [positions[4], positions[6]], [positions[5], positions[7]], [positions[6], positions[7]]
]

# 绘制3D图形
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制长方体的边线
for edge in edges:
    edge = np.array(edge)
    ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], color='black', linewidth=1)

# 在每个顶点上标注序号，并且稍微偏移位置，避免与边重合
labels_positions = [
    (positions[0], "7", (-0.06, 0.02, 0.0)),  # (0,0,0) 向x, y, z轴偏移0.02
    (positions[1], "6", (0.02, -0.02, -0.03)),  # (1,0,0) 向x, y轴偏移0.02, -0.02
    (positions[2], "4", (-0.02, 0.03, -0.13)),  # (0,1,0) 向x, z轴偏移-0.02, 0.02
    (positions[3], "5", (0.06, -0.02, 0.02)),  # (1,1,0) 向x, y轴偏移0.02, -0.02
    (positions[4], "3", (-0.06, 0.02, -0.02)), # (0,0,1) 向x, y轴偏移-0.02, 0.02
    (positions[5], "2", (0.01, -0.02, 0.1)), # (1,0,1) 向x, y轴偏移0.02, -0.02
    (positions[6], "0", (-0.02, 0.02, 0.03)), # (0,1,1) 向x, z轴偏移-0.02, -0.02
    (positions[7], "1", (0.02, 0.02, 0.03))  # (1,1,1) 向x, y轴偏移0.02, -0.02
]

# 标注序号并调整标签位置，同时加粗字体
for pos, label, offset in labels_positions:
    ax.text(pos[0] + offset[0],
            pos[1] + offset[1],
            pos[2] + offset[2],
            label, color='black', fontsize=20, ha='center', va='center', fontweight='bold')

# 定义蓝色直线连接的点对 (按照标签序号)
line_pairs = [
    (positions[0], positions[6]),  # (0, 6)
    (positions[1], positions[3]),  # (1, 3)
    (positions[1], positions[4]),  # (1, 4)
    (positions[1], positions[6]),  # (1, 6)
    (positions[3], positions[6])   # (3, 6)
]

# 绘制蓝色的直线连接点对
for pair in line_pairs:
    pair = np.array(pair)
    ax.plot(pair[:, 0], pair[:, 1], pair[:, 2], color='blue', linewidth=2)

# 设置坐标轴范围和隐藏刻度
ax.set_xlim([-0.05, 1.05])
ax.set_ylim([-0.05, 1.05])
ax.set_zlim([-0.05, 1.05])
ax.set_xticks([])  # 隐藏 X 轴刻度
ax.set_yticks([])  # 隐藏 Y 轴刻度
ax.set_zticks([])  # 隐藏 Z 轴刻度

# 设置图例和标题
ax.legend()

# 不显示网格
ax.grid(False)

# 保存图形到 diagrams 文件夹
output_file = os.path.join(output_dir, "3D_Critical_Point_Extraction_Labeled_Bold_Lines.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"图形已保存到 {output_file}")

# 显示图形
plt.show()
