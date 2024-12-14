

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


positions = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [1, 1, 0],
                      [0, 0, 1],
                      [1, 0, 1],
                      [0, 1, 1],
                      [1, 1, 1]])

# 定义等值面函数
def u_function(x, y, z):
    return x - 0.5  # 示例：u = x - 0.5

def v_function(x, y, z):
    return y - 0.5  # 示例：v = y - 0.5

def w_function(x, y, z):
    return z - 0.5  # 示例：w = z - 0.5

# 计算顶点符号
values = []
for pos in positions:
    u_val = np.sign(u_function(*pos))  # 计算 u 的符号
    v_val = np.sign(v_function(*pos))  # 计算 v 的符号
    w_val = np.sign(w_function(*pos))  # 计算 w 的符号
    values.append((u_val, v_val, w_val))
values = np.array(values)

# 准备绘制等值面 u=0, v=0, w=0
u_plane = [[0.5, 0, 0], [0.5, 1, 0], [0.5, 1, 1], [0.5, 0, 1]]  # u=0 面
v_plane = [[0, 0.5, 0], [1, 0.5, 0], [1, 0.5, 1], [0, 0.5, 1]]  # v=0 面
w_plane = [[0, 0, 0.5], [1, 0, 0.5], [1, 1, 0.5], [0, 1, 0.5]]  # w=0 面

# 关键点 (u=0, v=0, w=0)
critical_point = [0.5, 0.5, 0.5]

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

# 绘制等值面
u_poly = Poly3DCollection([u_plane], alpha=0.3, color='blue', label='u=0 Surface')
v_poly = Poly3DCollection([v_plane], alpha=0.3, color='red', label='v=0 Surface')
w_poly = Poly3DCollection([w_plane], alpha=0.3, color='green', label='w=0 Surface')
ax.add_collection3d(u_poly)
ax.add_collection3d(v_poly)
ax.add_collection3d(w_poly)

# 绘制两两交线
u_v_line = [[0.5, 0.5, 0], [0.5, 0.5, 1]]  # u=0 与 v=0 的交线
u_w_line = [[0.5, 0, 0.5], [0.5, 1, 0.5]]  # u=0 与 w=0 的交线
v_w_line = [[0, 0.5, 0.5], [1, 0.5, 0.5]]  # v=0 与 w=0 的交线

ax.plot([p[0] for p in u_v_line], [p[1] for p in u_v_line], [p[2] for p in u_v_line],
        color='purple', linewidth=2, label='u=0 ∩ v=0')
ax.plot([p[0] for p in u_w_line], [p[1] for p in u_w_line], [p[2] for p in u_w_line],
        color='orange', linewidth=2, label='u=0 ∩ w=0')
ax.plot([p[0] for p in v_w_line], [p[1] for p in v_w_line], [p[2] for p in v_w_line],
        color='cyan', linewidth=2, label='v=0 ∩ w=0')

# 标注顶点符号 (+, -, 0)
for i, (x, y, z) in enumerate(positions):
    u_sign = '+' if values[i, 0] > 0 else '-' if values[i, 0] < 0 else '0'
    v_sign = '+' if values[i, 1] > 0 else '-' if values[i, 1] < 0 else '0'
    w_sign = '+' if values[i, 2] > 0 else '-' if values[i, 2] < 0 else '0'
    ax.text(x, y, z, f"({u_sign}, {v_sign}, {w_sign})", color="black", fontsize=21)

# 标记关键点
ax.scatter(*critical_point, color='purple', s=100, label="Critical Point")

# 设置坐标轴范围和隐藏刻度
ax.set_xlim([0, 1.3])
ax.set_ylim([0, 1.825])
ax.set_zlim([0, 1])
ax.set_xticks([])  # 隐藏 X 轴刻度
ax.set_yticks([])  # 隐藏 Y 轴刻度
ax.set_zticks([])  # 隐藏 Z 轴刻度

# 设置图例和标题
ax.legend(frameon=False, fontsize=14,bbox_to_anchor=(0.67, 0.65))  # 去掉边框并设置字体大小为 14

# 不显示网格
ax.grid(False)

# 保存图形到 diagrams 文件夹
output_file = os.path.join(output_dir, "3D_Critical_Point_Extraction_No_Ticks.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"图形已保存到 {output_file}")

# 显示图形
plt.show()