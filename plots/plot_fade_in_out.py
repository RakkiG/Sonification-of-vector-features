import matplotlib.pyplot as plt
import numpy as np
import os

from PIL import Image

# 创建一个新的图形
fig, ax = plt.subplots(figsize=(10, 14))  # 增大图形尺寸

# 定义函数和常数
N = 10  # 假设 N = 10F
Km = 3  # 假设 K_m = 3
x = np.linspace(0, 4.5, 500)  # 设置 x 的范围为 0 到 4.5，调整宽度为 4.5

shift_fade_out_down = 0.3
shift_fade_in_up = 0.4


# 线性衰减函数 (fade-out)，调整位置使其原点在 (5.5, -0.8)
fade_out = 1 - x / 10 - 0.8-shift_fade_out_down-0.25 # 将 fade-out 调整为原点 (5.5, -0.8)

# 线性增长函数 (fade-in)，调整位置使其原点在 (5.5, 0.5)
fade_in = x / 10 + 0.5+shift_fade_in_up*3  # 将 fade-in 调整为原点 (5.5, 0.5)

# 绘制矩形块1
rect1 = plt.Rectangle((0, -0.1), N, 0.15, linewidth=2, edgecolor='black', fill = False)
ax.add_patch(rect1)
ax.text(N / 2, -0.025, r"$x_1$", ha='center', va='center', fontsize=18, color='black',fontweight='bold')

# 标注块的长度 N
# ax.text(N / 2, 0.3, r"$N$", ha='center', va='center', fontsize=12, color='black',fontweight='bold')

# 绘制虚线从 N-K_m 到矩形的右端
# ax.plot([N - Km, N], [0, 0], 'k--', lw=1)

# 添加 S_s 的标注
ax.text(2.5, -0.2, r"$S_s$", ha='center', va='center', fontsize=16, color='black',fontweight='bold')

# 画一个从 S_s 开始的线段
# ax.plot([Km, Km + 1], [0, 0], color='black', lw=2)

# 在 S_s 左右两边添加小箭头和竖线
# 左侧箭头
ax.annotate('', xy=(0, -0.15), xytext=(0.5, -0.15),
            arrowprops=dict(facecolor='black', edgecolor='black', shrinkA=0, shrinkB=0, width=1.5, headwidth=6))
ax.plot([0, 0], [-0.1, -0.20], color='black', lw=1.5)  # 小竖线

# 右侧箭头
ax.annotate('', xy=(5, -0.15), xytext=(4.5, -0.15),
            arrowprops=dict(facecolor='black', edgecolor='black', shrinkA=0, shrinkB=0, width=1.5, headwidth=6))
ax.plot([5, 5], [-0.1, -0.20], color='black', lw=1)  # 小竖线

# mark L-related
# left short line
ax.plot([5, 5], [-0.1, -0.20], color='green', lw=1.5)  # 小竖线
# left arrow
ax.annotate('', xy=(5, -0.15), xytext=(5.7, -0.15),
            arrowprops=dict(facecolor='green', edgecolor='green', shrinkA=0, shrinkB=0, width=1.5, headwidth=6))

ax.plot([7, 7], [-0.1, -0.20], color='green', lw=1.5)  # 小竖线
# left arrow
ax.annotate('', xy=(7, -0.15), xytext=(6.3, -0.15),
            arrowprops=dict(facecolor='green', edgecolor='green', shrinkA=0, shrinkB=0, width=1.5, headwidth=6))
ax.text(6, -0.20, r"$L$", ha='center', va='center', fontsize=16, color='green',fontweight='bold')

# 绘制第二个矩形块
rect2 = plt.Rectangle((5.5, -0.6), N, 0.15, linewidth=2, edgecolor='black',fill = False)
ax.add_patch(rect2)
ax.text(11, -0.53, r"$x_2$", ha='center', va='center', fontsize=18, color='black',fontweight='bold')

# 标注块的长度 N
ax.plot([0, 0], [0.05, 0.25], color='black', lw=1.5)  # 小竖线
ax.plot([10, 10], [0.05, 0.25], color='black', lw=1.5)  # 小竖线
# arrows
ax.annotate('', xy=(0, 0.15), xytext=(4.5, 0.15),
            arrowprops=dict(facecolor='black', edgecolor='black', shrinkA=0, shrinkB=0, width=1.5, headwidth=6))
ax.annotate('', xy=(10, 0.15), xytext=(5.5, 0.15),
            arrowprops=dict(facecolor='black', edgecolor='black', shrinkA=0, shrinkB=0, width=1.5, headwidth=6))
ax.text(5.25, 0.15, r"$N$", ha='center', va='center', fontsize=17, color='black',fontweight='bold')

# 标注块的长度 N---move downwards for down
down = 0.85
right = 5.5
# 标注块的长度 N
ax.plot([0 + right, 0 + right], [0.05 - down, 0.25 - down], color='black', lw=1.5)  # 小竖线
ax.plot([10 + right, 10 + right], [0.05 - down, 0.25 - down], color='black', lw=1.5)  # 小竖线
# arrows
ax.annotate('', xy=(0 + right, 0.15 - down), xytext=(4.5 + right, 0.15 - down),
            arrowprops=dict(facecolor='black', edgecolor='black', shrinkA=0, shrinkB=0, width=1.5, headwidth=6))
ax.annotate('', xy=(10 + right, 0.15 - down), xytext=(5.5 + right, 0.15 - down),
            arrowprops=dict(facecolor='black', edgecolor='black', shrinkA=0, shrinkB=0, width=1.5, headwidth=6))
ax.text(5.25 + right, 0.15 - down , r"$N$", ha='center', va='center', fontsize=17, color='black',fontweight='bold')

# 从 S_s 开始的线段
# ax.plot([Km, Km + 1], [0, 0], 'k-', lw=2)

# 两条虚线
ax.plot([5.5, 5.5], [-1.5, 0.7], 'k--', lw=1.5)
ax.plot([10, 10], [-1.8, 0.7], 'k--', lw=1.5)


# 绘制 fade-out 和 fade-in 的函数图像
ax.plot(x + 5.5, fade_out + 1.5-shift_fade_out_down, color='blue', label="Fade-out", lw=2)  # 将 x 轴平移到 (5.5, -0.8)
ax.plot(x + 5.5, fade_in - 3, color='red', label="Fade-in", lw=2)  # 将 x 轴平移到 (5.5, 0.5)

# 为 fade-out 函数添加坐标轴（竖直线）
ax.plot([5.5, 5.5], [0.5-shift_fade_out_down*3, 2-shift_fade_out_down*3], color='gray', linestyle='-', lw=1.5)  # fade-out 的竖线
ax.plot([5.3, 10.2], [0.7-shift_fade_out_down, 0.7-shift_fade_out_down], color='gray', linestyle='-', lw=1.5)  # fade-out 的横线
# ax.text(5.4, 0.6, r"$0$", ha='center', fontsize=15, color='black',fontweight='bold')
ax.text(5.6, 1.8-shift_fade_out_down*3 , r"$1$", ha='center', fontsize=15, color='black',fontweight='bold')

# 为 fade-in 函数添加坐标轴（竖直线）
ax.plot([5.5, 5.5], [-2.6+shift_fade_in_up*3, -1.2+shift_fade_in_up*3], color='gray', linestyle='-', lw=1.5)  # fade-out 的竖线
ax.plot([5.3, 10.2], [-2.5+shift_fade_in_up*3, -2.5+shift_fade_in_up*3], color='gray', linestyle='-', lw=1.5)
shift_1_down = 3.3
shift_1_right = 4.5  # fade-out 的横线
ax.text(5.6 + shift_1_right, 1.8 - shift_1_down+shift_fade_in_up*1.5, r"$1$", ha='center', fontsize=15, color='black',fontweight='bold')
# 添加坐标轴文本和标注
ax.text(6, 0.6-shift_fade_out_down , r"$K_m$", ha='center', fontsize=15, color='black')
ax.text(10.2, 0.6-shift_fade_out_down , r"$N$", ha='center', fontsize=15, color='black')
shift_k_m_down = 0.8
shift_k_m_left = 0.15
ax.text(5.8 - shift_k_m_left, 0.6 - shift_k_m_down + 0.25, r"$K_m$", ha='center', fontsize=15, color='black',fontweight='bold')

shift_n_k_m_down = 3.2
shift_n_k_m_right = 4.5
ax.text(5.8 + shift_n_k_m_right, 0.6 - shift_n_k_m_down+shift_fade_in_up*3, r"$N - K_m$", ha='center', fontsize=15, color='black',fontweight='bold')
ax.text(5.4, 0.6 - shift_n_k_m_down+shift_fade_in_up*3, r"$0$", ha='center', fontsize=15, color='black',fontweight='bold')

# 隐藏坐标轴刻度
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

# 调整图像布局，增大边距
fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.4, hspace=0.5)

# 保存图像到 diagrams 文件夹

# 去除最外围边框
for spine in ax.spines.values():
    spine.set_visible(False)
# 显示图像
plt.show()
save_path = os.path.join("../diagrams", "fade_in_out.png")
os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 创建文件夹（如果不存在的话）
fig.savefig(save_path)
print(f"图像已保存到 {save_path}")

with Image.open(save_path) as img:
    cropped = img.crop((89, 230, img.width - 89, img.height - 320) )  # 自定义裁剪范围
    cropped.save(save_path)


