
import os
import numpy as np
import matplotlib.pyplot as plt

# 参数定义
L = 30  # 每个块的长度
Ss=10  # 合成步长（分析块和自然延续块之间的时间间隔）

# 创建时间轴
signal_length = 200
time = np.arange(signal_length)
signal = np.sin(2 * np.pi * 0.05 * time)  # 创建一个示例正弦波信号

# 定义分析块（analysis block）的位置和自然延续块（natural continuation）的起始位置
analysis_start = 50
analysis_end = analysis_start + L
nat_start = analysis_start + Ss
nat_end = nat_start + L

# 创建 'diagrams' 文件夹（如果不存在的话）
os.makedirs("../diagrams", exist_ok=True)

# 绘制信号和方块
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制原始信号
ax.plot(time, signal, label="Original Signal", color='gray', alpha=0.7)

# 绘制分析块（Analysis Block）
ax.add_patch(plt.Rectangle((analysis_start, -1), L, 2, color='r', alpha=0.6, label="Previous Analysis Block"))

# 绘制自然延续块（Natural Continuation Block）
ax.add_patch(plt.Rectangle((nat_start, -1), L, 2, color='g', alpha=0.6, label="Natural Continuation block"))

# 标出合成步长（Synthesis Hop Size）
ax.annotate(f"Ss",
            xy=(50,1.2),
            xytext=(30,1.7),
            arrowprops=dict(facecolor='black', arrowstyle="->"),
            fontsize=12, color='blue')

# 绘制空心矩形框
rec = plt.Rectangle((50, -1.2), 10, 2.4, linewidth=4, edgecolor='yellow', facecolor='none')  # 使用 'none' 而不是 'transparent'
ax.add_patch(rec)

# 添加标签和图例
ax.set_title("Natural Continuation Block")
ax.set_xlabel("Time")
ax.set_ylabel("Signal Amplitude")
ax.legend(loc='upper right')

# 设置时间轴范围
ax.set_xlim(0, signal_length)
ax.set_ylim(-2, 2)

plt.tight_layout()

# 保存图像到 'diagrams' 文件夹
plt.savefig("diagrams/natural_continuation_block.png")

# 显示图像
plt.show()

