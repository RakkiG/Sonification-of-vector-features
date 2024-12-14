# # 画非线性变调，根据ana的窗口开始位置求出syn的窗口开始位置
import matplotlib.pyplot as plt

# 数据定义
n_ana = [0, 2, 4, 6, 8, 10]  # 分析块起始点
delay = [0, 0.5, 1, 1.5, 2, 2.5]  # 每个分析块的延迟
n_syn = [n + d for n, d in zip(n_ana, delay)]  # 合成块起始点
block_duration = 1.5  # 每个块的持续时间
overlap_duration = 0.5  # 重叠区域的持续时间

# 绘制图表
plt.figure(figsize=(12, 6))

# 分析块时间轴
for n in n_ana:
    plt.hlines(1, n, n + block_duration, colors='blue', linestyles='solid', label='Analysis Block' if n == 0 else "")
    plt.text(n + block_duration / 2, 1.1, f"A{n}", color='blue', ha='center', fontsize=9)

# 合成块时间轴
for n in n_syn:
    plt.hlines(0.5, n, n + block_duration, colors='green', linestyles='solid', label='Synthesis Block' if n == n_syn[0] else "")
    plt.text(n + block_duration / 2, 0.4, f"S{n:.1f}", color='green', ha='center', fontsize=9)

# 添加箭头表示延迟
for n_a, n_s in zip(n_ana, n_syn):
    plt.arrow(n_a + block_duration / 2, 1, n_s - n_a, -0.4, head_width=0.1, head_length=0.2, color='gray', alpha=0.7)
    plt.text((n_a + n_s) / 2, 0.7, f"Delay={n_s-n_a:.1f}", fontsize=8, color='gray', ha='center')

# 添加重叠区域标记
for n in n_syn:
    plt.fill_betweenx([0.5], n + block_duration - overlap_duration, n + block_duration, color='purple', alpha=0.3, label='Overlap Region' if n == n_syn[0] else "")

# 图例、标题和布局
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=10)
plt.title("Adaptive Pitch-Shifting: Analysis to Synthesis Blocks", fontsize=12, pad=20)
plt.xlabel("Time")
plt.yticks([1, 0.5], ["Analysis Blocks", "Synthesis Blocks"])
plt.ylim(0.2, 1.3)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# 隐藏左边框和顶部边框
plt.gca().spines["left"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)

# 显示图表
plt.tight_layout()
plt.show()
