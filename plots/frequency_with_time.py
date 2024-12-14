
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import os

# 创建文件夹 "diagrams"（如果不存在）
output_folder = "Echo"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

sampling_rate0 = 48000  # 采样率
ori_duration = 2.92  # 信号持续时间，单位：秒
t = np.linspace(0, ori_duration, int(sampling_rate0 * ori_duration), endpoint=False)  # 时间数组

freq1 = 600  # 频率1

signal0 = np.sin(2 * np.pi * freq1 * t)

# Specify the output path for saving the plot
output_path = 'Echo/frequency_time_plot.png'

sampling_rate, signal = sampling_rate0, signal0

# 如果信号是立体声（双通道），转换为单通道
if len(signal.shape) == 2:
    signal = signal.mean(axis=1)  # 取两个通道的平均值

# 计算短时傅里叶变换 (STFT)
f, t, Zxx = stft(signal, fs=sampling_rate, nperseg=1024)  # 可以调整 nperseg 参数

# 绘制频谱图
plt.figure(figsize=(10, 6))
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='Blues')

# 设置频率范围
plt.ylim(0, 4000)  # 设置频率范围为 0-4000 Hz

# 设置标题和坐标轴标签为粗体
plt.title('Original signal Frequency-Time', fontsize=16, fontweight='bold')
plt.xlabel('Time [sec]', fontsize=14, fontweight='bold')
plt.ylabel('Frequency [Hz]', fontsize=14, fontweight='bold')

# 设置坐标轴刻度标签为粗体
plt.tick_params(axis='both', labelsize=12, width=2)  # 设置刻度标签的大小和刻度线宽度
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontweight('bold')  # 设置坐标轴刻度标签为粗体

# 添加颜色条并设置其标签样式
cbar = plt.colorbar()
cbar.ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')  # 设置颜色条标签
cbar.ax.tick_params(labelsize=12, width=2)  # 设置颜色条刻度标签大小和线宽

# 保存图像到 "diagrams" 文件夹中
output_path = os.path.join(output_folder, "frequency_vs_time_ori_echo.png")
plt.savefig(output_path, dpi=300)  # 设置分辨率为 300 DPI
plt.close()  # 关闭图像窗口以释放内存

print(f"Frequency vs Time plot saved to {output_path}")
