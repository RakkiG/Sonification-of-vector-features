import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.io import wavfile
import os
import Tool_Functions as tf

# 创建文件夹 "diagrams"（如果不存在）
output_folder = "diagrams"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取 .wav 文件中的信号
wav_file = "/vector_field_process7/vortex_signals/vortex_signals/vortex_ori.wav"
sampling_rate, signal = wavfile.read(wav_file)

# 如果信号是立体声（双通道），转换为单通道
if len(signal.shape) == 2:
    signal = signal.mean(axis=1)  # 取两个通道的平均值

# 计算短时傅里叶变换 (STFT)
f, t, Zxx = stft(signal, fs=sampling_rate, nperseg=1024)  # 可以调整 nperseg 参数

# 绘制频谱图
plt.figure(figsize=(10, 6))
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='Blues')

# 设置频率范围为 0-4000 Hz
plt.ylim(0, 3000)

# 添加标题和标签，设置为粗体
plt.title('Frequency vs Time', fontsize=16, fontweight='bold')
plt.ylabel('Frequency [Hz]', fontsize=14, fontweight='bold')
plt.xlabel('Time [sec]', fontsize=14, fontweight='bold')

# 设置坐标轴刻度标签为粗体
plt.tick_params(axis='both', labelsize=12, width=2)  # 设置刻度标签的大小和刻度线的宽度
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontweight('bold')  # 设置刻度标签为粗体

# 添加颜色条，并设置颜色条标签为粗体
cbar = plt.colorbar(label='Amplitude')
cbar.ax.tick_params(labelsize=12, width=2)  # 设置颜色条刻度标签的大小和线宽
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')  # 设置颜色条刻度标签为粗体
cbar.set_label('Amplitude', fontsize=14, fontweight='bold')  # 设置颜色条标签为粗体

# 保存图像到 "diagrams" 文件夹中
output_path = os.path.join(output_folder, "frequency_vs_time.png")
plt.savefig(output_path, dpi=300)  # 设置分辨率为 300 DPI
plt.close()  # 关闭图像窗口以释放内存

print(f"Frequency vs Time plot saved to {output_path}")
