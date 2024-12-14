# --- Draw the frequency-time diagram of the original signal echo
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.io import wavfile
import os
import Tool_Functions as tf

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
# plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='Blues')

plt.ylim(0, 4000)  # 设置频率范围为 0-2000 Hz
plt.title('Original signal Frequency-Time')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Frequency')

# 保存图像到 "diagrams" 文件夹中
output_path = os.path.join(output_folder, "frequency_vs_time_ori_echo.png")
plt.savefig(output_path, dpi=300)  # 设置分辨率为 300 DPI
plt.close()  # 关闭图像窗口以释放内存

print(f"Frequency vs Time plot saved to {output_path}")






