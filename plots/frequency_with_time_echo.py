import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.io import wavfile
import os
from matplotlib import font_manager

# 定义输入和输出文件夹
input_folder = "Echo/signals"
output_folder = "Echo/diagrams"

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取系统字体并设置粗体
font = font_manager.FontProperties(weight='bold')

# 遍历 "vortex signals" 文件夹中的所有 .wav 文件
for wav_file in os.listdir(input_folder):
    if wav_file.endswith(".wav"):  # 检查是否是 .wav 文件
        wav_path = os.path.join(input_folder, wav_file)

        # 读取 .wav 文件
        sampling_rate, signal = wavfile.read(wav_path)

        # 如果信号是立体声（双通道），转换为单通道
        if len(signal.shape) == 2:
            signal = signal.mean(axis=1)  # 取两个通道的平均值

        # 计算短时傅里叶变换 (STFT)
        f, t, Zxx = stft(signal, fs=sampling_rate, nperseg=1024)  # 可以调整 nperseg 参数

        # 绘制频谱图
        plt.figure(figsize=(10, 6))

        # 绘制 STFT 的幅度图，使用pcolormesh
        im = plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='Blues')

        # 设置频率范围为 0-2000 Hz
        plt.ylim(0, 2000)

        # 添加标题和标签
        plt.title(f'Frequency-Time', fontproperties=font)
        plt.ylabel('Frequency [Hz]', fontproperties=font)
        plt.xlabel('Time [sec]', fontproperties=font)

        # 设置坐标轴刻度标签为粗体
        plt.tick_params(axis='both', labelsize=10, width=2)  # 设置刻度标签的大小和刻度线的宽度
        for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
            label.set_fontweight('bold')  # 设置字体粗细

        # 添加颜色条并设置粗体标签
        cbar = plt.colorbar(im, label='Amplitude')
        cbar.ax.tick_params(labelsize=10, width=2)  # 设置颜色条刻度标签的大小和宽度
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight('bold')  # 设置颜色条刻度标签为粗体
        cbar.set_label('Amplitude', fontsize=12, fontweight='bold')  # 设置颜色条标签为粗体

        # 保存图像到 "diagrams" 文件夹中
        output_path = os.path.join(output_folder, f"{os.path.splitext(wav_file)[0]}_frequency_vs_time.png")
        plt.savefig(output_path, dpi=300)  # 设置分辨率为 300 DPI
        plt.close()  # 关闭图像窗口以释放内存

        print(f"Frequency vs Time plot saved to {output_path}")
