
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.io import wavfile
import os

# 定义输入和输出文件夹
input_folder = "Method_1_3_sins"
output_folder = "diagrams/Method_1"

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

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
        plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap="Blues")
        plt.ylim(0, 1200)  # 设置频率范围为 0-4000 Hz
        plt.title("Frequency-Time", fontsize=16, fontweight='bold')  # 设置标题加粗
        plt.ylabel('Frequency [Hz]', fontsize=14, fontweight='bold')  # 设置Y轴标签加粗
        plt.xlabel('Time [sec]', fontsize=14, fontweight='bold')  # 设置X轴标签加粗
        colorbar = plt.colorbar(label='Amplitude')  # 添加颜色条

        # 设置颜色条的标签加粗
        colorbar.ax.set_ylabel('Amplitude', fontsize=14, fontweight='bold')

        # 设置颜色条的刻度标签加粗
        colorbar.ax.tick_params(labelsize=12, width=2)  # 设置刻度标签大小和线宽
        for label in colorbar.ax.get_yticklabels():
            label.set_fontweight('bold')  # 设置颜色条刻度标签为粗体

        # 设置坐标轴刻度标签的大小、字体及粗细
        plt.tick_params(axis='both', labelsize=12, width=2)  # 设置刻度字体大小和刻度线宽度
        for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
            label.set_fontweight('bold')  # 设置坐标轴刻度标签为粗体字

        # 设置坐标轴标签和标题的字体加粗
        plt.gca().set_title("Frequency-Time", fontsize=16, fontweight='bold')
        plt.gca().set_xlabel('Time [sec]', fontsize=14, fontweight='bold')
        plt.gca().set_ylabel('Frequency [Hz]', fontsize=14, fontweight='bold')

        # 保存图像到 "diagrams" 文件夹中
        output_path = os.path.join(output_folder, f"{os.path.splitext(wav_file)[0]}_frequency_vs_time.png")
        plt.savefig(output_path, dpi=300)  # 设置分辨率为 300 DPI
        plt.close()  # 关闭图像窗口以释放内存

        print(f"Frequency vs Time plot saved to {output_path}")





