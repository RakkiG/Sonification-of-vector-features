import os
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from matplotlib import font_manager

# 文件夹路径
folder_path = 'Echo/signals'

# 遍历文件夹中的所有文件
wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

# 创建存储图的文件夹（如果不存在）
output_folder = 'Echo/plots'
os.makedirs(output_folder, exist_ok=True)

# 设置字体为粗体
font_properties = font_manager.FontProperties(weight='bold')

for wav_file in wav_files:
    # 获取文件完整路径
    file_path = os.path.join(folder_path, wav_file)

    # 读取音频文件
    sample_rate, signal = read(file_path)

    # 计算时间轴（单位：秒）
    time = [i / sample_rate for i in range(len(signal))]

    # 绘制振幅-时间图像
    plt.figure(figsize=(10, 4))
    plt.plot(time, signal, label='Amplitude')

    # 设置标题、坐标轴标签和图例为粗体
    # plt.title(f'Amplitude-Time Plot for {wav_file}', fontsize=16, fontweight='bold')
    plt.title(f'Amplitude-Time', fontweight='bold')

    plt.xlabel('Time (s)', fontsize=14, fontweight='bold')
    plt.ylabel('Amplitude', fontsize=14, fontweight='bold')

    # 设置坐标轴刻度标签为粗体
    plt.tick_params(axis='both', labelsize=12, width=2)  # 设置刻度标签的大小和刻度线的宽度
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontweight('bold')  # 设置刻度标签为粗体

    # 添加网格线和图例，使用prop来设置字体属性
    plt.grid(True)
    plt.legend(fontsize=12, prop=font_properties)  # 使用prop设置字体属性

    plt.tight_layout()

    # 保存图像到输出文件夹
    output_path = os.path.join(output_folder, f'{os.path.splitext(wav_file)[0]}.png')
    plt.savefig(output_path)
    plt.close()

    print(f'Processed and saved plot for {wav_file}')
