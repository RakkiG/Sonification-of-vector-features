import numpy as np
import matplotlib.pyplot as plt

# 创建一个简单的正弦信号作为原始信号
signal_length = 500
t = np.linspace(0, 2 * np.pi, signal_length)
original_signal = np.sin(t)

# 模拟分析块和合成块
L = 100  # 分析块的长度
overlap = 50  # 重叠的长度
synthesis_signal = np.zeros(signal_length)

# 模拟延迟分析块并进行叠加
delayed_signal = np.zeros(signal_length)
for i in range(0, signal_length - L, overlap):
    analysis_block = original_signal[i:i + L]  # 获取分析块
    delayed_signal[i + overlap:i + L + overlap] = analysis_block  # 延迟分析块
    synthesis_signal[i:i + L] += analysis_block  # 直接叠加到合成信号中

# 绘制图形
plt.figure(figsize=(12, 8))

# 原始信号
plt.subplot(3, 1, 1)
plt.plot(t, original_signal, label='Original Signal', color='blue')
plt.title('Original Signal')
plt.legend()

# 延迟分析块
plt.subplot(3, 1, 2)
plt.plot(t, delayed_signal, label='Delayed Analysis Block', color='orange')
plt.title('Delayed Analysis Block')
plt.legend()

# 合成信号
plt.subplot(3, 1, 3)
plt.plot(t, synthesis_signal, label='Synthesis Signal', color='green')
plt.title('Synthesis Signal (with Overlap-Add)')
plt.legend()

plt.tight_layout()
plt.show()
