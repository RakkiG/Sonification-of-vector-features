
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import os


output_folder = "Echo"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

sampling_rate0 = 48000  #
ori_duration = 2.92
t = np.linspace(0, ori_duration, int(sampling_rate0 * ori_duration), endpoint=False)

freq1 = 600

signal0 = np.sin(2 * np.pi * freq1 * t)

# Specify the output path for saving the plot
output_path = 'Echo/frequency_time_plot.png'

sampling_rate, signal = sampling_rate0, signal0

if len(signal.shape) == 2:
    signal = signal.mean(axis=1)

f, t, Zxx = stft(signal, fs=sampling_rate, nperseg=1024)

plt.figure(figsize=(10, 6))
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='Blues')

plt.ylim(0, 4000)

plt.title('Original signal Frequency-Time', fontsize=16, fontweight='bold')
plt.xlabel('Time [sec]', fontsize=14, fontweight='bold')
plt.ylabel('Frequency [Hz]', fontsize=14, fontweight='bold')

plt.tick_params(axis='both', labelsize=12, width=2)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontweight('bold')

cbar = plt.colorbar()
cbar.ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=12, width=2)

output_path = os.path.join(output_folder, "frequency_vs_time_ori_echo.png")
plt.savefig(output_path, dpi=300)  #  300 DPI
plt.close()

print(f"Frequency vs Time plot saved to {output_path}")
