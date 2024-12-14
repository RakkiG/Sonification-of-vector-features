# --- Draw the frequency-time diagram of the original signal echo
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.io import wavfile
import os
import Tool_Functions as tf


output_folder = "Echo"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


sampling_rate0 = 48000  #
ori_duration = 2.92
t = np.linspace(0, ori_duration, int(sampling_rate0 * ori_duration), endpoint=False)  #

freq1 = 600  #

signal0 = np.sin(2 * np.pi * freq1 * t)

# Specify the output path for saving the plot
output_path = 'Echo/frequency_time_plot.png'

sampling_rate, signal = sampling_rate0, signal0

if len(signal.shape) == 2:
    signal = signal.mean(axis=1)  #

f, t, Zxx = stft(signal, fs=sampling_rate, nperseg=1024)  #

plt.figure(figsize=(10, 6))
# plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='Blues')

plt.ylim(0, 4000)  #  0-2000 Hz
plt.title('Original signal Frequency-Time')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Frequency')

output_path = os.path.join(output_folder, "frequency_vs_time_ori_echo.png")
plt.savefig(output_path, dpi=300)  #  300 DPI
plt.close()

print(f"Frequency vs Time plot saved to {output_path}")






