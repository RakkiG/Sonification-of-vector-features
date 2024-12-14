import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.io import wavfile
import os
import Tool_Functions as tf

output_folder = "diagrams"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

wav_file = "/vector_field_process7/vortex_signals/vortex_signals/vortex_ori.wav"
sampling_rate, signal = wavfile.read(wav_file)

if len(signal.shape) == 2:
    signal = signal.mean(axis=1)

f, t, Zxx = stft(signal, fs=sampling_rate, nperseg=1024)  #  nperseg

plt.figure(figsize=(10, 6))
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='Blues')

plt.ylim(0, 3000)

plt.title('Frequency vs Time', fontsize=16, fontweight='bold')
plt.ylabel('Frequency [Hz]', fontsize=14, fontweight='bold')
plt.xlabel('Time [sec]', fontsize=14, fontweight='bold')


plt.tick_params(axis='both', labelsize=12, width=2)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontweight('bold')

cbar = plt.colorbar(label='Amplitude')
cbar.ax.tick_params(labelsize=12, width=2)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
cbar.set_label('Amplitude', fontsize=14, fontweight='bold')

output_path = os.path.join(output_folder, "frequency_vs_time.png")
plt.savefig(output_path, dpi=300)  #  300 DPI
plt.close()  #

print(f"Frequency vs Time plot saved to {output_path}")
