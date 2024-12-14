import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.io import wavfile
import os
from matplotlib import font_manager

#
input_folder = "Echo/signals"
output_folder = "Echo/diagrams"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

font = font_manager.FontProperties(weight='bold')

for wav_file in os.listdir(input_folder):
    if wav_file.endswith(".wav"):  #  .wav
        wav_path = os.path.join(input_folder, wav_file)

        sampling_rate, signal = wavfile.read(wav_path)

        if len(signal.shape) == 2:
            signal = signal.mean(axis=1)  #

        f, t, Zxx = stft(signal, fs=sampling_rate, nperseg=1024)  #

        plt.figure(figsize=(10, 6))

        im = plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='Blues')

        plt.ylim(0, 2000)

        plt.title(f'Frequency-Time', fontproperties=font)
        plt.ylabel('Frequency [Hz]', fontproperties=font)
        plt.xlabel('Time [sec]', fontproperties=font)


        plt.tick_params(axis='both', labelsize=10, width=2)
        for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
            label.set_fontweight('bold')

        cbar = plt.colorbar(im, label='Amplitude')
        cbar.ax.tick_params(labelsize=10, width=2)
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight('bold')
        cbar.set_label('Amplitude', fontsize=12, fontweight='bold')

        output_path = os.path.join(output_folder, f"{os.path.splitext(wav_file)[0]}_frequency_vs_time.png")
        plt.savefig(output_path, dpi=300)  #  300 DPI
        plt.close()

        print(f"Frequency vs Time plot saved to {output_path}")
