
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.io import wavfile
import os


input_folder = "Method_1_3_sins"
output_folder = "diagrams/Method_1"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for wav_file in os.listdir(input_folder):
    if wav_file.endswith(".wav"):  #  .wav
        wav_path = os.path.join(input_folder, wav_file)

        #  .wav
        sampling_rate, signal = wavfile.read(wav_path)

        if len(signal.shape) == 2:
            signal = signal.mean(axis=1)  #

        f, t, Zxx = stft(signal, fs=sampling_rate, nperseg=1024)


        plt.figure(figsize=(10, 6))
        plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap="Blues")
        plt.ylim(0, 1200)  #  0-4000 Hz
        plt.title("Frequency-Time", fontsize=16, fontweight='bold')
        plt.ylabel('Frequency [Hz]', fontsize=14, fontweight='bold')
        plt.xlabel('Time [sec]', fontsize=14, fontweight='bold')
        colorbar = plt.colorbar(label='Amplitude')


        colorbar.ax.set_ylabel('Amplitude', fontsize=14, fontweight='bold')

        colorbar.ax.tick_params(labelsize=12, width=2)
        for label in colorbar.ax.get_yticklabels():
            label.set_fontweight('bold')

        plt.tick_params(axis='both', labelsize=12, width=2)
        for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
            label.set_fontweight('bold')

        plt.gca().set_title("Frequency-Time", fontsize=16, fontweight='bold')
        plt.gca().set_xlabel('Time [sec]', fontsize=14, fontweight='bold')
        plt.gca().set_ylabel('Frequency [Hz]', fontsize=14, fontweight='bold')

        output_path = os.path.join(output_folder, f"{os.path.splitext(wav_file)[0]}_frequency_vs_time.png")
        plt.savefig(output_path, dpi=300)  #  300 DPI
        plt.close()  #

        print(f"Frequency vs Time plot saved to {output_path}")





