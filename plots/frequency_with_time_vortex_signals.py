

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy.signal import stft
from scipy.io import wavfile
import os

# Define input and output folders
input_folder = "vortex_signals"
output_folder = "diagrams/vortex"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through all .wav files in the input folder
for wav_file in os.listdir(input_folder):
    if wav_file.endswith(".wav"):  # Check for .wav files
        wav_path = os.path.join(input_folder, wav_file)

        # Read the .wav file
        sampling_rate, signal = wavfile.read(wav_path)

        # Convert stereo signal to mono if needed
        if len(signal.shape) == 2:
            signal = signal.mean(axis=1)  # Take the mean of two channels

        # Compute Short-Time Fourier Transform (STFT)
        f, t, Zxx = stft(signal, fs=sampling_rate, nperseg=1024)  # Adjustable nperseg

        # Plot the spectrogram
        plt.figure(figsize=(10, 6))


        im = plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='Blues', norm=PowerNorm(gamma=6))
        im.set_clim(vmin=0, vmax=np.max(np.abs(Zxx)) * 0.5)  # ，
        plt.ylim(0, 2000)  # Set frequency range to 0-2000 Hz

        # Apply bold font to all text elements
        plt.title('Frequency-Time', fontweight='bold')
        plt.ylabel('Frequency [Hz]', fontweight='bold')
        plt.xlabel('Time [sec]', fontweight='bold')

        # Customize colorbar with bold labels and ticks
        cbar = plt.colorbar(label='Amplitude', aspect=10)
        cbar.set_label('Amplitude', fontweight='bold')
        cbar.ax.tick_params(labelsize=10, width=2, labelrotation=0, labelcolor='k', length=5)
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight('bold')

        # Set axis tick labels and ticks to bold
        plt.tick_params(axis='both', which='major', labelsize=10, width=2, length=6, labelcolor='k')
        ax = plt.gca()
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight('bold')  # Set tick labels to bold

        # Save the plot to the "diagrams" folder
        output_path = os.path.join(output_folder, f"{os.path.splitext(wav_file)[0]}_frequency_vs_time.png")
        plt.savefig(output_path, dpi=300)  # Save at 300 DPI resolution
        plt.close()  # Close the figure to free memory

        print(f"Frequency vs Time plot saved to {output_path}")
