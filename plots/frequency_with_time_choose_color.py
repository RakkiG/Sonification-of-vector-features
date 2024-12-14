# current top 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.io import wavfile
import os
import Tool_Functions as tf

# Create the "diagrams" folder if it doesn't exist
output_folder = "diagrams"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read the .wav file signal
wav_file = "/vector_field_process7/vortex_signals/vortex_signals/vortex_ori.wav"
sampling_rate, signal = wavfile.read(wav_file)

# If the signal is stereo (2 channels), convert to mono by averaging
if len(signal.shape) == 2:
    signal = signal.mean(axis=1)

# Calculate the Short-Time Fourier Transform (STFT)
f, t, Zxx = stft(signal, fs=sampling_rate, nperseg=1024)  # You can adjust the nperseg parameter

# Plot the spectrogram with custom colormap
plt.figure(figsize=(10, 6))

# We use a colormap where the lowest amplitude is white, and high amplitudes are represented in blue
# Using 'Blues' colormap which starts from white and gets darker blue as the value increases
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='Blues')

# Set the frequency range (e.g., 0 to 4000 Hz)
plt.ylim(0, 4000)

# Add title and labels
plt.title('Frequency vs Time')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

# Add a colorbar with label
plt.colorbar(label='Amplitude')

# Save the plot to the "diagrams" folder
output_path = os.path.join(output_folder, "frequency_vs_time.png")
plt.savefig(output_path, dpi=300)  # Set the resolution to 300 DPI
plt.close()  # Close the figure to free memory

print(f"Frequency vs Time plot saved to {output_path}")


