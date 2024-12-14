

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import os
from matplotlib import font_manager

def pad_signals_to_same_length(signal1, signal2):

    max_length = max(len(signal1), len(signal2))

    # Pad signal1
    if len(signal1) < max_length:
        padding = np.zeros(max_length - len(signal1))
        signal1 = np.concatenate((signal1, padding))

    # Pad signal2
    if len(signal2) < max_length:
        padding = np.zeros(max_length - len(signal2))
        signal2 = np.concatenate((signal2, padding))

    return signal1, signal2


def plot_amplitude_time(signal, sample_rate, save_plot=False, output_path="amplitude_time_plot.png"):

    # If the audio is stereo (2 channels), average it to mono
    if len(signal.shape) == 2:
        signal = signal.mean(axis=1)

    # Generate the time array
    time = np.linspace(0, len(signal) / sample_rate, num=len(signal))

    # Plot the amplitude-time graph
    plt.figure(figsize=(10, 5))
    plt.plot(time, signal, label="Amplitude")

    # Set title, labels, and legend to bold using FontProperties
    font_properties = font_manager.FontProperties(weight='bold')

    plt.title("Amplitude-Time", fontsize=16, fontproperties=font_properties)
    plt.xlabel("Time (s)", fontsize=14, fontproperties=font_properties)
    plt.ylabel("Amplitude", fontsize=14, fontproperties=font_properties)

    # Using the font_properties to set the legend font
    plt.legend(fontsize=12, prop=font_properties)
    plt.grid(True)

    # Set axis ticks to bold
    plt.tick_params(axis='both', labelsize=12, width=2)  # Set tick label size and width
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontweight('bold')  # Set tick labels to bold

    # Save plot to Echo/diagrams folder
    if save_plot:
        output_folder = "Echo/diagrams"
        os.makedirs(output_folder, exist_ok=True)
        plt.savefig(os.path.join(output_folder, output_path), dpi=300)  # Save with 300 DPI
        print(f"Plot saved as {os.path.join(output_folder, output_path)}")

    plt.show()



wav_file_path1 = "/vector_field_process6/Echo/signals/recursion: 0, received_signal.wav"
wav_file_path2 = "/vector_field_process6/Echo/signals/recursion: 0, received_signal_initial.wav"

# Read the .wav files
sample_rate1, signal1 = read(wav_file_path1)
sample_rate2, signal2 = read(wav_file_path2)

# Ensure both signals are mono
if len(signal1.shape) == 2:
    signal1 = signal1.mean(axis=1)
if len(signal2.shape) == 2:
    signal2 = signal2.mean(axis=1)

# Pad both signals to the same length
signal1, signal2 = pad_signals_to_same_length(signal1, signal2)

# Compute the difference
signal_pure = signal2 - signal1

plot_amplitude_time(signal_pure, sample_rate1, save_plot=True, output_path="amplitude_time_difference_plot.png")
