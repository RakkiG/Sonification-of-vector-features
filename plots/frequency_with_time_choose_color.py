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



#
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import stft
# from scipy.io import wavfile
# import os
# import Tool_Functions as tf
# import matplotlib.cm as cm
# from matplotlib.colors import LinearSegmentedColormap
#
# # Create the "diagrams" folder if it doesn't exist
# output_folder = "diagrams"
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# # Read the .wav file signal
# wav_file = "/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/vector_field_process5/vortex_ori.wav"
# sampling_rate, signal = wavfile.read(wav_file)
#
# # If the signal is stereo (2 channels), convert to mono by averaging
# if len(signal.shape) == 2:
#     signal = signal.mean(axis=1)
#
# # Calculate the Short-Time Fourier Transform (STFT)
# f, t, Zxx = stft(signal, fs=sampling_rate, nperseg=1024)  # You can adjust the nperseg parameter
#
# # Create a custom colormap that goes from white (0 amplitude) to a fluorescent blue
# colors = [(1, 1, 1), (0, 0, 1)]  # White to blue (R, G, B)
# n_bins = 100  # Number of bins (colors)
# cmap_name = "fluorescent_blue"
# cm_fluorescent_blue = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
#
# # Plot the spectrogram with custom colormap
# plt.figure(figsize=(10, 6))
#
# # Apply the custom colormap
# plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap=cm_fluorescent_blue)
#
# # Set the frequency range (e.g., 0 to 4000 Hz)
# plt.ylim(0, 4000)
#
# # Add title and labels
# plt.title('Frequency vs Time')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
#
# # Add a colorbar with label
# plt.colorbar(label='Amplitude')
#
# # Save the plot to the "diagrams" folder
# output_path = os.path.join(output_folder, "frequency_vs_time_fluorescent_blue.png")
# plt.savefig(output_path, dpi=300)  # Set the resolution to 300 DPI
# plt.close()  # Close the figure to free memory
#
# print(f"Frequency vs Time plot saved to {output_path}")






# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import stft
# from scipy.io import wavfile
# import os
# import Tool_Functions as tf
# import matplotlib.cm as cm
# from matplotlib.colors import LinearSegmentedColormap
#
# # Create the "diagrams" folder if it doesn't exist
# output_folder = "diagrams"
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# # Read the .wav file signal
# wav_file = "/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/vector_field_process5/vortex_ori.wav"
# sampling_rate, signal = wavfile.read(wav_file)
#
# # If the signal is stereo (2 channels), convert to mono by averaging
# if len(signal.shape) == 2:
#     signal = signal.mean(axis=1)
#
# # Calculate the Short-Time Fourier Transform (STFT)
# f, t, Zxx = stft(signal, fs=sampling_rate, nperseg=1024)  # You can adjust the nperseg parameter
#
# # Create a custom colormap with a deeper and more intense blue
# colors = [(1, 1, 1), (0, 0, 0.8), (0, 0, 1)]  # White to deep blue to fluorescent blue
# n_bins = 100  # Number of bins (colors)
# cmap_name = "deep_fluorescent_blue"
# cm_deep_fluorescent_blue = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
#
# # Plot the spectrogram with custom colormap
# plt.figure(figsize=(10, 6))
#
# # Apply the custom colormap
# plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap=cm_deep_fluorescent_blue)
#
# # Set the frequency range (e.g., 0 to 4000 Hz)
# plt.ylim(0, 4000)
#
# # Add title and labels
# plt.title('Frequency vs Time')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
#
# # Add a colorbar with label
# plt.colorbar(label='Amplitude')
#
# # Save the plot to the "diagrams" folder
# output_path = os.path.join(output_folder, "frequency_vs_time_deep_fluorescent_blue.png")
# plt.savefig(output_path, dpi=300)  # Set the resolution to 300 DPI
# plt.close()  # Close the figure to free memory
#
# print(f"Frequency vs Time plot saved to {output_path}")





# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import stft
# from scipy.io import wavfile
# import os
# import Tool_Functions as tf
# import matplotlib.cm as cm
# from matplotlib.colors import LinearSegmentedColormap
#
# # Create the "diagrams" folder if it doesn't exist
# output_folder = "diagrams"
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# # Read the .wav file signal
# wav_file = "/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/vector_field_process5/vortex_ori.wav"
# sampling_rate, signal = wavfile.read(wav_file)
#
# # If the signal is stereo (2 channels), convert to mono by averaging
# if len(signal.shape) == 2:
#     signal = signal.mean(axis=1)
#
# # Calculate the Short-Time Fourier Transform (STFT)
# f, t, Zxx = stft(signal, fs=sampling_rate, nperseg=1024)  # You can adjust the nperseg parameter
#
# # Create a custom colormap with a cyan/greenish-blue tone
# colors = [(1, 1, 1), (0, 0.8, 1), (0, 1, 1)]  # White to deep cyan-blue to fluorescent cyan
# n_bins = 100  # Number of bins (colors)
# cmap_name = "deep_cyan_fluorescent_blue"
# cm_deep_cyan_fluorescent_blue = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
#
# # Plot the spectrogram with custom colormap
# plt.figure(figsize=(10, 6))
#
# # Apply the custom colormap
# plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap=cm_deep_cyan_fluorescent_blue)
#
# # Set the frequency range (e.g., 0 to 4000 Hz)
# plt.ylim(0, 4000)
#
# # Add title and labels
# plt.title('Frequency vs Time')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
#
# # Add a colorbar with label
# plt.colorbar(label='Amplitude')
#
# # Save the plot to the "diagrams" folder
# output_path = os.path.join(output_folder, "frequency_vs_time_deep_cyan_fluorescent_blue.png")
# plt.savefig(output_path, dpi=300)  # Set the resolution to 300 DPI
# plt.close()  # Close the figure to free memory
#
# print(f"Frequency vs Time plot saved to {output_path}")






# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import stft
# from scipy.io import wavfile
# import os
# import Tool_Functions as tf
# import matplotlib.cm as cm
# from matplotlib.colors import LinearSegmentedColormap
#
# # Create the "diagrams" folder if it doesn't exist
# output_folder = "diagrams"
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# # Read the .wav file signal
# wav_file = "/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/vector_field_process5/vortex_ori.wav"
# sampling_rate, signal = wavfile.read(wav_file)
#
# # If the signal is stereo (2 channels), convert to mono by averaging
# if len(signal.shape) == 2:
#     signal = signal.mean(axis=1)
#
# # Calculate the Short-Time Fourier Transform (STFT)
# f, t, Zxx = stft(signal, fs=sampling_rate, nperseg=1024)  # You can adjust the nperseg parameter
#
# # Create a custom colormap with a deep and vibrant blue tone
# colors = [(1, 1, 1), (0, 0, 0.8), (0, 0, 1)]  # White to deep blue to vibrant blue
# n_bins = 100  # Number of bins (colors)
# cmap_name = "vibrant_blue"
# cm_vibrant_blue = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
#
# # Plot the spectrogram with custom colormap
# plt.figure(figsize=(10, 6))
#
# # Apply the custom colormap
# plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap=cm_vibrant_blue)
#
# # Set the frequency range (e.g., 0 to 4000 Hz)
# plt.ylim(0, 4000)
#
# # Add title and labels
# plt.title('Frequency vs Time')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
#
# # Add a colorbar with label
# plt.colorbar(label='Amplitude')
#
# # Save the plot to the "diagrams" folder
# output_path = os.path.join(output_folder, "frequency_vs_time_vibrant_blue.png")
# plt.savefig(output_path, dpi=300)  # Set the resolution to 300 DPI
# plt.close()  # Close the figure to free memory
#
# print(f"Frequency vs Time plot saved to {output_path}")





# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import stft
# from scipy.io import wavfile
# import os
# import Tool_Functions as tf
# import matplotlib.cm as cm
# from matplotlib.colors import LinearSegmentedColormap
#
# # Create the "diagrams" folder if it doesn't exist
# output_folder = "diagrams"
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# # Read the .wav file signal
# wav_file = "/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/vector_field_process5/vortex_ori.wav"
# sampling_rate, signal = wavfile.read(wav_file)
#
# # If the signal is stereo (2 channels), convert to mono by averaging
# if len(signal.shape) == 2:
#     signal = signal.mean(axis=1)
#
# # Calculate the Short-Time Fourier Transform (STFT)
# f, t, Zxx = stft(signal, fs=sampling_rate, nperseg=1024)  # You can adjust the nperseg parameter
#
# # Modify the "Blues" colormap to make the blue more intense
# # We will make a custom colormap based on 'Blues' but boost the blues
# colors = [(1, 1, 1), (0.6, 0.8, 1), (0, 0, 1)]  # From white to a stronger blue
# n_bins = 100  # Number of bins (colors)
# cmap_name = "intense_blue"
# cm_intense_blue = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
#
# # Plot the spectrogram with custom colormap
# plt.figure(figsize=(10, 6))
#
# # Apply the custom colormap to make the blue stronger
# plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap=cm_intense_blue)
#
# # Set the frequency range (e.g., 0 to 4000 Hz)
# plt.ylim(0, 4000)
#
# # Add title and labels
# plt.title('Frequency vs Time')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
#
# # Add a colorbar with label
# plt.colorbar(label='Amplitude')
#
# # Save the plot to the "diagrams" folder
# output_path = os.path.join(output_folder, "frequency_vs_time_intense_blue.png")
# plt.savefig(output_path, dpi=300)  # Set the resolution to 300 DPI
# plt.close()  # Close the figure to free memory
#
# print(f"Frequency vs Time plot saved to {output_path}")





# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import stft
# from scipy.io import wavfile
# import os
# import Tool_Functions as tf
# import matplotlib.cm as cm
# from matplotlib.colors import LinearSegmentedColormap
#
# # Create the "diagrams" folder if it doesn't exist
# output_folder = "diagrams"
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# # Read the .wav file signal
# wav_file = "/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/vector_field_process5/vortex_ori.wav"
# sampling_rate, signal = wavfile.read(wav_file)
#
# # If the signal is stereo (2 channels), convert to mono by averaging
# if len(signal.shape) == 2:
#     signal = signal.mean(axis=1)
#
# # Calculate the Short-Time Fourier Transform (STFT)
# f, t, Zxx = stft(signal, fs=sampling_rate, nperseg=1024)  # You can adjust the nperseg parameter
#
# # Modify the "Blues" colormap to make the blue more intense and increase contrast
# # We will create a custom colormap with sharper contrast and deeper blue tones
# colors = [(1, 1, 1), (0.4, 0.6, 1), (0, 0, 1)]  # From white to more intense blue
# n_bins = 100  # Number of bins (colors)
# cmap_name = "high_contrast_blue"
# cm_high_contrast_blue = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
#
# # Plot the spectrogram with custom colormap
# plt.figure(figsize=(10, 6))
#
# # Apply the custom colormap to make the blue more intense with higher contrast
# plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap=cm_high_contrast_blue)
#
# # Set the frequency range (e.g., 0 to 4000 Hz)
# plt.ylim(0, 4000)
#
# # Add title and labels
# plt.title('Frequency vs Time')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
#
# # Add a colorbar with label
# plt.colorbar(label='Amplitude')
#
# # Save the plot to the "diagrams" folder
# output_path = os.path.join(output_folder, "frequency_vs_time_high_contrast_blue.png")
# plt.savefig(output_path, dpi=300)  # Set the resolution to 300 DPI
# plt.close()  # Close the figure to free memory
#
# print(f"Frequency vs Time plot saved to {output_path}")







# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import stft
# from scipy.io import wavfile
# import os
# import matplotlib.colors as mcolors
#
# # Create the "diagrams" folder if it doesn't exist
# output_folder = "diagrams"
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# # Read the .wav file signal
# wav_file = "/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/vector_field_process5/vortex_ori.wav"
# sampling_rate, signal = wavfile.read(wav_file)
#
# # If the signal is stereo (2 channels), convert to mono by averaging
# if len(signal.shape) == 2:
#     signal = signal.mean(axis=1)
#
# # Calculate the Short-Time Fourier Transform (STFT)
# f, t, Zxx = stft(signal, fs=sampling_rate, nperseg=1024)  # You can adjust the nperseg parameter
#
# # Modify the Blues colormap to enhance brightness and contrast
# cmap = plt.cm.Blues
#
# # Create a new colormap by stretching the brightness
# new_cmap = mcolors.LinearSegmentedColormap.from_list(
#     "bright_blues",
#     [(0, 'white'), (0.5, 'lightblue'), (1, 'mediumblue')]  # Adjusted to make brighter blues
# )
#
# # Plot the spectrogram with custom colormap
# plt.figure(figsize=(10, 6))
#
# # Use the adjusted colormap
# plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap=new_cmap)
#
# # Set the frequency range (e.g., 0 to 4000 Hz)
# plt.ylim(0, 4000)
#
# # Add title and labels
# plt.title('Frequency vs Time')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
#
# # Add a colorbar with label
# plt.colorbar(label='Amplitude')
#
# # Save the plot to the "diagrams" folder
# output_path = os.path.join(output_folder, "frequency_vs_time.png")
# plt.savefig(output_path, dpi=300)  # Set the resolution to 300 DPI
# plt.close()  # Close the figure to free memory
#
# print(f"Frequency vs Time plot saved to {output_path}")







#
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import stft
# from scipy.io import wavfile
# import os
# import matplotlib.colors as mcolors
#
# # Create the "diagrams" folder if it doesn't exist
# output_folder = "diagrams"
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# # Read the .wav file signal
# wav_file = "/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/vector_field_process5/vortex_ori.wav"
# sampling_rate, signal = wavfile.read(wav_file)
#
# # If the signal is stereo (2 channels), convert to mono by averaging
# if len(signal.shape) == 2:
#     signal = signal.mean(axis=1)
#
# # Calculate the Short-Time Fourier Transform (STFT)
# f, t, Zxx = stft(signal, fs=sampling_rate, nperseg=1024)  # You can adjust the nperseg parameter
#
# # Define a custom colormap that transitions from off-white to blue
# cmap = mcolors.LinearSegmentedColormap.from_list(
#     "white_to_blue",
#     [(0, "w"),  # White at zero amplitude
#      (0.3, "lightgray"),  # Light gray for low amplitude values
#      (0.6, "skyblue"),  # Transition to light blue
#      (1, "blue")]  # Blue for high amplitude values
# )
#
# # Plot the spectrogram with the custom colormap
# plt.figure(figsize=(10, 6))
# plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap=cmap)
#
# # Set the frequency range (e.g., 0 to 4000 Hz)
# plt.ylim(0, 4000)
#
# # Add title and labels
# plt.title('Frequency vs Time')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')

# Add a colorbar with label
# plt.colorbar(label='Amplitude')
#
# # Save the plot to the "diagrams" folder
# output_path = os.path.join(output_folder, "frequency_vs_time.png")
# plt.savefig(output_path, dpi=300)  # Set the resolution to 300 DPI
# plt.close()  # Close the figure to free memory
#
# print(f"Frequency vs Time plot saved to {output_path}")









# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import stft
# from scipy.io import wavfile
# import os
# import matplotlib.colors as mcolors
#
# # Create the "diagrams" folder if it doesn't exist
# output_folder = "diagrams"
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# # Read the .wav file signal
# wav_file = "/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/vector_field_process5/vortex_ori.wav"
# sampling_rate, signal = wavfile.read(wav_file)
#
# # If the signal is stereo (2 channels), convert to mono by averaging
# if len(signal.shape) == 2:
#     signal = signal.mean(axis=1)
#
# # Calculate the Short-Time Fourier Transform (STFT)
# f, t, Zxx = stft(signal, fs=sampling_rate, nperseg=1024)  # You can adjust the nperseg parameter
#
# # Define a custom colormap that transitions from very pale yellow to blue
# cmap = mcolors.LinearSegmentedColormap.from_list(
#     "pale_yellow_to_blue",
#     [(0, "ffffe0"),  # Very pale yellow (super light yellow) at zero amplitude
#      (0.3, "lightgray"),  # Light gray for low amplitude values
#      (0.6, "skyblue"),  # Transition to light blue
#      (1, "blue")]  # Blue for high amplitude values
# )
#
# # Plot the spectrogram with the custom colormap
# plt.figure(figsize=(10, 6))
# plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap=cmap)
#
# # Set the frequency range (e.g., 0 to 4000 Hz)
# plt.ylim(0, 4000)
#
# # Add title and labels
# plt.title('Frequency vs Time')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
#
# # Add a colorbar with label
# plt.colorbar(label='Amplitude')
#
# # Save the plot to the "diagrams" folder
# output_path = os.path.join(output_folder, "frequency_vs_time.png")
# plt.savefig(output_path, dpi=300)  # Set the resolution to 300 DPI
# plt.close()  # Close the figure to free memory
#
# print(f"Frequency vs Time plot saved to {output_path}")
