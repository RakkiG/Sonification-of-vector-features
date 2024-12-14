import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
import os

# Create the "drawa" folder if it doesn't exist
output_folder = "diagrams"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Generate a Hanning window
window_length = 1024
hanning_window = windows.hann(window_length)

# Time array for plotting
t = np.linspace(0, 1, window_length, endpoint=False)

# Plot the Hanning window with similar style as the given example
plt.figure(figsize=(10, 6))
plt.plot(t, hanning_window, label="Hanning Window", color="blue", alpha=0.8, linewidth=2)

# Enhancements for a cleaner, publication-quality plot
plt.legend(fontsize=12, loc="upper right")
plt.xlabel("Time (s)", fontsize=14)
plt.ylabel("Amplitude", fontsize=14)
plt.title("Hanning Window", fontsize=16)
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()

# Save the plot to the "drawa" folder
output_path = os.path.join(output_folder, "hanning_window_plot.png")
plt.savefig(output_path, dpi=300)  # Save with high resolution
plt.close()  # Close the plot to free memory

print(f"Hanning window plot saved to {output_path}")
