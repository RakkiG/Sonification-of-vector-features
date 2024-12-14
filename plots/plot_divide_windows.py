# Re-import libraries and rewrite the modified code to set x-axis limits to x_max
import os
import numpy as np
import matplotlib.pyplot as plt

# Create the diagrams directory if it doesn't exist
output_dir = "../diagrams"
os.makedirs(output_dir, exist_ok=True)

# Define the signal and window function
L = 100  # Length of block
signal_length = 500  # Length of the input signal

# Create an example input signal (random for illustration)
X = np.random.randn(signal_length)

# Create a simple window function (e.g., Hann window)
w = np.hanning(L)

# Create empty arrays for accumulated signals and overlap
Y = np.zeros(signal_length)
O = np.zeros(signal_length)

# List to store windowed signals and corresponding windows for visualization
windowed_signals = []
windows = []

# Simulate the overlap-add process
colors = plt.cm.viridis(np.linspace(0, 1, (signal_length - L) // (L // 2)))
for idx, s_i in enumerate(range(0, signal_length - L, L // 2)):  # Use 50% overlap
    X_ana_block = X[s_i : s_i + L]
    windowed_signal = X_ana_block * w
    Y[s_i : s_i + L] += windowed_signal  # Accumulate windowed signal
    O[s_i : s_i + L] += w  # Accumulate window function
    windowed_signals.append((s_i, windowed_signal))  # Store for visualization
    windows.append((s_i, w * 0.5))  # Scale for better visualization

x_max = windows[-1][0] + L - 3

# Normalize the output signal
O[O < 1e-3] = 1  # Prevent division by zero
Y_normalized = Y / O  # Normalize the accumulated signal

# Plot the original signal, windowed signals, and the updated graphs
plt.figure(figsize=(12, 12))

# Define x-axis range for all plots
x_range = range(signal_length)

# Original signal (for comparison)
plt.subplot(5, 1, 1)
plt.plot(x_range, X, label='Original Signal', color='blue')
plt.title('Original Signal')
plt.legend()
plt.xlim(0, x_max)  # Set x-axis to x_max

# Individual windows
plt.subplot(5, 1, 2)
for idx, (s_i, win) in enumerate(windows):
    plt.plot(range(s_i, s_i + L), win, alpha=0.6, label=f'Window at {s_i}', color=colors[idx])
plt.title('Window Functions')
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.xlim(0, x_max)  # Set x-axis to x_max

# Individual windowed signals
plt.subplot(5, 1, 3)
for idx, (s_i, win_signal) in enumerate(windowed_signals):
    plt.plot(range(s_i, s_i + L), win_signal, alpha=0.6, label=f'Windowed Signal at {s_i}', color=colors[idx])
plt.title('Windowed Signals')
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.xlim(0, x_max)  # Set x-axis to x_max

# Accumulated Window (New Fourth Plot)
plt.subplot(5, 1, 4)
plt.plot(x_range, O, label='Accumulated Window', color='red', alpha=0.5)
plt.title('Accumulated Window')
plt.legend()
plt.xlim(0, x_max)  # Set x-axis to x_max

# Normalized Signal (New Fifth Plot)
plt.subplot(5, 1, 5)
plt.plot(x_range, Y_normalized, label='Normalized Signal', color='green')
plt.title('Normalized Signal')
plt.legend()
plt.xlim(0, x_max)  # Set x-axis to x_max

plt.tight_layout()

# Save the plot to the diagrams folder
output_path = os.path.join(output_dir, "overlap_add_visualization.png")
plt.savefig(output_path)
print(f"Figure saved to {output_path}")

plt.show()
