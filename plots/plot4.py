import numpy as np
import matplotlib.pyplot as plt

# Define parameters
b_min = 0.1
b_max = 5.0
new_min = 0.5
new_max = 3.0
b_values = np.linspace(b_min, b_max, 100)
mapped_speed = new_min + (b_values - b_min) * (new_max - new_min) / (b_max - b_min)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(b_values, mapped_speed, label="Mapped Speed")
plt.title("Mapping Imaginary Part to Playback Speed")
plt.xlabel("|b| (Absolute Imaginary Part)")
plt.ylabel("Playback Speed")
plt.grid(True)
plt.legend()
plt.show()


from scipy.interpolate import CubicSpline

# Original signal
t_original = np.linspace(0, 10, 50)  # Original time index
signal = np.sin(2 * np.pi * 0.2 * t_original)  # Example sine wave

# Resampling
t_new = np.linspace(0, 10, 100)  # New time index
spline = CubicSpline(t_original, signal)
resampled_signal = spline(t_new)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(t_original, signal, 'o', label="Original Points")
plt.plot(t_new, resampled_signal, '-', label="Resampled Signal (Cubic Spline)")
plt.title("Cubic Spline Resampling")
plt.xlabel("Time Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()
