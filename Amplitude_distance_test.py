import numpy as np
import math
import Amplitude_distance2
import matplotlib.pyplot as plt

# Assume that the amplitude_distance function has been imported

# Generate a test signal (sine wave) and normalize it
def generate_normalized_signal(frequency, duration, sample_rate):
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t)
    # Normalize to the range -0.5 to 0.5
    signal = 0.5 * signal / np.max(np.abs(signal))
    return t, signal

# Set simulated trajectories for the source and receiver
def generate_trajectories(num_points):
    source_trajectory = []
    receiver_trajectory = []
    # Set different distances at different time points
    for i in range(num_points):
        # Assume the source position is stationary, while the receiver position increases with time
        source_trajectory.append((0, 0, 0))  # Fixed source position
        receiver_trajectory.append((i * 0.1, 0, 0))  # Receiver position gradually increases
    return source_trajectory, receiver_trajectory

# Test the amplitude_distance function
def test_amplitude_distance():
    sample_rate = 48000  # Sampling rate
    duration = 2.92        # Signal duration in seconds
    frequency = 600      # Sine wave frequency in Hz
    time_interval = 0.04  # Time interval

    # Generate normalized test signal
    t, signal = generate_normalized_signal(frequency, duration, sample_rate)

    # Generate source and receiver trajectories
    num_points = math.ceil(duration/ time_interval)
    source_trajectory, receiver_trajectory = generate_trajectories(num_points)

    # Call the amplitude_distance function
    modified_signal = Amplitude_distance2.amplitude_distance(
        sample_rate=sample_rate,
        source_trajectory=source_trajectory,
        receiver_trajectory=receiver_trajectory,
        signal=signal,
        time_interval=time_interval,
        timeline = t
    )

    # Output the results
    print("Original signal amplitude range:", np.min(signal), "to", np.max(signal))
    print("Adjusted signal amplitude range:", np.min(modified_signal), "to", np.max(modified_signal))

    # Visualize the results with time on the x-axis
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal, label="Original Signal")
    plt.plot(t, modified_signal, label="Adjusted Signal", alpha=0.7)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Amplitude Adjustment Based on Distance")
    plt.show()

test_amplitude_distance()













