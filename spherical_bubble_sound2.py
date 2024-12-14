import math
import os
import numpy as np
from scipy.io.wavfile import write

# tunable parameter that determines the initial excitation of the bubbles
epsilon = 0.072
surface_tension = 0.0728
RHO_WATER = 998.
ksi = 0.1
gamma = 1.4
g_th = 1.6e6

PATM = 101325.
G = 9.81
# Assume the bubble is in 1m depth in water
h = 2.
p0 = PATM + RHO_WATER * G * h

speed_sound_water = 1500.  # the speed of sound in water

def f0(r0):
    return 3 / r0

def delta_th(f0):
    return math.sqrt((9 * (gamma - 1)**2) / (4 * g_th) * f0)

delta_rad = math.sqrt((3 * gamma * p0) / (RHO_WATER * speed_sound_water**2))

def delta_tol(f0):
    return delta_th(f0) + delta_rad

def beta_0(r0):
    return math.pi * f0(r0) * delta_tol(f0(r0))

# def f_t(r0, t):
#     return f0(r0) * (1 + ksi * beta_0(r0) * t)
def f_t(r0, beta0, t):
    return f0(r0) * (1 + ksi * beta0 * t)

# def spherical_p_t(r0, t):
#     return epsilon * r0 * np.sin(2 * math.pi * f_t(r0, t) * t) * np.exp(-beta_0(r0) * t)

def spherical_p_t(r0,beta0, t):
    return epsilon * r0 * np.sin(2 * math.pi * f_t(r0,beta0, t) * t) * np.exp(-beta0 * t)

def find_last_nonzero(signal):
    """Find the index of the last non-zero element in a signal"""
    nonzero_indices = np.nonzero(signal)[0]
    if len(nonzero_indices) == 0:
        return None
    return nonzero_indices[-1]

def generate_and_repeat_bubble_sound(r0, beta0, fs, duration, total_sound_duration):
    # Generate the initial bubble sound
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sound_signal = spherical_p_t(r0, beta0, t)

    # Normalize the sound signal to the range of int16
    sound_signal = sound_signal / np.max(np.abs(sound_signal))
    sound_signal *= 32767
    sound_signal = sound_signal.astype(np.int16)

    # Find the last non-zero index of the first sound signal
    last_nonzero_index = find_last_nonzero(sound_signal)
    # Trim the initial sound signal by removing trailing zeros
    trimmed_signal = sound_signal[:last_nonzero_index + 1]

    trimmed_signal_length = len(trimmed_signal)

    silence_length = 1*fs
    silence = np.zeros(silence_length, dtype=np.int16)

    single_sound_duration = trimmed_signal_length / fs  # 单个信号的时长（秒）

    repeat_times = math.ceil(np.ceil(total_sound_duration/ (single_sound_duration + silence_length / fs)))

    # Start the repeated bubble sound with the trimmed signal
    repeated_signal = np.concatenate((trimmed_signal, silence))

    # Repeat the signal for the given number of times
    for _ in range(repeat_times - 1):
        # Trim the repeated signal again to remove trailing zeros before concatenation
        # Concatenate the next sound signal starting from the last non-zero element
        repeated_signal = np.concatenate((repeated_signal,trimmed_signal, silence))

    total_samples = int(total_sound_duration * fs)

    if len(repeated_signal) > total_samples:
        repeated_signal = repeated_signal[:total_samples]
    repeated_signal = repeated_signal / np.max(np.abs(repeated_signal))

    return repeated_signal

def generate_spherical_bubble_sound(r0, beta0, fs, duration):
    # Generate the initial bubble sound
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sound_signal = spherical_p_t(r0, beta0, t)
    # Normalize the sound signal to the range of int16
    sound_signal = sound_signal / np.max(np.abs(sound_signal))

    return sound_signal

# Parameters
r0 = 0.004
fs = 48000
duration = 0.8
repeat_times = 3  # Change this value to set the number of repetitions
total_sound_duration = 10

# Create the "spherical bubble sounds" folder if it doesn't exist
output_dir = "bubble sounds"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

beta0 = beta_0(r0)

# Generate and repeat the bubble sound
repeated_sound_signal = generate_and_repeat_bubble_sound(r0, beta0, fs, duration, total_sound_duration)

# Save the repeated sound signal to a file
output_path = os.path.join(output_dir, f'sound_signal_{r0}_repeated.wav')
write(output_path, fs, repeated_sound_signal)

print(f"Repeated sound signal saved to {output_path}")


























