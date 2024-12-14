# Implementation ideas reference:
# 3. P. Dutilleux, G. De Poli, and U. Zölzer. “Time-segment Processing”.
# In: DAFX. John Wiley Sons, Ltd, 2002. Chap. 7, pp. 201–236. ISBN:
# 9780470858639

import numpy as np
import scipy.signal
import scipy.interpolate
from scipy.interpolate import CubicSpline
from scipy.io.wavfile import write
from scipy.signal import correlate

def cross_correlation(x_L1, x_L2):
    L = len(x_L1)  # Length of the overlap interval
    r = np.zeros(L)  # Initialize cross-correlation array

    # Compute cross-correlation for each m
    for m in range(L):
        r[m] = np.sum(x_L1[:L-m] * x_L2[m:L]) / L

    return r

def find_max_cc_index(r):
    # Find the index where the cross-correlation has its maximum value
    max_index = np.argmax(r)
    return max_index, r[max_index]

def resampling(freq_ratio, signal, sampling_rate):
    # get timeline according to signal and its sampling rate
    time_points = np.linspace(0, len(signal) / sampling_rate, len(signal), endpoint=False)

    # Increase/decrease the time size of the time point to 1/freq_ratio times of the original time point,
    # and the corresponding signal value remains unchanged
    new_time_points = time_points / freq_ratio

    # Generate a new timeline with the same sampling rate as the original signal
    new_duration = new_time_points[-1]
    new_length = int(new_duration * sampling_rate)  # the number of sampling points of the new timeline

    # resampling
    resampling_time_points = np.linspace(0, new_duration, new_length)

    spl = CubicSpline(new_time_points, signal)

    resampling_signal = spl(resampling_time_points)

    return resampling_signal

def SOLA(signal,freq_ratio,Sa=256,block_size=2048):
# Ss: hop size of synthesis window
# block_size: size of analysis block and synthesis block
# L: size of area for calculating cross correlation
# Sa: hop size of analysis window

#  calculate Ss
    Ss = Sa * freq_ratio

    L = int(256 * freq_ratio /2)

    M = np.ceil(len(signal)/Sa)

    Overlap = signal[:block_size]

    for i in range(1, int(M)):
        grain = signal[i * Sa - 1:i * Sa - 1 + block_size]
        # if len(grain) < L:
        #     corr = cross_correlation(grain[:], Overlap[i * Ss - 1: i * Ss - 1 + len(grain)])
        # if len(Overlap[i * Ss - 1: i * Ss - 1+L]) < L:
        #     corr = cross_correlation(grain[:len(Overlap[i * Ss - 1: i * Ss - 1+L])], Overlap[i * Ss - 1: i * Ss - 1 + len(grain)])
        # else:
        #     corr = cross_correlation(grain[:L], Overlap[i * Ss - 1: i * Ss - 1+L])

        min_len = min(len(grain), len(Overlap[i * Ss - 1: i * Ss - 1+L]))
        if min_len < L:
            corr = cross_correlation(grain[:min_len], Overlap[i * Ss - 1: i * Ss - 1 + min_len])
        else:
            corr = cross_correlation(grain[:L], Overlap[i * Ss - 1: i * Ss - 1+L])
        # corr = cross_correlation(grain[:L], Overlap[i * Ss - 1: i * Ss - 1 + L])
        if len(corr) == 0:
            break  # if grain's length is 0

        index, _ = find_max_cc_index(corr)

        len_fade = len(Overlap) - (i*Ss-1+index)

        start_pos = i * Ss -1 +index

        if len_fade > len(grain):
            len_fade = len(grain)

        fadeout = np.linspace(1, 0, len_fade)
        fadein = np.linspace(0, 1, len_fade)
        Tail = Overlap[start_pos: start_pos+len_fade] * fadeout

        Begin = grain[:len(fadein)] * fadein
        Add = Tail + Begin

        Overlap = np.concatenate((Overlap[:(i * Ss - 1 + index)], Add, grain[len(fadein):block_size]))

    return Overlap

freq = 500

duration = 10

sampling_rate = 48000

time = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
signal = np.sin(2 * np.pi * freq * time)

freq_ratio = 2

resampled_signal = resampling(freq_ratio, signal, sampling_rate)

# SOLA
SOLA_signal = SOLA(resampled_signal, freq_ratio)


write("doppler_results/original_signal.wav", sampling_rate, signal.astype(np.float32))


write("doppler_results/resampled_signal_single.wav", sampling_rate, resampled_signal.astype(np.float32))

write("doppler_results/sola_signal_single.wav", sampling_rate, SOLA_signal.astype(np.float32))

print(resampled_signal)
















































