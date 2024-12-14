# --- implementation of frequency shifting
# Implementation ideas reference:
# 1. https://github.com/meinardmueller/libtsm/blob/master/libtsm/pitchshift.py
# 2. https://github.com/meinardmueller/libtsm/blob/master/libtsm/tsm.py
# 3. P. Dutilleux, G. De Poli, and U. Zölzer. “Time-segment Processing”.
# In: DAFX. John Wiley Sons, Ltd, 2002. Chap. 7, pp. 201–236. ISBN:
# 9780470858639

import math

import numpy as np
import scipy.signal
import scipy.interpolate
from scipy.interpolate import CubicSpline
from scipy.io.wavfile import write
from scipy.signal import correlate
import matplotlib.pyplot as plt

def win(win_len, beta) -> np.ndarray:
    # use hanning window
    # w = scipy.signal.hann(win_len) ** beta
    w = scipy.signal.windows.hann(win_len)
    return w

def cross_corr(x, y, win_len) -> np.ndarray:
    # refer to https://github.com/meinardmueller/libtsm/blob/master/libtsm/utils.py#L477   cross_corr
    cc = np.convolve(x, np.flip(y))
    # restrict the cross correlation result to just the relevant values
    # Values outside of this range are related to deltas bigger or smaller than our tolerance values.
    cc = cc[win_len-1:-(win_len-1)]
    return cc


def resampling(freq_ratios, pitch_changing_times, signal, sampling_rate):
    # get timeline according to signal and its sampling rate
    time_points = np.linspace(0, len(signal) / sampling_rate, len(signal), endpoint=False)

    new_time_points_marks = [pitch_changing_times[0]]
    # calculate new time points
    for i in range(0, len(pitch_changing_times) - 1):
        diff = pitch_changing_times[i + 1] - pitch_changing_times[i]
        new_time_points_marks.append(new_time_points_marks[i] + diff / freq_ratios[i])

    # Interpolate to generate a new timeline
    new_time_points = np.interp(time_points, pitch_changing_times, new_time_points_marks)

    # Generate a new timeline with the same sampling rate as the original signal
    new_duration = new_time_points[-1]
    new_length = int(new_duration * sampling_rate)  # the number of sampling points of the new timeline

    # resampling
    resampling_time_points = np.linspace(0, new_duration, new_length)

    spl = CubicSpline(new_time_points, signal)

    resampling_signal = spl(resampling_time_points)

    return resampling_signal, time_points, new_time_points

def SOLA(signal, original_time_points, new_time_points, sampling_rate, Ss=512,block_size=1024):
# Ss: hop size of synthesis window
# block_size: size of analysis block and synthesis block
# L: size of area for calculating cross correlation
# Sa: hop size of analysis window

#  calculate Ss
    # createanalysis windows
    w = win(block_size, 2)
    extension_ana = 512
    signal_longer = np.concatenate((np.zeros(extension_ana),signal, np.zeros(block_size+extension_ana)))
    synthesis_blocks_begins = []
    pos = 0
    while pos <= original_time_points[-1]*sampling_rate:
        synthesis_blocks_begins.append(pos)
        pos += Ss
    synthesis_blocks_begins = np.array(synthesis_blocks_begins)

    ori_sample_indexes = original_time_points * sampling_rate

    new_sample_indexes = new_time_points * sampling_rate

    # get synthesis blocks begins（beginning positions）
    # Need to return the stretched or shrunk time to the previous time,
    # need to interpolate.ana_begins, new_time_points, old_time_points)
    analysis_block_begins = np.interp(synthesis_blocks_begins, ori_sample_indexes, new_sample_indexes)

    analysis_block_begins = analysis_block_begins.astype(int)
    analysis_block_begins += extension_ana

    # current_sig = signal_longer[:block_size]*w
    y_c = np.zeros(int(original_time_points[-1]*sampling_rate+block_size))
    ow = np.zeros_like(y_c)
    y_c[:block_size] += signal_longer[analysis_block_begins[0]:analysis_block_begins[0]+block_size]*w
    # right_long =
    ow[:block_size] += w
    delays = np.zeros_like(analysis_block_begins)

# Start from the second ana block and select the s that is most similar to the s starting from the begin position
# of the previous ana block plus the hop from the previous s to this s
    for i in range(1,len(analysis_block_begins)):
        # overlapping_L = current_sig[synthesis_blocks_begins[i]:]

        ana_choosing_range_begin = analysis_block_begins[i]-extension_ana
        ana_choosing_range_end = analysis_block_begins[i]+block_size + extension_ana

        current_syn_begin = synthesis_blocks_begins[i]
        current_syn_end = current_syn_begin + block_size

        ana_choosing_range =signal_longer[ana_choosing_range_begin:ana_choosing_range_end]

        natural_indices_begin = analysis_block_begins[i - 1]+delays[i-1] + Ss
        nat_prog = signal_longer[natural_indices_begin:natural_indices_begin + block_size]

        # Find the starting point of the block that is closest to ana_choosing_range and overlapping_L,
        # plus blocksize is ana_block
        cc = cross_corr(ana_choosing_range, nat_prog, block_size)  # compute the cross correlation
        max_index = np.argmax(cc)  # pick the optimizing index in the cross correlation

        # delay = extension_ana - max_index
        delay = max_index - extension_ana
        delays[i] = delay

        # add delay，get analysis block
        ana_block = signal_longer[analysis_block_begins[i]+delay:analysis_block_begins[i]+block_size+delay]

        # current_sig = np.concatenate((current_sig[:synthesis_blocks_begins[i]],add,ana_block[len(fadein):block_size]))
        y_c[current_syn_begin:current_syn_end] += ana_block*w
        ow[current_syn_begin:current_syn_end] += w

    # re-normalize the signal by dividing by the added windows
    ow[ow < 10 ** (-3)] = 1  # avoid potential division by zero
    y_c /= ow

    y_c = y_c[:math.ceil(ori_sample_indexes[-1]+1)]

    return y_c

#---Test

sampling_rate = 48000
duration = 10
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
signal = np.sin(2 * np.pi * 500 * t)


freq_ratios = [0.4, 2, 1.3, 0.5, 2,2, 2,1]
pitch_changing_times = [0,1,2,3, 4,6, 8, t[-1]]

# call resampling
resampled_signal, original_time_points, new_time_points = resampling(freq_ratios, pitch_changing_times, signal, sampling_rate)


original_signal_path = "original_signal.wav"
resampled_signal_path = "doppler_results/resampled_signal_tvps.wav"


# write(original_signal_path, sampling_rate, signal)
# #
#
# write(resampled_signal_path, sampling_rate, resampled_signal)
# #
#
# SOLA_signal = SOLA(resampled_signal, original_time_points, new_time_points,sampling_rate)
# #
#
# write("doppler_results/sola_signal_tvps.wav", sampling_rate, SOLA_signal.astype(np.float32))

















































