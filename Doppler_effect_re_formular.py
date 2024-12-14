# --- Doppler effect implementation
import math
import os

import numpy as np
import matplotlib
matplotlib.use('MacOSX')
import re_doppler
import doppler_mul_freq3 as doppler_mul
import csv

def Doppler_effect_re_formular(sample_rate,timeline,source_trajectory,receiver_trajectory,signal,time_interval):
    csv_file = 'Echo/Trajectories/doppler_ratio_values.csv'
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Start Time', 'Distance', 'Ratio'])  #

    original_signal = signal
    sample_rate = sample_rate

    p = np.zeros_like(timeline)
    ratios_for_freqs = np.zeros_like(timeline)
    ratios_for_freqs2 = np.zeros_like(timeline)
    save_end_index = 0

    # if not os.path.exists('doppler_ratio_values.csv'):
    #     open('doppler_ratio_values.csv', 'w').close()

    current_time = 0
    for i in range(len(source_trajectory)-1):
        #Get the start and end points of the current time interval
        source_position_start = source_trajectory[i]
        source_position_end = source_trajectory[i+1]

        receiver_position_start = np.array(receiver_trajectory[i])*2
        receiver_position_end = np.array(receiver_trajectory[i+1])*2

        source_position_start = np.array(source_position_start)
        source_position_end = np.array(source_position_end)
        receiver_position_start = np.array(receiver_position_start)
        receiver_position_end = np.array(receiver_position_end)

        delta_t = time_interval
        delta_x_source = source_position_end - source_position_start
        u = delta_x_source / delta_t
        delta_x_receiver = (receiver_position_end - receiver_position_start)
        v = delta_x_receiver / delta_t
        delta_v = u - v
        x = receiver_position_start - source_position_start
        magnitude_delta_v = np.linalg.norm(delta_v)

        magnitude_delta_v = magnitude_delta_v
        print("magnitude_delta_v: ",magnitude_delta_v)
        magnitude_x = np.linalg.norm(x)
        dot_product = np.dot(delta_v, x)
        #The angle between two vectors
        # Skip the current loop if magnitude_delta_v or magnitude_x is 0
        if magnitude_delta_v == 0 or magnitude_x == 0:
            continue

        cos_theta = dot_product / (magnitude_delta_v * magnitude_x)
        print("cos_theta: ",cos_theta)

        ratio_for_freq =1+magnitude_delta_v*cos_theta/340
        # print("ratio_for_freq: ",ratio_for_freq)

        start_index = math.ceil(i*time_interval*sample_rate)
        end_index = math.ceil((i+1)*time_interval*sample_rate)

        if (source_position_start is None or source_position_end is None
                or receiver_position_start is None or receiver_position_end is None):
            original_signal = original_signal.astype(np.float64)  # 转换为float64类型
            original_signal[start_index:end_index] *= 0.001
            original_signal = original_signal.astype(np.int16)  # 如果需要，转换回int16类型
        else:
            shift_cent = round(12 * math.log2(ratio_for_freq)*100)
            p[start_index:end_index] = shift_cent

            # ratios_for_freqs[start_index:end_index] = ratio_for_freq
            # if ratio_for_freq <=1:
            #     ratios_for_freqs[start_index:end_index] = 0.6 * ratio_for_freq
            # else:
            #     ratios_for_freqs[start_index:end_index] = 1.4 * ratio_for_freq

            print(f"ratios_for_freqs[start_index:end_index]: {ratios_for_freqs[start_index:end_index]}")
            save_end_index = end_index
            ratios_for_freqs[start_index:end_index] = 2**(shift_cent/1200)
            ratios_for_freqs2[start_index:end_index] = ratio_for_freq**6
            with open(csv_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([current_time,  2**(shift_cent/1200)])
            current_time = current_time + time_interval

    ratios_for_freqs[save_end_index:] = 1
    ratios_for_freqs2[save_end_index:] = 1

    # complete_sound = re_doppler.pitch_shift(x=original_signal, p=p, t_p=timeline, Fs=sample_rate)
    resampled_signal, original_time_points, new_time_points = doppler_mul.resampling(freq_ratios=ratios_for_freqs2, pitch_changing_times=timeline,
                                             signal=original_signal, sampling_rate=sample_rate)

    complete_sound = doppler_mul.SOLA(resampled_signal,original_time_points ,new_time_points,sample_rate)

    return complete_sound














