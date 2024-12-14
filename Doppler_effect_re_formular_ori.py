import math

import numpy as np
import matplotlib
matplotlib.use('MacOSX')
import re_doppler

def Doppler_effect_re_formular(sample_rate,timeline,source_trajectory,receiver_trajectory,signal):
    original_signal = signal
    sample_rate = sample_rate

    p = np.zeros_like(timeline)

    for i in range(len(source_trajectory)-1):
        #Get the start and end points of the current time interval
        time_start, source_position_start,source_speed_start = source_trajectory[i]
        time_end, source_position_end,_ = source_trajectory[i+1]

        # if receiver_object.label.startswith('microphone'):
        #     _,receiver_position_start, receiver_speed_start = receiver_trajectory['point0'][i]
        #     _,receiver_position_end, receiver_speed_end = receiver_trajectory['point0'][i+1]
        # else:
        #     _,receiver_position_start, receiver_speed_start = receiver_trajectory['centroid'][i]
        #     _,receiver_position_end,receiver_speed_end = receiver_trajectory['centroid'][i+1]

        _,receiver_position_start, receiver_speed_start = receiver_trajectory[i]
        _,receiver_position_end,receiver_speed_end = receiver_trajectory[i+1]

        source_position_start = np.array(source_position_start)
        source_position_end = np.array(source_position_end)
        receiver_position_start = np.array(receiver_position_start)
        receiver_position_end = np.array(receiver_position_end)

        delta_t = time_end - time_start
        delta_x_source = source_position_end - source_position_start
        u = delta_x_source / delta_t
        delta_x_receiver = receiver_position_end - receiver_position_start
        v = delta_x_receiver / delta_t
        delta_v = u - v
        x = receiver_position_start - source_position_start
        magnitude_delta_v = np.linalg.norm(delta_v)

        magnitude_delta_v = magnitude_delta_v * 10
        print("magnitude_delta_v: ",magnitude_delta_v)
        magnitude_x = np.linalg.norm(x)
        dot_product = np.dot(delta_v, x)
        #The angle between two vectors
        # Skip the current loop if magnitude_delta_v or magnitude_x is 0
        if magnitude_delta_v == 0 or magnitude_x == 0:
            continue

        cos_theta = dot_product / (magnitude_delta_v * magnitude_x)
        print("cos_theta: ",cos_theta)

        ratio_for_freq = 1+magnitude_delta_v*cos_theta/340
        print("ratio_for_freq: ",ratio_for_freq)


        start_index = math.ceil(time_start*sample_rate)
        end_index = math.ceil(time_end*sample_rate)

        if (source_position_start is None or source_position_end is None
                or receiver_position_start is None or receiver_position_end is None):
            original_signal = original_signal.astype(np.float64)
            original_signal[start_index:end_index] *= 0.001
            original_signal = original_signal.astype(np.int16)
        else:

            shift_cent = round(12 * math.log2(ratio_for_freq)*100)

            p[start_index:end_index] = shift_cent

    complete_sound = re_doppler.pitch_shift(x=original_signal, p=p, t_p=timeline, Fs=sample_rate)
    return complete_sound














