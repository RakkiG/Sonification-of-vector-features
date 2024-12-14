# --- Amplitude changing with distance
import math
import os


import numpy as np
import csv
import matplotlib
matplotlib.use('MacOSX')
import Tool_Functions as tf


def amplitude_distance(sample_rate,timeline,source_trajectory,receiver_trajectory,signal,time_interval):
    original_signal = np.copy(signal)
    ln1_5 = math.log(1.5)
    for i in range(len(source_trajectory)-1):
        #Get the start and end points of the current time interval
        source_position_start = source_trajectory[i]

        receiver_position_start = receiver_trajectory[i]

        start_index = math.ceil(i * time_interval * sample_rate)
        end_index = math.ceil((i + 1) * time_interval * sample_rate)

        distance_value = tf.distance(source_position_start,receiver_position_start)

        # if distance_value <= ln1_5:
        #     distance_value = ln1_5 #avoid division by zero
        if distance_value <= 1/2:
            distance_value = 1/2 #avoid division by zero

            #modify ratio using distance to the power of 'power_factor'
        # ratio = 1/np.exp(distance_value)-1
        # ratio = 1/(np.sqrt(distance_value))
        ratio = 1 / distance_value
        print("---amplitude-distance ratio:---------",ratio)

        original_signal[start_index:end_index] = original_signal[start_index:end_index]*ratio

    # calculate the last distance
    last_distance = tf.distance(source_trajectory[-1],receiver_trajectory[-1])

    if last_distance <= 1 / 2:
        last_distance = 1 / 2  # avoid division by zero
    last_ratio = 1 / last_distance
    last_start_index = math.ceil((len(source_trajectory)-1) * time_interval * sample_rate)
    original_signal[last_start_index:] = original_signal[last_start_index:]*last_ratio

    end_time_signal= math.ceil((len(source_trajectory))*time_interval)
    end_index_signal = math.ceil(end_time_signal*sample_rate)
    return original_signal[:len(timeline)]










