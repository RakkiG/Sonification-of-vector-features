
import math
import os


import numpy as np
import csv
import matplotlib
matplotlib.use('MacOSX')
import Tool_Functions as tf

def amplitude_distance(sample_rate,source_trajectory,receiver_trajectory,signal):
    csv_file = 'Echo/Trajectories/distance_ratio.csv'
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Start Time', 'Distance', 'Ratio'])  # 添加标题行

    original_signal = np.copy(signal)

    sample_rate = sample_rate

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

        start_index = math.ceil(time_start * sample_rate)
        end_index = math.ceil(time_end * sample_rate)

        #check if the source position is None for this segment
        if source_position_start == None or source_position_end == None or receiver_position_start == None or receiver_position_end == None:
            #if one of the source positions is None, then the segment is silent
            original_signal[start_index:end_index] = original_signal[start_index:end_index]*0.0001
            # ori_sig_seg = original_signal[start_index:end_index]
            # new_sig_seg= ori_sig_seg*0.0001
        else:
            distance_value = tf.distance(source_position_start,receiver_position_start)


            ratio = 1/(distance_value+1)
            # ratio = np.sqrt(1 / (distance_value + 1))
            ratio = ratio**(1/4)
            # if ratio>=1:
            #     ratio = 0.9
            # amplitude_factor = 0.3
            # ratio *= amplitude_factor
            # print("---amplitude-distance ratio:---------",ratio)

            with open(csv_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([time_start, distance_value, ratio])
            # ratio_rounded = round(ratio, 1)

            # ori_sig_seg = original_signal[start_index:end_index]
            # new_sig_seg = ori_sig_seg*ratio_rounded
            original_signal[start_index:end_index] = original_signal[start_index:end_index]*ratio
        # all_segments.append(new_sig_seg)

    end_time_signal,_,_ = source_trajectory[-1]
    end_index_signal = math.ceil(end_time_signal*sample_rate)
    return original_signal[:end_index_signal]










