# -- Echo implementation
import math

import numpy as np
from scipy.io.wavfile import write
import csv
from scipy.io.wavfile import read
import matplotlib

import re_doppler
import os
from scipy.io.wavfile import write

matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import os

import scipy.constants as sc
import Doppler_effect_re_formular as Doppler_effect_re_formular
import vtk
import Echo_dist_attenuation as eda


trajectory_folder = "Echo/Trajectories"
if not os.path.exists(trajectory_folder):
    os.makedirs(trajectory_folder)

def save_trajectory_data_csv(folder, filename, data):
    full_path = os.path.join(folder, filename)

    with open(full_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        for row in data:
            writer.writerow(row)

    print(f"Trajectory data saved (CSV): {full_path}")

received_signal = []


csv_reflection_ratio_file = 'Echo/Trajectories/reflection_ratios.csv'
if not os.path.exists(csv_reflection_ratio_file):
    with open(csv_reflection_ratio_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Time', 'Ratio'])

def calculate_travel_time(object1, object2):
    distance = np.linalg.norm(np.array(object1.position) - np.array(object2.position))
    speed_of_sound = 343
    return distance / speed_of_sound

def calculate_distance(position1, position2):
    print("The position1 is: ", position1)
    print("The position2 is: ", position2)
    distance = np.sqrt((position1[0] - position2[0]) ** 2 +
                   (position1[1] - position2[1]) ** 2 +
                   (position1[2] - position2[2]) ** 2)
    distance *=1
    print("The distance is: ", distance)
    return distance



def get_position_at_time(current_time, trajectory, is_microphone=False, triangle_id=None, center=False, tri_point=False, triangle=False):
    if triangle:
        if 'triangle' in trajectory:
            if tri_point:
                for i in range(len(trajectory['triangle'][triangle_id]["points"]['point0']) - 1):
                    start_time, start_pos, start_speed = trajectory['triangle'][triangle_id]["points"]['point0'][i]
                    end_time, end_pos, _ = trajectory['triangle'][triangle_id]["points"]['point0'][i + 1]
                    if start_time <= current_time < end_time:
                        return [trajectory['triangle'][triangle_id]["points"][label][i][1] for label in trajectory['triangle'][triangle_id]["points"]]
                return [trajectory['triangle'][triangle_id]["points"][label][-1][1] for label in trajectory['triangle'][triangle_id]["points"]]
            if center:
                for i in range(len(trajectory['triangle'][triangle_id]["center"]) - 1):
                    start_time, start_pos, start_speed = trajectory['triangle'][triangle_id]["center"][i]
                    end_time, end_pos, _ = trajectory['triangle'][triangle_id]["center"][i + 1]
                    if start_time <= current_time < end_time:
                        return start_pos
                return trajectory['triangle'][triangle_id]["center"][-1][1]
        else:
            return None
    else:
        if is_microphone:
             length = len(trajectory['point0'])
        else:
             length = len(trajectory['centroid'])
        for i in range(length - 1):
            if is_microphone:
                start_time, start_pos, start_speed = trajectory['point0'][i]
                end_time, end_pos, _ = trajectory['point0'][i + 1]
            else:
                start_time, start_pos, start_speed = trajectory['centroid'][i]
                end_time, end_pos, _ = trajectory['centroid'][i + 1]
            if start_time <= current_time < end_time:
                return start_pos
            else:
                continue
        if is_microphone:
            return trajectory['point0'][-1][1]
        else:
            return trajectory['centroid'][-1][1]

def segment_intersects_triangle(P1, P2, V0, V1, V2):
    EPSILON = 1e-9

    P1, P2, V0, V1, V2 = map(np.array, (P1, P2, V0, V1, V2))

    #  calculate 2 edge vectors of the triangle
    edge1 = V1 - V0
    edge2 = V2 - V0

    # Normal vector h
    h = np.cross(P2 - P1, edge2)
    a = np.dot(edge1, h)

    # Determine whether a line segment is parallel to a triangle
    if -EPSILON < a < EPSILON:
        return False  # parallel no intersection

    f = 1.0 / a
    s = P1 - V0
    u = f * np.dot(s, h)

    # Check if the u parameter is within the legal range
    if u < 0.0 or u > 1.0:
        return False

    q = np.cross(s, edge1)
    v = f * np.dot(P2 - P1, q)

    # Check if the v parameter is within the legal range
    if v < 0.0 or u + v > 1.0:
        return False

    t = f * np.dot(edge2, q)

    # Check if the t parameter is in the legal range (the intersection point is between P1 and P2, and does not include P1 and P2)
    if t > EPSILON and t < 1.0 - EPSILON:
        return True

    # print(f"t={t},u={u},v={v}")
    return False

def detect_occlusions(source_label, signal, sample_rate, current_time, source_objects,
                      microphone_object, time_interval, is_source_triangle=False, source_triangle_id=None,
                      is_receiver_microphone=False, receiver_object_label=None, receiver_triangle_id=None):
    for t in np.arange(current_time, current_time + len(signal) / sample_rate, time_interval):
        source_current_pos = get_position_at_time(current_time=t, trajectory=source_objects[source_label].trajectory,
                                                  is_microphone=False, triangle=is_source_triangle, triangle_id=source_triangle_id,
                                                  center=True, tri_point=False)
        is_occluded = False
        if source_current_pos is None:
            continue
        for other_label, other_object in source_objects.items():
            if is_occluded == True:
                break
            if is_source_triangle is False and other_label == source_label:
                continue
            if 'triangle' in other_object.trajectory:
                for other_triangle_id, triangle_positions in other_object.trajectory['triangle'].items():
                    if other_label == source_label and other_triangle_id == source_triangle_id:
                        continue
                    other_triangles_pos_list = get_position_at_time(current_time=t, trajectory=other_object.trajectory,
                                                                    triangle=True, triangle_id=other_triangle_id, tri_point=True)
                    if is_receiver_microphone is True:
                        receiver_current_pos = get_position_at_time(t, microphone_object.trajectory, is_microphone=True)
                    else:
                        receiver_current_pos = get_position_at_time(current_time=t, trajectory=source_objects[receiver_object_label].trajectory,
                                                                    is_microphone=False, triangle=True, triangle_id=receiver_triangle_id, center=True, tri_point=False)
                    if segment_intersects_triangle(source_current_pos, receiver_current_pos, *other_triangles_pos_list):
                        start_index = int((t - current_time) * sample_rate)
                        end_index = min(start_index + int(time_interval * sample_rate), len(signal))
                        signal[start_index:end_index] *= 0.0
                        is_occluded=True
                        break
    return signal


def apply_reflection_coefficient(source_trajectory, receiver_trajectory, source_triangle_trajectory, signal, sample_rate, incident_point_trj, s=0.8):
    #Calculate the position of the source and receiver for each 0.04 second interval, then calculate the corresponding cos(theta), and apply that coefficient to each 0.04 second segment of the signal.

    def calculate_cosine_angle(vector1, vector2):
        vector1 = np.array(vector1)
        vector2 = np.array(vector2)
        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        return cosine_angle

    def calculate_normal(triangle_points, incident_point):
#Calculate the normal vector of the triangle surface and make sure it points to the same side as the incident point.
        V0, V1, V2 = map(np.array, triangle_points)
        edge1 = V1 - V0
        edge2 = V2 - V0
        normal = np.cross(edge1, edge2)
        normal = normal / np.linalg.norm(normal)  # 归一化法向量

        triangle_center = (V0 + V1 + V2) / 3

        # Correct normal vector direction
        incident_vector = np.array(incident_point) - triangle_center
        if np.dot(normal, incident_vector) < 0:
            normal = -normal  # turn normal

        return normal

    processed_signal = signal.copy()

    for i in range(len(source_trajectory) - 1):
        # Get the start and end points of the current time interval
        time_start, source_position_start, source_speed_start = source_trajectory[i]
        time_end, source_position_end, _ = source_trajectory[i + 1]

        _, receiver_position_start, receiver_speed_start = receiver_trajectory[i]
        _, receiver_position_end, receiver_speed_end = receiver_trajectory[i + 1]

        _, source_triangle_points_start, _ = source_triangle_trajectory[i]

        start_index = math.ceil(time_start * sample_rate)
        end_index = math.ceil(time_end * sample_rate)

        # Calculate the current reflection coefficient
        # Calculate the surface normal vector of the source triangle
        normal = calculate_normal(source_triangle_points_start, incident_point_trj[i][1])

        # Calculate the line vector between the source triangle center and the receiver triangle center
        center_vector = np.array(receiver_position_start) - np.array(source_position_start)

        # Calculate the cosine of the angle between the normal vector and the line vector
        cos_theta = calculate_cosine_angle(normal, center_vector)

        # if cos_theta < 0, set the amplitude to 0
        if cos_theta < 0:
            reflection_coefficient = 0.0
        else:
            reflection_coefficient = np.sqrt(s * cos_theta / np.pi)

        #Apply reflection coefficient to signal
        processed_signal[start_index:end_index] = processed_signal[start_index:end_index] * reflection_coefficient

        print(f"Time {time_start:.2f}-{time_end:.2f} s: Reflection coefficient = {reflection_coefficient}")

        # Save reflection coefficients to CSV file
        with open(csv_reflection_ratio_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([time_start, reflection_coefficient])

    return processed_signal


def propagate_signal(current_source_label, current_signal, current_time,source_objects,
                     path_traveled,microphone_object,initial_amplitude,recursion_time=0,threshold=0.00000001,is_source_triangle=False, triangle_id=None,
                     incident_source_trj=None):
    global received_signal
    global sum_count
    current_amplitude = np.max(np.abs(current_signal))

    print(f"recursion time:{recursion_time}")

    # if np.abs(current_amplitude) < np.abs(initial_amplitude*threshold) or recursion_time>=5:
    if np.abs(current_amplitude) < np.abs(initial_amplitude * threshold) :
        return
    if recursion_time > 10:
        return


    for other_label, other_object in source_objects.items():
        if other_label == current_source_label:
            continue
        if 'triangle' in other_object.trajectory:
            for other_triangle_id,other_triangle_trajectory in other_object.trajectory['triangle'].items():

                current_source_pos_save = get_position_at_time(current_time=current_time,
                                                          trajectory=source_objects[current_source_label].trajectory,
                                                          is_microphone=False,
                                                          triangle=is_source_triangle, triangle_id=triangle_id,
                                                          center=True, tri_point=False)
                other_source_pos_save = get_position_at_time(current_time=current_time,
                                                        trajectory=source_objects[other_label].trajectory,
                                                        is_microphone=False,
                                                        triangle=True, triangle_id=other_triangle_id, center=True,
                                                        tri_point=False)


                source_trajectory = []
                source_triangle_trajectory = []
                source_triangle_trajectory2 = []
                receiver_trajectory = []

                duration = len(current_signal) / source_objects[current_source_label].sample_rate  # 计算信号的持续时间
                time_interval = 0.04
                # for t in np.arange(0, duration, time_interval):
                for t in np.linspace(0, duration, int(duration / time_interval) + 1):
                    current_source_pos = get_position_at_time(current_time=current_time+t,
                                   trajectory=source_objects[current_source_label].trajectory, is_microphone=False,
                                   triangle=is_source_triangle,triangle_id=triangle_id,center=True,tri_point=False)
                    other_source_pos = get_position_at_time(current_time=current_time+t,
                                trajectory=source_objects[other_label].trajectory,is_microphone=False,
                                triangle=True,triangle_id=other_triangle_id,center=True,tri_point=False)
                    source_trajectory.append((t, current_source_pos, 0))
                    receiver_trajectory.append((t, other_source_pos, 0))

                save_trajectory_data_csv(trajectory_folder, f"recursion_{recursion_time}_source_trajectory.csv",
                                         source_trajectory)
                save_trajectory_data_csv(trajectory_folder, f"recursion_{recursion_time}_receiver_trajectory.csv",
                                         receiver_trajectory)

                if is_source_triangle:
                    for t in np.linspace(0, duration, int(duration / time_interval) + 1):
                        current_source_triangle_pos = get_position_at_time(current_time=current_time+t,
                                   trajectory=source_objects[current_source_label].trajectory, is_microphone=False,
                                   triangle=is_source_triangle,triangle_id=triangle_id,center=False,tri_point=True)
                        source_triangle_trajectory.append((t, current_source_triangle_pos, 0))
                    save_trajectory_data_csv(trajectory_folder, f"recursion_{recursion_time}_source_triangle_trajectory.csv",
                                             source_triangle_trajectory)


                    current_signal = apply_reflection_coefficient(source_trajectory=source_trajectory,receiver_trajectory=receiver_trajectory,
                                                              source_triangle_trajectory=source_triangle_trajectory,signal=current_signal,
                                                              sample_rate=source_objects[current_source_label].sample_rate,incident_point_trj=incident_source_trj)
                    # Save current_signal
                    output_folder = "Echo/signals"
                    output_filename = f"recursion: {recursion_time}, current_signal.wav"


                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)


                    output_path = os.path.join(output_folder, output_filename)

                    write(output_path, source_objects[current_source_label].sample_rate,
                          current_signal)

                attenuated_signal = eda.amplitude_distance(
                                                        sample_rate=source_objects[current_source_label].sample_rate,
                                                        source_trajectory=source_trajectory,
                                                        receiver_trajectory=receiver_trajectory, signal=current_signal)
                # Save current_signal
                output_folder = "Echo/signals"
                output_filename = f"recursion: {recursion_time}, attenuated_signal_sources.wav"



                output_path = os.path.join(output_folder, output_filename)

                write(output_path, source_objects[current_source_label].sample_rate,
                      attenuated_signal)

                print(f"File saved successfully at: {output_path}")
                source_trajectory_for_do = [pos for _, pos, _ in source_trajectory]
                receiver_trajectory_for_do = [pos for _, pos, _ in receiver_trajectory]
                #Doppler
                doppler_signal = Doppler_effect_re_formular.Doppler_effect_re_formular(sample_rate=source_objects[current_source_label].sample_rate,
                                                               timeline=source_objects[current_source_label].timeline,
                                                               source_trajectory=source_trajectory_for_do,
                                                               receiver_trajectory=receiver_trajectory_for_do,
                                                               signal=attenuated_signal , time_interval = 0.04)

                doppler_signal = np.reshape(doppler_signal, (-1,))

                # Save current_signal
                output_folder = "Echo/signals"
                output_filename = f"recursion: {recursion_time}, doppler_signal_sources.wav"



                output_path = os.path.join(output_folder, output_filename)

                write(output_path, source_objects[current_source_label].sample_rate,
                      doppler_signal)

                #detect_occlusions
                # doppler_signal = detect_occlusions(source_label=current_source_label, signal=doppler_signal,
                #                                    sample_rate=source_objects[current_source_label].sample_rate,
                #                                    current_time=current_time, source_objects=source_objects,
                #                                    microphone_object=microphone_object, time_interval=0.04,
                #                                    is_source_triangle=is_source_triangle,
                #                                    source_triangle_id=triangle_id,
                #                                    is_receiver_microphone=False, receiver_object_label=other_label,
                #                                    receiver_triangle_id=other_triangle_id)


                distance1 = calculate_distance(current_source_pos_save,other_source_pos_save)
                t1 = distance1 / 343  # 343 m/s
                # t1 = distance1 / 100
                new_time = current_time + t1
                #--------------------------------------------------------------------------------------------

                current_other_source_pos2_save = get_position_at_time(current_time=new_time,
                                                                 trajectory=source_objects[other_label].trajectory,
                                                                 is_microphone=False, triangle=True,
                                                                 triangle_id=other_triangle_id, center=True,
                                                                 tri_point=False)
                current_microphone_pos_save = get_position_at_time(current_time=new_time,
                                                          trajectory=microphone_object.trajectory,
                                                          is_microphone=True, triangle=False)


                # if segment_directly_intersects_microphone(P1=current_other_source_pos2_save, P2=current_microphone_pos_save,
                # source_objects=source_objects, source_label=other_label,is_source_triangle=True,source_triangle_id= other_triangle_id,
                #                                           current_time=new_time):
                #source_trajectory和receiver_trajectory
                #start_time: new_time
                # end_time: new_time+duration
                #
                source_trajectory2 = []
                receiver_trajectory2 = []
                duration = len(current_signal) / source_objects[current_source_label].sample_rate  # 计算信号的持续时间
                time_interval = 0.04
                for t in np.linspace(0, duration, int(duration / time_interval) + 1):
                    current_source_pos = get_position_at_time(current_time=new_time + t,
                                                              trajectory=source_objects[other_label].trajectory,
                                                              is_microphone=False,
                                                              triangle=True, triangle_id=other_triangle_id,
                                                              center=True, tri_point=False)
                    other_source_pos = get_position_at_time(current_time=new_time + t,
                                                            trajectory=microphone_object.trajectory,
                                                            is_microphone=True,
                                                            triangle=False)
                    source_trajectory2.append((t, current_source_pos, 0))
                    receiver_trajectory2.append((t, other_source_pos, 0))


                ################################################################################################
                # since this source must be a tri ---------------------- reflection
                for t in np.linspace(0, duration, int(duration / time_interval) + 1):
                    current_other_source_triangle_pos = get_position_at_time(current_time=new_time+t,
                                                                             trajectory=source_objects[other_label].trajectory,
                                                                             is_microphone=False, triangle=True,
                                                                             triangle_id=other_triangle_id,
                                                                             center=False, tri_point=True)
                    source_triangle_trajectory2.append((t, current_other_source_triangle_pos, 0))

                save_trajectory_data_csv(trajectory_folder,
                                         f"recursion_{recursion_time}_source_triangle_trajectory2.csv",
                                         source_triangle_trajectory2)
                doppler_signal_ori = doppler_signal
                doppler_signal = apply_reflection_coefficient(source_trajectory=source_trajectory2,receiver_trajectory=
                        receiver_trajectory2,source_triangle_trajectory=source_triangle_trajectory2,
                        signal=doppler_signal, sample_rate=48000,incident_point_trj=source_trajectory)

                # Save doppler_signal after reflection
                output_folder = "Echo/signals"
                output_filename = f"recursion: {recursion_time}, doppler_signal_after_reflection.wav"



                output_path = os.path.join(output_folder, output_filename)

                write(output_path, source_objects[current_source_label].sample_rate,
                      doppler_signal)


                save_trajectory_data_csv(trajectory_folder, f"recursion_{recursion_time}_source_trajectory2.csv",
                                         source_trajectory2)
                save_trajectory_data_csv(trajectory_folder, f"recursion_{recursion_time}_receiver_trajectory2.csv",
                                         receiver_trajectory2)

                attenuated_signal2 = eda.amplitude_distance(
                                                        sample_rate=source_objects[current_source_label].sample_rate,
                                                        source_trajectory=source_trajectory2,
                                                        receiver_trajectory=receiver_trajectory2, signal=doppler_signal)

                # Save current_signal
                output_folder = "Echo/signals"
                output_filename = f"recursion: {recursion_time}, attenuated_signal_microphone.wav"



                output_path = os.path.join(output_folder, output_filename)

                write(output_path, source_objects[current_source_label].sample_rate,
                      attenuated_signal2)

                source_trajectory2_for_do = [pos for _, pos, _ in source_trajectory2]
                receiver_trajectory2_for_do = [pos for _, pos, _ in receiver_trajectory2]

                doppler_signal2 = Doppler_effect_re_formular.Doppler_effect_re_formular(
                                                        sample_rate=source_objects[current_source_label].sample_rate,
                                                          timeline=source_objects[current_source_label].timeline,
                                                        source_trajectory=source_trajectory2_for_do,
                                                        receiver_trajectory=receiver_trajectory2_for_do,
                                                          signal=attenuated_signal2,time_interval=0.04)
                doppler_signal2 = np.reshape(doppler_signal2, (-1,))

                # Save current_signal
                output_folder = "Echo/signals"
                output_filename = f"recursion: {recursion_time}, doppler_signal_microphone.wav"



                output_path = os.path.join(output_folder, output_filename)

                write(output_path, source_objects[current_source_label].sample_rate,
                      doppler_signal2)

                #
                # doppler_signal2 = detect_occlusions(source_label=other_label, signal=doppler_signal2,
                #                                    sample_rate=source_objects[current_source_label].sample_rate,
                #                                    current_time=new_time, source_objects=source_objects,
                #                                    microphone_object=microphone_object, time_interval=0.04,
                #                                    is_source_triangle=True,
                #                                    source_triangle_id=other_triangle_id,
                #                                    is_receiver_microphone=True)



                # if other_source_pos2==None or microphone_pos==None:
                #     continue

                distance2 = calculate_distance(current_other_source_pos2_save, current_microphone_pos_save)
                t2 = distance2 / 343
                # t2 = distance2 / 100
                # t2 = distance2 / 200
                total_time = new_time + t2
                # attenuation_factor2 = calculate_attenuation(distance2)
                # attenuated_signal2 = doppler_signal2 * attenuation_factor2

                sample_rate = source_objects[current_source_label].sample_rate

                time_index = int(total_time * sample_rate)
                end_index = time_index + len(doppler_signal2)
                if end_index > len(received_signal):

                    extension = end_index - len(received_signal)
                    received_signal = np.concatenate((received_signal, np.zeros(extension)))

                received_signal[time_index:end_index] += doppler_signal2
                # Save current_signal
                output_folder = "Echo/signals"
                output_filename = f"recursion: {recursion_time}, received_signal.wav"



                output_path = os.path.join(output_folder, output_filename)

                write(output_path, source_objects[current_source_label].sample_rate,
                      received_signal)
                # max_abs_value = np.max(np.abs(received_signal))
                # if max_abs_value != 0:
                #     received_signal = received_signal / max_abs_value
                recursion_time+=1
                #
                propagate_signal(current_source_label = other_label, current_signal=doppler_signal_ori,
                             current_time=new_time,
                             path_traveled= path_traveled + [other_label],
                             source_objects=source_objects, microphone_object=microphone_object,
                             initial_amplitude=initial_amplitude,threshold=threshold,is_source_triangle=True,
                             triangle_id=other_triangle_id,recursion_time=recursion_time,incident_source_trj=source_trajectory)

def Echo(source_objects, microphone_object):
    #
    global received_signal

    # global sum_count
    for source_label, source_object in source_objects.items():
        if source_object.signal is None:
            continue
        initial_signal = source_object.signal
        initial_time = 0
        initial_amplitude = np.max(np.abs(initial_signal))
        path_traveled = [source_label]
        to_microphone = eda.amplitude_distance(sample_rate=source_object.sample_rate,
                                                               source_trajectory=source_object.trajectory['centroid'],
                                                               receiver_trajectory=microphone_object.trajectory['point0'],
                                                               signal=initial_signal)
        initial_source_trajectory_for_do = [pos for _, pos, _ in source_object.trajectory['centroid']]
        initial_receiver_trajectory_for_do = [pos for _, pos, _ in microphone_object.trajectory['point0']]
        #
        to_microphone = Doppler_effect_re_formular.Doppler_effect_re_formular(sample_rate=source_objects[source_label].sample_rate,
                                                      timeline=source_objects[source_label].timeline,
                                                      source_trajectory=initial_source_trajectory_for_do,
                                                      receiver_trajectory=initial_receiver_trajectory_for_do,
                                                      signal=to_microphone,time_interval=0.04)
        to_microphone = np.reshape(to_microphone, (-1,))

        save_trajectory_data_csv(trajectory_folder, f"initial_{source_label}_trajectory.csv",
                                 source_object.trajectory['centroid'])
        save_trajectory_data_csv(trajectory_folder, f"initial_microphone_trajectory.csv",
                                 microphone_object.trajectory['point0'])

        recursion_time = 0
        output_folder = "Echo/signals"


        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_filename = f"recursion: {recursion_time}, to_microphone.wav"


        output_path = os.path.join(output_folder, output_filename)

        write(output_path, source_objects[source_label].sample_rate,
              to_microphone)

        #
        # to_microphone = detect_occlusions(source_label=source_label, signal=to_microphone,
        #                                     sample_rate=source_objects[source_label].sample_rate,
        #                                     current_time=initial_time, source_objects=source_objects,
        #                                     microphone_object=microphone_object, time_interval=0.04,
        #                                     is_source_triangle=False,is_receiver_microphone=True)

        # the time when signal arrives to microphone
        # current_distance between source and micophone/c
        distance = calculate_distance(source_object.trajectory['centroid'][0][1],microphone_object.trajectory['point0'][0][1])
        t_m = distance/343
        time_index = int((initial_time+t_m) * source_object.sample_rate)
        # time_index = int((initial_time ) * source_object.sample_rate)
        end_index = time_index + len(to_microphone)
        if end_index > len(received_signal):
            #
            extension = end_index - len(received_signal)
            received_signal = np.concatenate((received_signal, np.zeros(extension)))

        received_signal[time_index:end_index] += to_microphone

        recursion_time = 0
        output_folder = "Echo/signals"


        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_filename = f"recursion: {recursion_time}, received_signal_initial.wav"


        output_path = os.path.join(output_folder, output_filename)

        write(output_path, source_objects[source_label].sample_rate,
              received_signal)



        propagate_signal(current_source_label=source_label, current_signal=initial_signal, current_time=initial_time,
                                                  path_traveled=path_traveled, is_source_triangle=False, triangle_id='triangle0',
                                                  source_objects=source_objects, microphone_object=microphone_object,
                                                  initial_amplitude=initial_amplitude,recursion_time=recursion_time, threshold=0.00000001,
                         incident_source_trj=None)

    # received_signal = received_signal/sum_count
    # normalized_sound = received_signal / received_signal.max()
    print("The duration of received signal is: ", len(received_signal)/48000)
    return received_signal











