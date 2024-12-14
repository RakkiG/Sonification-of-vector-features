# Method 1: Positive eigenvalues match 200-400, negative eigenvalues match 800-1000
# Generate videos for the whole results
import math
import os

import numpy as np
from scipy.io import wavfile
from vtkmodules.vtkIOOggTheora import vtkOggTheoraWriter
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor, vtkWindowToImageFilter

import Tool_Functions as tf
import vtk
import Source_Microphone
import Source_Microphone_ori
from scipy.interpolate import CubicHermiteSpline
import Animation_flying_microphone as afm
import scipy.io.wavfile as wav
import Doppler_effect_re_formular as doppler_effect
import Amplitude_distance2 as ad
import Echo

vector_filename = 'vector_field_complex.vti'
# filename = 'vector_field_complex_single.vti'
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(vector_filename)
reader.Update()

image_data = reader.GetOutput()
critical_points, jacobians, eigenvalues_list, eigenvectors_list = tf.extract_critical_points_and_jacobians(image_data)

indices_to_remove = {3, 4}

critical_points = [point for i, point in enumerate(critical_points) if i not in indices_to_remove]
jacobians = [jacobian for i, jacobian in enumerate(jacobians) if i not in indices_to_remove]
eigenvalues_list = [eigenvalues for i, eigenvalues in enumerate(eigenvalues_list) if i not in indices_to_remove]
eigenvectors_list = [eigenvectors for i, eigenvectors in enumerate(eigenvectors_list) if i not in indices_to_remove]

# planes = tf.generate_planes(image_data)
planes = tf.generate_planes(image_data, selected_bounds=["X-min", "X-max"])

# Using a custom plane
# custom_planes = [
#     [(-30, 0, 20), (-30, 20, 20), (-30, 20, 0), (-30, 0, 0)],
#     [(50, 20, 0), (50, 20, 20), (50, 0, 20), (50, 0, 0)]
# ]
# planes = tf.generate_planes(custom_planes=custom_planes)

#Get triangle array
max_triangle_area = 5000
triangles = tf.divide_planes_into_triangles(planes, max_triangle_area)

is_echo = False

if is_echo:
    plane_actors = tf.create_plane_actors(planes)

tetra_filename = 'vortex_core_line.vtu'
tvtu_file = tf.hexahedron_to_tetrahedra(vector_filename, tetra_filename)

#Create a frequency array to store the frequency of each eigenvalue map
freqs = tf.get_frequencies_for_eigenvalues2(eigenvalues_list)

#1. extract and save vortex core lines
vortex_lines, eigenvalues_for_each_core_line = tf.read_vtu_file_and_process_cells(tetra_filename)
sources = {}

# Remove duplicates from vortex lines and corresponding eigenvalues
unique_vortex_lines, unique_eigenvalues = tf.remove_duplicate_vortex_lines(vortex_lines, eigenvalues_for_each_core_line)

# 计算两个点之间的欧氏距离
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

#Group unique_vortex_lines by connectivity. If the start of a vortex line coincides with the end of another
# (the distance is less than or equal to 10^(-6)),
# or the end of a line coincides with the start of another (the distance is less than or equal to 10^(-6)),
# they are considered connected and added to a group.

# Group the vortex lines based on connectivity
connected_vortex_groups,connected_vortex_groups_eigenvalues = tf.group_connected_vortex_lines(unique_vortex_lines,unique_eigenvalues)

# calculate distances between critical points
distances = []

for i in range(len(critical_points)):
    min_distance = float('inf')
    for j in range( len(critical_points)):
        if i != j:
            # Calculate the distance between the current point (i) and another point (j)
            distance = np.linalg.norm(np.array(critical_points[i]) - np.array(critical_points[j]))
            # Update the minimum distance if the current distance is smaller
            if distance < min_distance:
                min_distance = distance
    # Store the minimum distance for the current point
    distances.append(min_distance)

# find the maximum and minimum distances
max_distance = max(distances) if distances else None
min_distance = min(distances) if distances else None
average_distance = np.mean(distances) if distances else None

# create critical points' sources
for index,critical_point in enumerate(critical_points):
    sources[f"source{index}"] = Source_Microphone.Source(signal=None, positions=[critical_point], sample_rate=None,
                                         timeline=None, label=f"source{index}", initial_pos=critical_point,
                                         radius=average_distance/2, jacobian_eigenvalues=eigenvalues_list[index])

#Traverse connected_vortex_groups. For each group, every 6 lines is used as a sound source.
# The sound source point is at the end of the third point. #First select the third point from bottom to top,
# and then every 6 points as a sound source. If there are less than 6 points, select the midpoint in the z direction as
# the sound source point.
num_critical_sources = len(sources)
length_per_source = 100
i = num_critical_sources

for index, group in enumerate(connected_vortex_groups):
    num_lines = len(group)
    if num_lines < 3:
        continue
    elif 3 <= num_lines < length_per_source:
        # line numbers from bottom to top
        line_th = num_lines // 2
        #source_pos
        source_pos = group[line_th][0]
        # Subtract the positions of the first and last points of this group and take half of their length as the radius
        group_1st_point = group[0][0]
        group_last_point = group[len(group)-1][1]

        radius = euclidean_distance(group_1st_point,group_last_point)/2
        jacobian_eigenvalues = connected_vortex_groups_eigenvalues[index][line_th][0]

        #Process the signal directly here
        sources[f"source{i}"] = Source_Microphone.Source(signal=None, positions=[source_pos],sample_rate=None,
            timeline=None, label=f"source{i}", initial_pos=source_pos, radius=radius, jacobian_eigenvalues=jacobian_eigenvalues)

        i+=1
    else:
        #First, take the points within the range divisible by length_per_source as the sound source
        j = 0
        remain_num_lines = len(group)-j*length_per_source
        while remain_num_lines >= length_per_source:
            source_pos = group[j*length_per_source+length_per_source//2][0]

            start_source_point = group[j*length_per_source][0]
            end_source_point = group[j*length_per_source+length_per_source-1][1]

            radius = euclidean_distance(start_source_point,end_source_point)/2
            jacobian_eigenvalues = connected_vortex_groups_eigenvalues[index][j*length_per_source+length_per_source//2][0]  # 取中间

            sources[f"source{i}"] = Source_Microphone.Source(signal=None, positions=[source_pos],sample_rate=None,
                timeline=None, label=f"source{i}",initial_pos=source_pos,radius=radius,jacobian_eigenvalues=jacobian_eigenvalues)
            j += 1
            remain_num_lines = len(group)-j*length_per_source
            i+=1
        # The points within the remainder range of 6 are used as sound sources
        # The number of j is the current number of sound sources
        # The number of lines in the last remaining line
        if remain_num_lines:
            line_th = remain_num_lines // 2
            source_pos = group[j*length_per_source+line_th][0]

            start_source_point = group[j * length_per_source][0]
            end_source_point = group[j * length_per_source + remain_num_lines - 1][1]

            radius = euclidean_distance(start_source_point, end_source_point)/2
            jacobian_eigenvalues = \
            connected_vortex_groups_eigenvalues[index][j * length_per_source + remain_num_lines // 2][0]

            sources[f"source{i}"] = Source_Microphone.Source(signal=None, positions=[source_pos],sample_rate=None,
                timeline=None, label=f"source{i}", initial_pos=source_pos,radius=radius,jacobian_eigenvalues=jacobian_eigenvalues)
            i += 1

imag_parts = []
# Find the absolute value of the imaginary part of the eigenvalues of all vortex core lines sources, and then find the maximum value
for i in range(num_critical_sources, len(sources)):
    eigenvalues = sources[f"source{i}"].jacobian_eigenvalues

    for number in eigenvalues:
        if np.iscomplex(number):
            chosen_complex_number = number
            break

    # extract the imaginary part of
    imag_part = np.imag(chosen_complex_number)

    imag_parts.append(imag_part)

vortex = 0
if len(imag_parts):
    # Mark to indicate vortex core line related
    vortex = 1
    max_ab_imag_eigenvalue = max(abs(imag_part) for imag_part in imag_parts)
    min_ab_imag_eigenvalue = min(abs(imag_part) for imag_part in imag_parts)


def generate_random_points_on_sphere(center, radius, num_points=4):
    points = []
    for _ in range(num_points):
        # Randomly generate points on spherical coordinates
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        x = center[0] + radius * np.sin(phi) * np.cos(theta)
        y = center[1] + radius * np.sin(phi) * np.sin(theta)
        z = center[2] + radius * np.cos(phi)
        points.append([x, y, z])
    return points

path_points =[]

for i in range(num_critical_sources):
    #
    random_points = generate_random_points_on_sphere(sources[f"source{i}"].positions[0], 3,10)
    for point in random_points:
        path_points.append(point)

for i in range(num_critical_sources, len(sources)):
    #
    random_points = generate_random_points_on_sphere(sources[f"source{i}"].positions[0], 3,8)
    for point in random_points:
        path_points.append(point)

#sort path_points
def sort_points(points):
    # Firstly find the point with the smallest x-axis coordinate value
    sorted_points = [min(points, key=lambda p: p[0])]
    points.remove(sorted_points[0])

    while points:
        # Find the point with the smallest distance to the last sorted point
        next_point = min(points, key=lambda p: euclidean_distance(sorted_points[-1], p))
        sorted_points.append(next_point)
        points.remove(next_point)

    return sorted_points

# Sort the path points
sorted_path_points = sort_points(path_points)

sorted_path_points = np.array(sorted_path_points)

t = np.zeros(sorted_path_points.shape[0])
for i in range(1, sorted_path_points.shape[0]):
    t[i] = t[i - 1] + euclidean_distance(sorted_path_points[i - 1], sorted_path_points[i])

#
derivatives = np.gradient(sorted_path_points, t, axis=0)

num = 20*len(sorted_path_points)

t_new = np.linspace(t[0], t[-1], num)

hermite_x = CubicHermiteSpline(t, sorted_path_points[:, 0], derivatives[:, 0])
hermite_y = CubicHermiteSpline(t, sorted_path_points[:, 1], derivatives[:, 1])
hermite_z = CubicHermiteSpline(t, sorted_path_points[:, 2], derivatives[:, 2])

interpolated_x = hermite_x(t_new)
interpolated_y = hermite_y(t_new)
interpolated_z = hermite_z(t_new)

interpolated_path_points = np.vstack((interpolated_x, interpolated_y, interpolated_z)).T

num_steps = len(interpolated_path_points)

time_interval = 40  # 40 milliseconds

sound_duration = time_interval/1000 * num_steps
sampling_rate = 48000

t = np.linspace(0, sound_duration, math.ceil(sampling_rate * sound_duration), endpoint=False)

for i in range(num_critical_sources):
    eigenvalues = sources[f"source{i}"].jacobian_eigenvalues
    # get the 3 freqs of the current eigenvalues set
    # freqs = tf.get_frequencies_for_eigenvalues(eigenvalues)
    freqens = freqs[i][0], freqs[i][1], freqs[i][2]

    # Generate sin signal
    signal = np.sin(2 * np.pi * freqens[0] * t)+ np.sin(2 * np.pi * freqens[1] * t)+ np.sin(2 * np.pi * freqens[2] * t)

    signal = signal / np.max(np.abs(signal))

    # Add the signal to the source
    sources[f"source{i}"].signal = signal
    sources[f"source{i}"].timeline = t

if vortex == 1:
    for i in range(num_critical_sources, len(sources)):
        eigenvalues = sources[f"source{i}"].jacobian_eigenvalues
        filename = "/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/vector_field_process8/sounds/wormhole_2.wav"
        signal = tf.generate_sound_for_vortex_core_line(eigenvalues=eigenvalues, filename=filename,
            max_ab_imag_eigenvalue=max_ab_imag_eigenvalue, min_ab_imag_eigenvalue=min_ab_imag_eigenvalue,
            new_max=3, new_min = 0.5,signal_duration=sound_duration)
        signal = signal / np.max(np.abs(signal))

        # Add the signal to the source
        sources[f"source{i}"].signal = signal
        sources[f"source{i}"].timeline = t

video_signal = np.zeros(int(sampling_rate * sound_duration))  # 1-channel audio signal

#create microphone actor

def create_octagonal_microphone():
    octahedron_source = vtk.vtkPlatonicSolidSource()
    octahedron_source.SetSolidTypeToOctahedron()
    octahedron_source.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(octahedron_source.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor

# Create mirophone object
microphone = Source_Microphone.Microphone(label='microphone', initial_pos=sorted_path_points[0],trajectory=interpolated_path_points)
microphone_actor = create_octagonal_microphone()

# create VTK rendering pipeline
renderer = vtkRenderer()
renderer.SetBackground(1, 1, 1)


vector_field_actors = []

# filename = 'vector_field_complex.vti'
critical_points_actors, lines, vortex_lines_actor, legend_actors = tf.vector_field_actors(vector_filename)

for critical_point_actor in critical_points_actors:
    vector_field_actors.append(critical_point_actor)
    renderer.AddActor(critical_point_actor)

for legend_actor in legend_actors:
    vector_field_actors.append(legend_actor)
    renderer.AddActor(legend_actor)

if is_echo:
    for plane_actor in plane_actors:
        vector_field_actors.append(plane_actor)
        renderer.AddActor(plane_actor)

vector_field_actors.append(lines)
vector_field_actors.append(vortex_lines_actor)
renderer.AddActor(lines)
renderer.AddActor(vortex_lines_actor)

renderer.AddActor(microphone_actor)


def create_transparent_sphere(center, radius,color=(0.5, 0.5, 1), opacity=0.1):
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetCenter(center)
    sphere_source.SetRadius(radius)
    sphere_source.SetThetaResolution(50)
    sphere_source.SetPhiResolution(50)
    sphere_source.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere_source.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetColor(color)  # Light blue color

    return actor

microphone_trajectory = interpolated_path_points
doppler_signals = []
doppler_amplitude_signals = []


##########################################################create echo-sources and microphone(down)##########################################################
if is_echo:

    # planes = tf.generate_planes(image_data)
    planes = tf.generate_planes(image_data, selected_bounds=["X-min", "X-max"])


    # custom_planes = [
    #     [(-30, 0, 20), (-30, 20, 20), (-30, 20, 0), (-30, 0, 0)],
    #     [(50, 20, 0), (50, 20, 20), (50, 0, 20), (50, 0, 0)]
    # ]
    # planes = tf.generate_planes(custom_planes=custom_planes)


    max_triangle_area = 5000
    triangles = tf.divide_planes_into_triangles(planes, max_triangle_area)

    plane_actors = tf.create_plane_actors(planes)
    if is_echo:
        for plane_actor in plane_actors:
            vector_field_actors.append(plane_actor)
            renderer.AddActor(plane_actor)
    echo_sources = {}
    i = 0
    for source_label, source in sources.items():
        echo_source = Source_Microphone_ori.Source(signal=source.signal, positions=source.positions,
                                                   sample_rate=48000,
                                                   timeline=source.timeline, label=source.label,
                    initial_points_pos=(source.initial_pos[0],source.initial_pos[1],source.initial_pos[2]), triangles=None)
        echo_traj = {}
        echo_traj['centroid']=[]
        # echo_traj['triangle']={}
        # echo_traj['triangle']['triangle0'] = {}
        # echo_traj['triangle']['triangle0']['center'] = []
        # echo_traj['triangle']['triangle0']['points'] = {}
        # echo_traj['triangle']['triangle0']['points']['point0']=[]
        # echo_traj['triangle']['triangle0']['points']['point1'] = []
        # echo_traj['triangle']['triangle0']['points']['point2'] = []

        current_time = 0
        for a in range(len(interpolated_path_points)):
            echo_traj['centroid'].append((current_time,echo_source.initial_points_pos,0))
            current_time += time_interval / 1000

        # add trajectory
        echo_source.addTrajectory(echo_traj)

        echo_sources[f'source{i}'] = echo_source
        i+=1

    for tri_label, triangle in triangles.items():
        echo_traj = {}
        echo_traj['centroid'] = []
        echo_traj['triangle']={}
        echo_traj['triangle']['triangle0'] = {}
        echo_traj['triangle']['triangle0']['center'] = []
        echo_traj['triangle']['triangle0']['points'] = {}
        echo_traj['triangle']['triangle0']['points']['point0']=[]
        echo_traj['triangle']['triangle0']['points']['point1'] = []
        echo_traj['triangle']['triangle0']['points']['point2'] = []


        echo_source = Source_Microphone_ori.Source(signal=None, positions=None, sample_rate=48000,
                                                   timeline=t, label=f'source{i}',
                                                   initial_points_pos=None, triangles=None)
        echo_source.triangles = {}
        echo_source.triangles['triangle0']=triangle
        current_time = 0
        for j in range(len(interpolated_path_points)):
            echo_traj['centroid'].append((current_time,triangles[tri_label]['center'],0))
            echo_traj['triangle']['triangle0']['center'].append((current_time,triangles[tri_label]['center'],0))
            echo_traj['triangle']['triangle0']['points']['point0'].append((current_time,triangles[tri_label]['points'][0],0))
            echo_traj['triangle']['triangle0']['points']['point1'].append((current_time,triangles[tri_label]['points'][1],0))
            echo_traj['triangle']['triangle0']['points']['point2'].append((current_time,triangles[tri_label]['points'][2],0))
            current_time += time_interval/1000

        echo_source.addTrajectory(echo_traj)
        echo_sources[f'source{i}'] = echo_source
        i+=1

    # create echo-based microphone object
    echo_microphone = Source_Microphone_ori.Microphone(signal=None, positions=None, sample_rate=48000,timeline=t, label='microphone0',initial_points_pos=0)
    echo_micro_traj = {}
    echo_micro_traj['point0'] = []
    current_time = 0
    for k in range(len(interpolated_path_points)):
        echo_micro_traj['point0'].append((current_time,(microphone_trajectory[k][0],microphone_trajectory[k][1],microphone_trajectory[k][2]),0))
        current_time += time_interval/1000
    echo_microphone.addTrajectory(echo_micro_traj)

    echo_signal = Echo.Echo(echo_sources,echo_microphone)

    max_amplitude = np.max(np.abs(echo_signal))
    if max_amplitude > 0:
        normalized_echo_signal = echo_signal / max_amplitude
    else:
        normalized_echo_signal = echo_signal

    echo_filename = "echo_signal_output.wav"

    wavfile.write(echo_filename, sampling_rate, normalized_echo_signal)
##########################################################create echo-sources and microphone(up)##########################################################

#Create folder for critical points' signals
output_folder = 'doppler_signals'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for i in range(len(sources)):
    # Apply doppler effect to the audio signals
    source_trajectory = np.zeros_like(microphone_trajectory)
    source_pos = sources[f"source{i}"].positions[0]
    signal = sources[f"source{i}"].signal
    for j in range(len(source_trajectory)):
        source_trajectory[j] = source_pos
    doppler_signal = doppler_effect.Doppler_effect_re_formular(sample_rate=sampling_rate, timeline=t,
                source_trajectory=source_trajectory, receiver_trajectory=microphone_trajectory, signal=signal,
                time_interval=time_interval/1000)
    doppler_signal = np.squeeze(doppler_signal)  # Remove single-dimensional array

    doppler_signal/=2

    print(f"The length of doppler signal is {len(doppler_signal)}")

    doppler_amplitude_signal = ad.amplitude_distance(sample_rate=sampling_rate, timeline=t,source_trajectory=source_trajectory,
            receiver_trajectory=microphone_trajectory, signal=doppler_signal, time_interval=time_interval/1000)

    doppler_amplitude_signal = doppler_amplitude_signal / np.max(np.abs(doppler_amplitude_signal))  # Normalize the signal
    print(f"The length of doppler_amplitude_signal is {len(doppler_amplitude_signal)}")
    # Save signal
    output_filename = os.path.join(output_folder, f'doppler_amplitude_signal_{i}.wav')
    # save the signal to the output folder
    wavfile.write(output_filename, sampling_rate, doppler_amplitude_signal)

    doppler_amplitude_signals.append(doppler_amplitude_signal)

# process sphere signal Accumulation
video_signal = np.zeros_like(t)
for time_step, pos in enumerate(microphone_trajectory):
    start_index = math.ceil(time_step * time_interval / 1000 * sampling_rate)
    end_index = math.ceil((time_step+1) * time_interval / 1000 * sampling_rate)
    for i in range(len(sources)):
        sphere_center = sources[f"source{i}"].positions[0]
        radius = sources[f"source{i}"].radius
        if euclidean_distance(pos, sphere_center) <= radius:
            a= doppler_amplitude_signals[i][start_index:end_index]
            #Accumulate signal
            video_signal[start_index:end_index] += doppler_amplitude_signals[i][start_index:end_index]
            video_signal[start_index:end_index] = video_signal[start_index:end_index] / np.max(np.abs(video_signal[start_index:end_index]))  # Normalize the signal


def sigmoid(x, x_shift=0, scale=1):
    return 1 / (1 + np.exp(-scale * (x - x_shift)))

def flipped_sigmoid(x, x_shift=0, scale=1):
    #Sigmoid function flipped along the y-axis and translated along the x-axis.
    return 1 - sigmoid(x, x_shift, scale)

# Initialize the state variable to determine whether the microphone is currently inside the sphere
inside_sphere = {f"source{i}": False for i in range(len(sources))}

inside_periods = {f"source{i}": [] for i in range(len(sources))}

video_signal = np.zeros_like(t)
for time_step, pos in enumerate(microphone_trajectory):
    start_index = math.ceil(time_step * time_interval / 1000 * sampling_rate)
    end_index = math.ceil((time_step + 1) * time_interval / 1000 * sampling_rate)

    for i in range(len(sources)):
        sphere_center = sources[f"source{i}"].positions[0]
        radius = sources[f"source{i}"].radius
        distance_to_sphere = euclidean_distance(pos, sphere_center)


        if distance_to_sphere <= radius:
            if not inside_sphere[f"source{i}"]:
                inside_sphere[f"source{i}"] = True  #
                inside_periods[f"source{i}"].append([start_index, end_index])
            else:
                inside_periods[f"source{i}"][-1][1] = end_index
        else:
            if inside_sphere[f"source{i}"]:
                inside_sphere[f"source{i}"] = False

for i in range(len(sources)):
    for period in inside_periods[f"source{i}"]:
        start_index, end_index = period

        signal_segment = doppler_amplitude_signals[i][start_index:end_index]

        time_segment = np.linspace(0, 1, len(signal_segment))

        fade_in = sigmoid(time_segment, x_shift=0, scale=0.75)

        fade_out = flipped_sigmoid(time_segment, x_shift=0, scale=0.75)

        smooth_signal = signal_segment * fade_in * fade_out

        video_signal[start_index:end_index] += smooth_signal

max_abs_value = np.max(np.abs(video_signal))
if max_abs_value != 0:
    video_signal /= max_abs_value

# for i, source in enumerate(sources):
#     print(f"Source {i}:")
#     print(f"  Inside periods: {inside_periods[f'source{i}']}")

# for i in range(len(sources)):
#     sphere_center = sources[f"source{i}"].positions[0]
#     radius = sources[f"source{i}"].radius
#     sphere_actor = create_transparent_sphere(sphere_center, radius)
#     # spheres.append(sphere_actor)
#     renderer.AddActor(sphere_actor)

for i in range(num_critical_sources):
    sphere_center = sources[f"source{i}"].positions[0]
    radius = sources[f"source{i}"].radius
    sphere_actor = create_transparent_sphere(sphere_center, radius, color=(1, 0, 0))  # Red color
    # spheres.append(sphere_actor)
    renderer.AddActor(sphere_actor)

for i in range(num_critical_sources,len(sources)):
    sphere_center = sources[f"source{i}"].positions[0]
    radius = sources[f"source{i}"].radius
    sphere_actor = create_transparent_sphere(sphere_center, radius)  # Blue color
    # spheres.append(sphere_actor)
    renderer.AddActor(sphere_actor)

# Set the position and focal point of the camera
camera = renderer.GetActiveCamera()
camera.SetPosition(0, 0, 300)  # Move the camera higher
camera.SetFocalPoint(0, 0, 0)  # Set the focal point to the cube(microphone)

renderWindow = vtkRenderWindow()
renderWindow.AddRenderer(renderer)

renderWindow.SetSize(1920, 1300)

renderWindowInteractor = vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

w2i = vtkWindowToImageFilter()
w2i.SetInput(renderWindow)

movieWriter = vtkOggTheoraWriter()
movieWriter.SetInputConnection(w2i.GetOutputPort())
movieWriter.SetFileName("animation.ogv")
movieWriter.Start()

# create animation callback
callback = afm.AnimationCallback(vector_field_actors=vector_field_actors, microphone_object=microphone,
        microphone_actor=microphone_actor, iren=renderWindowInteractor, w2i=w2i,
        movieWriter=movieWriter, num_steps=num_steps,renderer=renderer, camera_distance=30, sample_rate=sampling_rate)

renderWindowInteractor.AddObserver('TimerEvent', callback.execute)

timer_id = renderWindowInteractor.CreateRepeatingTimer(time_interval)
callback.timerId = timer_id

renderWindow.Render()
renderWindowInteractor.Initialize()
renderWindowInteractor.Start()

movieWriter.End()

audio_path = 'audio.wav'


wav.write(audio_path, sampling_rate, video_signal)

video_path = 'animation.ogv'
output_path = '3-sin.mp4'

if is_echo:
    tf.video_generator(video_path, echo_filename, output_path, is_echo=True)
else:
    tf.video_generator(video_path,audio_path,output_path,is_echo=False)
















































