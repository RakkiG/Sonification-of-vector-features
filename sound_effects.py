#The original experiment was in the scalar field, and now it is retained mainly for echo testing
import json
import math

import numpy as np
from vtkmodules.vtkIOOggTheora import vtkOggTheoraWriter
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor, vtkWindowToImageFilter

# define the number of position points for the source
import Tool_Functions as tf
from scipy.io.wavfile import write
import Animation_Generation_unstructured2 as AGu
import Source_Microphone_ori as Source_Microphone

import vtk

from scipy.io import wavfile
import pandas as pd

# Split the unstructured grid dataset into connected graphs and save each graph and .json file to the components folder,
# and get the number of connected components (number of sound sources)
output_dir = "components"
unstructured_dataset = "two_triangles.vtu"
# unstructured_dataset = "datasets/two_triangles.vtu"
config_file = "two_triangles_config.json"
position_file = "dataset.csv"

#Extract the connected parts of the unstructured grid and save the object
# and the corresponding json color configuration file to the /components folder
num_source = tf.get_all_connected_components(unstructured_dataset, output_dir,
                                             config_file)

# read the positions of all the points in the .vtu file
df = pd.read_csv(position_file)

initial_points_positions = {}

source_0th_positions = {}

all_positions = {}
all_positions['microphone'] = {}
all_positions['source'] = {}
time_array = []
# triangle_centers = {}
triangles = {}

# ----------------------Complete the both_positions array construction for all connected source objects (down)----------------------------------
for i in range(num_source):
    source_0th_positions[f'source{i}'] = []
    initial_points_positions[f'source{i}'] = {}
    vtu_path = f"{output_dir}/connected_component_{i}.vtu"
    with (open(f"{output_dir}/connected_component_{i}_config.json", 'r') as file): data = json.load(file)

    first_index = data['point_indices_in_original_grid'][0]

    # ---------------Complete initialization of initial_points_positions (down)----------------
    # Find the initial coordinates corresponding to this first_index in the original unstructured_dataset
    initial_points_positions[f'source{i}']['point0'] = tf.get_point_coordinate_from_vtu(
        unstructured_dataset, first_index)

    # Calculate the centroid coordinates of this f"{output_dir}/connected_component_{i}.vtu"
    initial_points_positions[f'source{i}']['centroid'] = tf.cal_centroid_unstructured(
        vtu_path)

    # Divide the object into small triangles and get the list of their triangle centers
    triangles[f'source{i}'] = tf.get_triangles_with_centers(vtu_path,40)
    if triangles[f'source{i}'] is not None:
        initial_points_positions[f'source{i}']['triangle'] = triangles[f'source{i}']
        # for k in range(len(triangle_centers[f'source{i}'])):
        #     initial_points_positions[f'source{i}'][f'triangle_center{k}'] = triangle_centers[f'source{i}'][k]

    # ---------------Complete the construction of both_positions array for this source object (down)----------------
    # Get the position array of column 0 of each connected object
    # Construct column name
    column_names = f'point{first_index}'

    # Extract the data in the column where point0 is located and delete the rows containing None values
    column_data = df[column_names]


    source_0th_positions[f'source{i}'] = [eval(point) for point in column_data.dropna().tolist()]

    print(f"source_0th_positions['source{i}']:", source_0th_positions[f'source{i}'])

    time_array = df['time'].tolist()
    print("time_array:", time_array)

    all_positions['source'][f'source{i}'] = tf.generate_positions_for_all_points(
        initial_points_positions=initial_points_positions[f'source{i}'],
        zeroth_point_positions=source_0th_positions[f'source{i}'],
        time_array=time_array
    )
    # ---------------Complete the construction of both_positions array for this source object (up)----------------

    # -----------------------Complete the construction of both_positions array for all connected source objects (up)---------------------------------

    # -----------------------Complete the construction of both_positions array for two microphones (down)----------------------------------
initial_microphone_positions = {
    # 'microphone0':(0,0,0)
    # 'microphone0':(-13,-5,0),
    # 'microphone1':(13,-5,0)
# 'microphone0':(0,50,0)   #mic on top
'microphone0':(0,30,0)   #mic in the middle
}

microphone_positions = tf.generate_positions_for_microphone(initial_microphones_positions=initial_microphone_positions,
                                                            time_array=time_array)

for microphone_label, microphone_position in microphone_positions.items():
    all_positions['microphone'][microphone_label] = microphone_position
# -----------------------Complete the construction of both_positions arrays for both microphones (up)----------------------------------

# -----------------------Pick the sound for the sources and save the sound signal + sampling rate to an unprocessed_source_signals (down)----------------------------------
sampling_rate0 = 48000
ori_duration = 30
t = np.linspace(0, ori_duration, int(sampling_rate0 * ori_duration), endpoint=False)
samples = int(sampling_rate0 * ori_duration)  # 样本总数

freq1 = 600  # 频率1

signal0 = np.sin(2 * np.pi * freq1 * t)


unprocessed_source_signals = {}
unprocessed_source_signals['source0'] = {'signal': signal0, 'sample_rate': sampling_rate0}

unprocessed_source_signals['source1'] = {'signal': None, 'sample_rate': sampling_rate0}
unprocessed_source_signals['source2'] = {'signal': None, 'sample_rate': sampling_rate0}
# -----------------------Pick the sound for the sources and save the sound signal + sampling rate to an unprocessed_source_signals (up)--------------------

# ------------------------Create source and microphone objects (down)----------------------------------
source_objects = {}
microphone_objects = {}

for i in range(num_source):
    source_objects[f'source{i}'] = Source_Microphone.Source(signal=unprocessed_source_signals[f'source{i}']['signal'],
                                                    positions=all_positions['source'][f'source{i}'],
                                                    sample_rate=unprocessed_source_signals[f'source{i}']['sample_rate'],
                                                    timeline=t, label=f'source{i}',
                                                    initial_points_pos=initial_points_positions[f'source{i}'],
                                                    triangles = triangles[f'source{i}'])

for i in range(len(initial_microphone_positions)):
    microphone_objects[f'microphone{i}'] = Source_Microphone.Microphone(signal=None,
                                                            positions=all_positions['microphone'][f'microphone{i}'],
                                                            sample_rate=None, timeline=t,label=f'microphone{i}',
                                                 initial_points_pos=initial_microphone_positions[f'microphone{i}'])

# ------------------------Create source and microphone objects (up)----------------------------------

#Unify the sampling rate of all signals
tf.unified_sample_rate(source_objects)
for i in range(num_source):
    print("source_objects[f'source{i}'].sample_rate:", source_objects[f'source{i}'].sample_rate)

# Create VTK rendering pipeline
renderer = vtkRenderer()
renderer.SetBackground(0.96, 0.96, 0.86)

# Set the position and focal point of the camera
camera = renderer.GetActiveCamera()
camera.SetPosition(0, 0, 300)  # Move the camera higher
# camera.SetFocalPoint(0, 0, 0)  # Set the focal point to the cube(microphone)
camera.SetFocalPoint(2.5, 2.5, 2.5) # Set the camera focus to the center of the triangle
camera.SetViewUp(0, 1, 0)

camera.Elevation(60)

renderWindow = vtkRenderWindow()
renderWindow.AddRenderer(renderer)

renderWindow.SetSize(1920, 1080)

renderWindowInteractor = vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)
#-----------------------------------------Finish creating microphone and source actors (down)----------------------------------
# Create microphone actors and add to the list
microphone_actor0 = tf.create_actor(vtk.vtkSphereSource(), "microphone0")
microphone_actor1 = tf.create_actor(vtk.vtkSphereSource(), "microphone1")# Use cone to represent the microphone

# Set the color of the microphone to fluorescent green
fluorescent_green = [0.0, 1.0, 0.0]

microphone_actor0.GetProperty().SetColor(fluorescent_green)
microphone_actor1.GetProperty().SetColor(fluorescent_green)

microphone_actors = {
    'microphone0':microphone_actor0,
     # 'microphone1':microphone_actor1
}

source_actors = {}
for i in range(num_source):
    file_path = f"{output_dir}/connected_component_{i}.vtu"
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()

    source_actor = tf.create_actor(reader, "source" + str(i))
    source_actors["source" + str(i)] = source_actor

#-----------------------------------------Complete the creation of microphone and source actors (up)----------------------------------
#Add each actor to the scene
for object_label, actor in source_actors.items():
    number = object_label[6:]
    # Load color configuration from json
    with open(f"/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/components/connected_component_{number}_config.json", 'r') as file:
        data = json.load(file)

    tf.set_actor_color(actor, data['point_colors'])

    renderer.AddActor(actor)
for microphone_label,microphone_actor in microphone_actors.items():

    renderer.AddActor(microphone_actor)

trajectory_manager = tf.TrajectoryManager()

w2i = vtkWindowToImageFilter()
w2i.SetInput(renderWindow)

movieWriter = vtkOggTheoraWriter()
movieWriter.SetInputConnection(w2i.GetOutputPort())
movieWriter.SetFileName("animation_echo.ogv")
movieWriter.Start()

# The time interval between every 2 executions of the animation
time_interval = 40  # unit is ms

for object_label, source_object in source_objects.items():
    print(f"{object_label}.positions:", source_object.positions)


time_array_max = 0
for object_label, source_object in source_objects.items():
    try:
        current_max = source_object.positions["point0"][-1][0]
        if current_max > time_array_max:
            time_array_max = current_max
    except IndexError:
        continue

# Determine the number of animation steps by the maximum time in the trajectory list
num_steps = tf.divide_and_round_up(time_array_max, time_interval / 1000)
animation_duration = num_steps * time_interval  # unit is ms

tf.Unify_audio_len(source_objects, animation_duration)
callback = AGu.AnimationCallback(source_actors=source_actors, microphone_actors=microphone_actors, num_steps=num_steps,
                                 source_objects=source_objects, microphone_objects=microphone_objects,
                                 iren=renderWindowInteractor,
                                 trajectory_manager=trajectory_manager, w2i=w2i, movieWriter=movieWriter,
                                 time_interval=time_interval,
                                 animation_duration=animation_duration, initial_points_pos=initial_points_positions,
                                 microphone_positions=initial_microphone_positions,renderer = renderer)

renderWindowInteractor.AddObserver('TimerEvent', callback.execute)

timer_id = renderWindowInteractor.CreateRepeatingTimer(time_interval)
callback.timerId = timer_id

renderWindow.Render()
renderWindowInteractor.Initialize()
renderWindowInteractor.Start()

movieWriter.End()

for object_label, source_object in source_objects.items():
    source_object.addTrajectory(trajectory_manager.get_trajectory(object_label))
    #cut the length of the signals to be the largest time in the trajectory
    duration_of_signal,_,_ = source_object.trajectory['centroid'][-1]
    length_of_signal = math.ceil(48000 * duration_of_signal)
    if source_object.signal is None:
        continue
    source_object.signal = source_object.signal[:length_of_signal]
    source_object.timeline = source_object.timeline[:length_of_signal]

for microphone_label, microphone_object in microphone_objects.items():
    microphone_object.addTrajectory(trajectory_manager.get_trajectory(microphone_label))

average_signals = tf.result_generation(microphone_objects = microphone_objects,source_objects = source_objects,
                                       is_amplitude_distance=False, is_doppler = True, is_echo = True)

audio_path1 = None

if 'microphone1' in average_signals:
    # If 'microphone1' exists, merge to stereo
    left_channel = np.int16(average_signals['microphone0'])
    right_channel = np.int16(average_signals['microphone1'])
    tf.combine_to_stereo(left_channel=average_signals['microphone0'], right_channel=average_signals['microphone1'],
                        sample_rate=48000,output_filename='stereo_unstructured.wav')
    audio_path1 = "stereo_unstructured.wav"
else:
    # If 'microphone1' does not exist, just write 'microphone0' signal to file
    write("mono_unstructured.wav", 48000, average_signals['microphone0'])
    audio_path1 = "mono_unstructured.wav"

video_path = 'animation_echo.ogv'
output_path1 = 'stereo_unstructured.mp4'

tf.video_generator(video_path,audio_path1,output_path1,is_echo=True)













