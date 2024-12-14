# Method 1: Positive eigenvalues match 200-400, negative eigenvalues match 800-1000
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

# filename = '/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/vector_field_process5/vector_field_complex_single.vti'
# filename = '/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/vector_field_process5/vector_field_complex.vti'
filename_vec = '/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/vector_field_process5/vector_field_complex_critical_points.vti'


reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(filename_vec)
reader.Update()

image_data = reader.GetOutput()
critical_points, jacobians, eigenvalues_list, eigenvectors_list = tf.extract_critical_points_and_jacobians(image_data)


########################################################################
# Define and filter the indexes want to remove
indices_to_remove = {0,1,2,3,8,11, 13,14,16,   20, 24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51}

critical_points = [point for i, point in enumerate(critical_points) if i not in indices_to_remove]
jacobians = [jacobian for i, jacobian in enumerate(jacobians) if i not in indices_to_remove]
eigenvalues_list = [eigenvalues for i, eigenvalues in enumerate(eigenvalues_list) if i not in indices_to_remove]
eigenvectors_list = [eigenvectors for i, eigenvectors in enumerate(eigenvectors_list) if i not in indices_to_remove]
########################################################################
# Because this method is only about critical points,
# we remove the vortex coreline and just save the critical points signal to a folder.
# There is no need to create sources, just generate and save them directly
# time_interval = 40  # 40 milliseconds
# sound_duration = time_interval/1000 * num_steps
sound_duration = 10
sampling_rate = 48000
# create timeline
t = np.linspace(0, sound_duration, math.ceil(sampling_rate * sound_duration), endpoint=False)
output_folder = "Method_1_3_sins"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
#Create a frequency array to store the frequency of each eigenvalue map
freqs = tf.get_frequencies_for_eigenvalues2(eigenvalues_list)
for i in range(len(critical_points)):
    # eigenvalues = eigenvalues_list[i]
    # get the 3 freqs of the current eigenvalues set
    # freqs = tf.get_frequencies_for_eigenvalues(eigenvalues)
    freqens = freqs[i][0], freqs[i][1], freqs[i][2]

    # Generate sin signal
    signal = np.sin(2 * np.pi * freqens[0] * t)+ np.sin(2 * np.pi * freqens[1] * t)+ np.sin(2 * np.pi * freqens[2] * t)

    # Normalize the signal
    signal = signal / np.max(np.abs(signal))

    #save signals as .wav to folder "Method 1: 3 sins"
    # wavfile.write(os.path.join(output_folder, f"source_{i}_3sins.wav"), 48000, signal)
    eigens = eigenvalues_list[i]
    eigenvalue_str = "_".join([str(int(np.round(val))) for val in eigens])
    # output_file = os.path.join(output_dir, f"source{i}_{eigens}_bubble_sound.wav")
    # output_file = f"vortex_signal_({eigenvalue_str}).wav"
    # filename = os.path.join(output_folder, f"bubble{k}_({eigenvalue_str}).wav")
    wavfile.write(os.path.join(output_folder, f"source_{i}_({eigenvalue_str})_3sins.wav"), 48000, signal)
########################################################################
# Generate sound for neutral critical point, whose 3 eigenvalues are all 0
eigenvalues_0 = [0,0,0]
freqs_0 = tf.get_frequencies_for_eigenvalues2(eigenvalues_0)
signal0 = np.sin(2 * np.pi * freqs_0[0] * t) + np.sin(2 * np.pi * freqs_0[1] * t) + np.sin(2 * np.pi * freqs_0[2] * t)

wavfile.write(os.path.join(output_folder, f"source_neutral_{eigenvalues_0}_3sins.wav"), 48000, signal0)
########################################################################



