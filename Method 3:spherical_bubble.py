# method 3: spherical bubble
import math
import os

import numpy as np
from scipy.io import wavfile
from vtkmodules.vtkIOOggTheora import vtkOggTheoraWriter
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor, vtkWindowToImageFilter

import Tool_Functions as tf
import vtk
import Source_Microphone
from scipy.interpolate import CubicHermiteSpline
import Animation_flying_microphone as afm
import scipy.io.wavfile as wav
import Doppler_effect_re_formular as doppler_effect
import Amplitude_distance2 as ad

import spherical_bubble_sound2 as sbs

# filename = '/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/vector_field_process/vector_field_complex.vti'
filename_vec = '/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/vector_field_process5/vector_field_complex_critical_points.vti'

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(filename_vec)
reader.Update()

image_data = reader.GetOutput()
critical_points, jacobians, eigenvalues_list, eigenvectors_list = tf.extract_critical_points_and_jacobians(image_data)


indices_to_remove = {0,1,2,3,8,11, 13,14,16,   20, 24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51}


critical_points = [point for i, point in enumerate(critical_points) if i not in indices_to_remove]
jacobians = [jacobian for i, jacobian in enumerate(jacobians) if i not in indices_to_remove]
eigenvalues_list = [eigenvalues for i, eigenvalues in enumerate(eigenvalues_list) if i not in indices_to_remove]
eigenvectors_list = [eigenvectors for i, eigenvectors in enumerate(eigenvectors_list) if i not in indices_to_remove]


#create a radius array, to store the radius of every eigenvalue map
radiuses = tf.get_radiuses_for_eigenvalues2(eigenvalues_list)
sampling_rate = 48000

critical_points_signals = []
output_dir = ("Method3_spherical_bubble_signals")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(len(critical_points)):
    eigenvalues = eigenvalues_list[i]

    radiuses_for_one = radiuses[i][0], radiuses[i][1], radiuses[i][2]

    beta_0s = []
    for k in range(3):
        beta_0s.append(sbs.beta_0(radiuses_for_one[k]))

    signal = (sbs.generate_spherical_bubble_sound(radiuses_for_one[0], beta_0s[0] / 13, sampling_rate, 3) +
              sbs.generate_spherical_bubble_sound(radiuses_for_one[1], beta_0s[1] / 13, sampling_rate, 3) +
              sbs.generate_spherical_bubble_sound(radiuses_for_one[2], beta_0s[2] / 13, sampling_rate, 3))


    signal = signal / np.max(np.abs(signal))
    eigens = eigenvalues_list[i]
    # filename = os.path.join(output_dir, f"source_{k}_{eigens}_3sins.wav")
    eigenvalue_str = "_".join([str(int(np.round(val))) for val in eigens])
    output_file = os.path.join(output_dir, f"source{i}_({eigenvalue_str})_bubble_sound.wav")
    # output_file = f"vortex_signal_({eigenvalue_str}).wav"
    wav.write(output_file, sampling_rate, signal.astype(np.float32))


















