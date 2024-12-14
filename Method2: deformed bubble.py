# Method 2 non-spherical bubbles deformation
# Frequency linear interpolation
import math

import vtk
import numpy as np
from scipy.io import wavfile

import Tool_Functions as tf
from scipy.special import sph_harm
from numpy.linalg import lstsq
import spherical_bubble_sound as sbs
from scipy.optimize import fsolve
import os
from scipy.io.wavfile import write
from scipy.integrate import solve_ivp
from scipy.signal import hilbert, chirp

surface_tension = 0.0728
nu = 1e-6    # Kinematic viscosity, the kinematic viscosity of water is about 1e-6 m^2/s
RHO_WATER = 998.

reader = vtk.vtkXMLImageDataReader()

# filename = '/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/vector_field_process4/vector_field_complex.vti'
filename_vec = '/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/vector_field_process5/vector_field_complex_critical_points.vti'

reader.SetFileName(filename_vec)
reader.Update()

image_data = reader.GetOutput()

critical_points, jacobians, eigenvalues_list, eigenvectors_list = tf.extract_critical_points_and_jacobians(image_data)

# Define the index to be removed
indices_to_remove = {0,1,2,3,8,11, 13,14,16,   20, 24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51}


critical_points = [point for i, point in enumerate(critical_points) if i not in indices_to_remove]
jacobians = [jacobian for i, jacobian in enumerate(jacobians) if i not in indices_to_remove]
eigenvalues_list = [eigenvalues for i, eigenvalues in enumerate(eigenvalues_list) if i not in indices_to_remove]
eigenvectors_list = [eigenvectors for i, eigenvectors in enumerate(eigenvectors_list) if i not in indices_to_remove]

def velocity_field(t, position, image_data):
    """
    Returns the velocity vector at a position
    t: current time
    position: current particle‘s pos (x, y, z)
    """
    velocity = tf.get_interpolated_vector_at_point(image_data, position)
    return velocity

def integrate_position_over_time(image_data, initial_position, t_span):
    """
By integrating the velocity field, find the final position of the particle after a period of time
initial_position: initial pos of the particle (x, y, z)
t_span: time interval (such as [0, t])
    """

    # integrating the velocity field
    result = solve_ivp(velocity_field, t_span, initial_position, args=(image_data,), method='RK45')

    # return finial pos of the final time point
    return result.y[:, -1]


modes = tf.get_modes_for_eigenvalues(eigenvalues_list)

def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (r, theta, phi).
    r: radius
    theta: polar angle (0 <= theta <= pi)
    phi: azimuthal angle (0 <= phi < 2pi)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # polar angle
    phi = np.arctan2(y, x)    # azimuthal angle
    return r, theta, phi

def equation_for_n(n, f_n, sigma, rho, r0):
    """
Calculate the roots of the equation f_n^2 = (1 / (4 * pi^2)) * (n-1) * (n+1) * (n+2) * (sigma / (rho * r0^3))
n : the order of the spherical harmonics (n)
f_n : the given value of f_n
sigma : surface tension (surface tension), unit: N/m
rho : liquid density (density), kg/m^3
r0 : initial radius of the bubble,  m

Return:The result of the equation, when it is zero, it is the root
    """
    return (n - 1) * (n + 1) * (n + 2) - (4 * np.pi ** 2 * f_n ** 2 * rho * r0 ** 3) / sigma

def calculate_n(f_n, sigma, rho, r0, initial_guess=2):

    #Calculate the value of n

    n_solution = fsolve(equation_for_n, initial_guess, args=(f_n, sigma, rho, r0))
    return max(0, n_solution[0])  # 确保 n >= 1

def calculate_beta_n(n, nu, rho, r0):

    # beta_n

    beta_n = (n + 2) * (2 * n + 1) * (nu / (rho * r0 ** 2))
    return beta_n

def calculate_beta_0(r0):
    return math.pi * sbs.f0(r0) * sbs.delta_tol(sbs.f0(r0))

def calculate_omega_b(r0):
    #ωb
    f0 = sbs.f0(r0)
    beta_0 = calculate_beta_0(r0)

    omega_b = 2 * np.pi * np.sqrt(f0 ** 2 - beta_0 ** 2)
    return omega_b

def calculate_f_n(n,sigma,rho, r0):
    f_n = np.sqrt((1 / (4 * np.pi ** 2)) * (n - 1) * (n + 1) * (n + 2) * (sigma / (rho * r0 ** 3)))
    return f_n

def calculate_omega_n(f_n):
    omega_n = np.pi * f_n
    return omega_n

def calculate_p_n( t,r0, omega_n):
    # betan = calculate_beta_n(n=15, nu=nu, rho=RHO_WATER, r0=r0)
    betan = 0.8
    t = 0.6*t
    p_n = np.exp(-betan * t) * np.cos(2 * omega_n * t)

    return p_n


def find_last_nonzero(signal):
    #Find the index of the last non-zero element in a signal
    nonzero_indices = np.nonzero(signal)[0]
    if len(nonzero_indices) == 0:
        return None
    return nonzero_indices[-1]

def calculate_fb(r0):
    omega_b = calculate_omega_b(r0)
    return omega_b / (2 * np.pi)

p_signals_bubble = []

sound_duration = 3
sampling_rate = 48000
t = np.linspace(0, sound_duration, int(sampling_rate * sound_duration), endpoint=False)
l = 1

neutral_bubble_radius = 0.005

#freq_n
freq_ns = []
beta_ns = []
bubble_samples_after_spherical_all = []
for index, critical_point in enumerate(critical_points):
    # p_signals_bubble.append(np.zeros_like(t))
    bubble_samples_after_spherical_all.append([])
    eigenvalues = eigenvalues_list[index]
    eigenvector = eigenvectors_list[index]

    bubble_samples_cartesian = tf.generate_spherical_coordinates_cartesian(0.005, 16, 16, critical_point)

    r0 = neutral_bubble_radius
    # stretched_radiuses, stretched_angles = compute_stretched_radiuses_angles_one_critical_point(r0, index)
    f_b = calculate_fb(r0)
    f_n_around = 0.5 * f_b  #  f_n ≈ 1/2 f

    n = int(calculate_n(f_n_around, surface_tension, RHO_WATER, r0))

    selected_indices = []

    modes_n = modes[index]
    for a in range(3):
        mode_n = modes_n[a]
        selected_indices.append((mode_n, 0))

    selected_coefficients_list = [1, 1, 1]
    freq_ns.append([])
    beta_ns.append([])
    for j, coefficient in enumerate(selected_coefficients_list):
        n_mode = selected_indices[j][0]
        f_n = calculate_f_n(n=n_mode, sigma=surface_tension, rho=RHO_WATER, r0=r0)
        omega_n = calculate_omega_n(f_n=f_n)

        freq_ns[index].append(omega_n)

flat_freq_ns = [freq for sublist in freq_ns for freq in sublist]

lower_bound = 200*np.pi
upper_bound = 1000*np.pi

def map_arr(arr, lower_bound, upper_bound):
    min_val = np.min(arr)
    max_val = np.max(arr)

    def map_value(value):
        return lower_bound + (value - min_val) * (upper_bound - lower_bound) / (max_val - min_val)

    return [map_value(number) for number in arr]

# Apply the mapping to the flattened array
mapped_flat_freq_ns = map_arr(flat_freq_ns, lower_bound, upper_bound)

# Reshape back into original nested structure
reshaped_mapped_freq_ns = []
index = 0
for sublist in freq_ns:
    reshaped_mapped_freq_ns.append(mapped_flat_freq_ns[index:index + len(sublist)])
    index += len(sublist)

index = 0

selected_coefficients_list_all = []

for i, critical_point in enumerate(critical_points):
    # signal_length = len(t)+4*48000
    p_signals_bubble.append([])
    # For each critical point, calculate the spherical coordinates of all sampling points on the bubble
    # with a radius of 0.005 and this critical point as the center of the sphere
    bubble_samples_cartesian = tf.generate_spherical_coordinates_cartesian(0.005, 16, 16, critical_point)

    bubble_samples_after = []
    bubble_samples_after_spherical = []

    for j, bubble_sample_cartesian in enumerate(bubble_samples_cartesian):
        position_after_time = integrate_position_over_time(image_data, bubble_sample_cartesian, [0, 0.1])
        # bubble_samples_cartesian[j] = position_after_time
        bubble_samples_after.append(position_after_time)

        # Shift the position so the critical point becomes the origin
        shifted_x = position_after_time[0] - critical_point[0]
        shifted_y = position_after_time[1] - critical_point[1]
        shifted_z = position_after_time[2] - critical_point[2]

        # Convert the shifted Cartesian coordinates to spherical coordinates
        r, theta, phi = cartesian_to_spherical(shifted_x, shifted_y, shifted_z)

        bubble_samples_after_spherical.append((r, theta, phi))

    r_values = [sample[0] * np.sin(sample[1]) for sample in bubble_samples_after_spherical]
    r0_new = np.mean(r_values)

    selected_indices = []

    modes_n = modes[i]
    for a in range(3):
        mode_n = modes_n[a]
        selected_indices.append((mode_n, 0))

    selected_coefficients_list = [1,1,1]

    selected_coefficients_list_all.append(selected_coefficients_list)

    # calculate the neutral bubble signal
    neutral_modes = 16

    f_n_neutral = calculate_f_n(n=16, sigma=surface_tension, rho=RHO_WATER, r0=neutral_bubble_radius)
    omega_n_neutral = calculate_omega_n(f_n=f_n_neutral)

    neu_sound_duration = 0.5
    sampling_rate = 48000
    neu_t = np.linspace(0, neu_sound_duration, int(sampling_rate * neu_sound_duration), endpoint=False)


    p_n_neutral = calculate_p_n(t=neu_t, r0=neutral_bubble_radius, omega_n=omega_n_neutral).real

    # freq_ns.append([])
    for j, coefficient in enumerate(selected_coefficients_list):
        n_mode = int(selected_indices[j][0])
        # f_n = calculate_f_n(n=n_mode, sigma=surface_tension, rho=RHO_WATER, r0=r0_new)
        f_n = calculate_f_n(n=n_mode, sigma=surface_tension, rho=RHO_WATER, r0=neutral_bubble_radius)
        omega_n = calculate_omega_n(f_n=f_n)

        # freq_ns[index].append(f_n)
        omega_b = calculate_omega_b(r0_new)
        # beta_n = calculate_beta_n(n=n_mode, nu=nu, rho=RHO_WATER, r0=r0_new)
        p_n = calculate_p_n(t=t, r0=neutral_bubble_radius, omega_n=omega_n).real
        # Generate unattenuated neutral bubble sound
        # t_spherical = np.linspace(0, 0.5, int(48000 * 1), endpoint=False)
        # p_neutral = sbs.p_t_non_at(r0=neutral_bubble_radius,t=t_spherical)
        p_neutral_normalized = p_n_neutral / np.max(np.abs(p_n_neutral))  # 归一化到[-1, 1]范围
        p_n_normalized = p_n / np.max(np.abs(p_n))  # 归一化到[-1, 1]范围

        # f_neutral = sbs.f_t(r0=neutral_bubble_radius,t=1)
        # f_neutral = omega_n_neutral/2*math.pi
        # f_mode_n = omega_n/2*math.pi

        analytic_signal_neutral = hilbert(p_neutral_normalized)
        analytic_signal_p_n = hilbert(p_n_normalized)
        instantaneous_phase_neu = np.unwrap(np.angle(analytic_signal_neutral))
        instantaneous_phase_p_n = np.unwrap(np.angle(analytic_signal_p_n))
        instantaneous_frequency_neu = (np.diff(instantaneous_phase_neu) /
                                   (2.0 * np.pi) * 48000)
        instantaneous_frequency_p_n = (np.diff(instantaneous_phase_p_n) /
                                       (2.0 * np.pi) * 48000)

        duration_transition = 0.5

        #We need to get 2/3 length of a's f and 1/3 of b's f
        d_neu_2_3 = len(p_neutral_normalized)*2//3
        d_p_n_1_3 = len(p_n_normalized)//3

        transition_length = duration_transition * sampling_rate

        first_third_length_transi = transition_length // 20
        last_third_start_transi = transition_length * 19 // 20

        # d_neu_2_3 = int(len(p_neutral_normalized)*2//3 + first_third_length_transi)
        # d_p_n_1_3 = int(len(p_n_normalized)//3 + last_third_start_transi)

        freq_neu_2_3 = instantaneous_frequency_neu[d_neu_2_3]
        freq_p_n_1_3 = instantaneous_frequency_p_n[d_p_n_1_3]

        # Generate sound for the mode
        p_n = tf.smooth_transition_signal(signal_a=p_neutral_normalized, signal_b=p_n_normalized,
        freq_a=freq_neu_2_3, freq_b=freq_p_n_1_3, sampling_rate=48000,
        duration_transition=duration_transition)

        if len(p_signals_bubble[i]) < len(p_n):
            extension_length = len(p_n) - len(p_signals_bubble[i])
            p_signals_bubble[i] = np.concatenate((p_signals_bubble[i], np.zeros(extension_length)))

        p_signals_bubble[i] += p_n

        if np.max(np.abs(p_signals_bubble[i])) > 0:  # divide by 0 error
            p_signals_bubble[i] = p_signals_bubble[i] / np.max(np.abs(p_signals_bubble[i]))

# Create the folder 'bubble_signal_2' if it does not exist
output_folder = 'Method2_deformed_bubble_repeated_signals'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

_, signal_neutral = wavfile.read('sound_signal_0.005_3.wav')

signal_neutral = signal_neutral / np.max(np.abs(signal_neutral))

# Save each signal in p_signals_bubble as a .wav file
# for k, p_signal in enumerate(p_signals_bubble_normalized):
for k, p_signal in enumerate(p_signals_bubble):
    eigens = eigenvalues_list[k]
    eigenvalue_str = "_".join([str(int(np.round(val))) for val in eigens])
    # output_file = os.path.join(output_dir, f"source{i}_{eigens}_bubble_sound.wav")
    # output_file = f"vortex_signal_({eigenvalue_str}).wav"
    filename = os.path.join(output_folder, f"bubble{k}_({eigenvalue_str}).wav")
    # wavfile.write(os.path.join(output_folder, f"source_{i}_{eigens}_3sins.wav"), 48000, signal)

    p_signal_normalized = p_signal / np.max(np.abs(p_signal))  # 归一化到[-1, 1]范围

    # Save the file with the desired sampling rate
    wavfile.write(filename, sampling_rate, p_signal_normalized)




























