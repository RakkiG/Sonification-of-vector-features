# --- original bubble model
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

surface_tension = 0.0728
nu = 1e-6
RHO_WATER = 998.

reader = vtk.vtkXMLImageDataReader()

# filename = '/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/vector_field_process/vector_field_complex.vti'
filename = '/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/vector_field_process5/vector_field_complex_critical_points.vti'

reader.SetFileName(filename)
reader.Update()

image_data = reader.GetOutput()

critical_points, jacobians, eigenvalues_list, eigenvectors_list = tf.extract_critical_points_and_jacobians(image_data)

indices_to_remove = {0,1,2,3,8,   13,14,16,   20,  23, 24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51}

critical_points = [point for i, point in enumerate(critical_points) if i not in indices_to_remove]
jacobians = [jacobian for i, jacobian in enumerate(jacobians) if i not in indices_to_remove]
eigenvalues_list = [eigenvalues for i, eigenvalues in enumerate(eigenvalues_list) if i not in indices_to_remove]
eigenvectors_list = [eigenvectors for i, eigenvectors in enumerate(eigenvectors_list) if i not in indices_to_remove]
print("critical points:", critical_points)

def velocity_field(t, position, image_data):

    velocity = tf.get_interpolated_vector_at_point(image_data, position)
    return velocity

def integrate_position_over_time(image_data, initial_position, t_span):


    result = solve_ivp(velocity_field, t_span, initial_position, args=(image_data,), method='RK45')


    return result.y[:, -1]

def compute_selected_coefficients(stretched_radiuses_angles, theta_resolution,phi_resolution, selected_indices):

    coefficients = []

    delta_theta = np.pi / theta_resolution
    delta_phi = 2 * np.pi / phi_resolution

    for l, m in selected_indices:
        integral_result = 0.0

        for radius, theta, phi in stretched_radiuses_angles:

            P_theta_phi = radius

            # Y_l^m(θ,φ)
            Y_l_m = sph_harm(m, l, phi, theta)

            weight = np.sin(theta) * delta_theta * delta_phi

            integral_result += P_theta_phi * np.conjugate(Y_l_m) * weight
            # integral_result += P_theta_phi * weight

        coefficient = integral_result / len(stretched_radiuses_angles)
        coefficients.append(coefficient)

        # coefficients.append(integral_result)

    return coefficients


def cartesian_to_spherical(x, y, z):

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # polar angle
    phi = np.arctan2(y, x)    # azimuthal angle
    return r, theta, phi

# Flatten the eigenvalues list and find the maximum absolute value
all_eigenvalues = np.hstack(eigenvalues_list)
max_abs_value = np.max(np.abs(all_eigenvalues))

# Normalize the eigenvalues by dividing them by the maximum absolute value
normalized_eigenvalues_list = [eigenvalues / max_abs_value for eigenvalues in eigenvalues_list]


def classify_critical_point(eigenvalues):

    real_parts = np.real(eigenvalues)

    positive_count = np.sum(real_parts > 0)
    negative_count = np.sum(real_parts < 0)

    if positive_count == 3:
        return 'Source'
    elif negative_count == 3:
        return 'Sink'
    elif positive_count == 2 and negative_count == 1:
        return '1:2 saddle'
    elif positive_count == 1 and negative_count == 2:
        return '2:1 saddle'
    else:
        return 'Unknown'  #

def get_radius_for_eigenvalues(eigenvalues):
    radius = {
        'Source': 5.0 / 1000,
        'Sink': 5.0 / 1000,
        '1:2 Saddle': 5.0 / 1000,
        '2:1 Saddle': 5.0 / 1000
    }

    real_parts = np.real(eigenvalues)

    sorted_real_parts = np.sort(real_parts)[::-1]

    #Set point_type according to the values of sorted_real_parts
    r1 = sorted_real_parts[0]
    r2 = sorted_real_parts[1]
    r3 = sorted_real_parts[2]

    if r1<0:
        point_type = 'Sink'
    elif r1>0 and r2<0:
        point_type = '2:1 Saddle'
    elif r2>0 and r3<0:
        point_type = '1:2 Saddle'
    elif r3>0:
        point_type = 'Source'

    a = radius[point_type]

    print('radius[point_type]:',a)

    return radius[point_type]

def equation_for_n(n, f_n, sigma, rho, r0):

    return (n - 1) * (n + 1) * (n + 2) - (4 * np.pi ** 2 * f_n ** 2 * rho * r0 ** 3) / sigma

def calculate_n(f_n, sigma, rho, r0, initial_guess=2):

    n_solution = fsolve(equation_for_n, initial_guess, args=(f_n, sigma, rho, r0))
    return max(0, n_solution[0])  # 确保 n >= 1

def calculate_beta_n(n, nu, rho, r0):

    beta_n = (n + 2) * (2 * n + 1) * (nu / (rho * r0 ** 2))
    return beta_n

def calculate_beta_0(r0):
    return math.pi * sbs.f0(r0) * sbs.delta_tol(sbs.f0(r0))

def calculate_omega_b(r0):

    f0 = sbs.f0(r0)
    # beta_0 = calculate_beta_n(0,nu,RHO_WATER, r0)
    beta_0 = calculate_beta_0(r0)
    # omega_b = 2 * np.pi * np.sqrt(f0 ** 2 - beta_0 ** 2)
    # return

    omega_b = 2 * np.pi * np.sqrt(f0 ** 2 - beta_0 ** 2)
    return omega_b

def calculate_f_n(n,sigma,rho, r0):
    f_n = np.sqrt((1 / (4 * np.pi ** 2)) * (n - 1) * (n + 1) * (n + 2) * (sigma / (rho * r0 ** 3)))
    return f_n

def calculate_omega_n(f_n):
    omega_n = 2 * np.pi * f_n
    return omega_n

def omega_b(r0):
    omega_b = calculate_omega_b(r0)
    return omega_b


def calculate_p_n(t, n, sigma, r0, omega_n,l,c_n,omega_b,beta_n):

    p_n = (-1 / l) * ((n - 1) * (n + 2) * (4 * n - 1) / (2 * n + 1) * sigma * (c_n ** 2) / (r0 ** 2)) * \
          ((omega_n ** 2) / (np.sqrt((4 * (omega_n ** 2) - omega_b ** 2) ** 2 + (4 * beta_n * omega_n) ** 2))) * \
          np.exp(-beta_n * t) * np.cos(2 * omega_n * t)

    # p_n = (-1 / l) * (1 /sigma * (c_n ** 2) / (r0 ** 2)) * \
    #       ((omega_n ** 2) / (np.sqrt((4 * beta_n * omega_n) ** 2))) * \
    #       np.exp(-beta_n * t) * np.cos(2 * omega_n * t)

    # p_n = (-1 / l) * ((n - 1) * (n + 2) * (4 * n - 1) / (2 * n + 1) * sigma * (c_n ** 2) / (r0 ** 2)) * \
    #       ((omega_n ** 2) / np.sqrt((4 * beta_n * omega_n) ** 2)) * \
    #       np.exp(-beta_n * t) * np.cos(2 * omega_n * t)

    return p_n


# t = np.linspace(0, 0.1, 1000)  #

def calculate_fb(r0):
    omega_b = calculate_omega_b(r0)
    return omega_b / (2 * np.pi)

p_signals_bubble = []

sound_duration = 3  #
sampling_rate = 48000  #
t = np.linspace(0, sound_duration, int(sampling_rate * sound_duration), endpoint=False)
l = 1

freq_ns = []
beta_ns = []
bubble_samples_after_spherecal_all = []
for index, critical_point in enumerate(critical_points):
    # p_signals_bubble.append(np.zeros_like(t))
    bubble_samples_after_spherecal_all.append([])
    eigenvalues = eigenvalues_list[index]
    eigenvector = eigenvectors_list[index]

    bubble_samples_cartesian = tf.generate_spherical_coordinates_cartesian(0.005, 16, 16, critical_point)

    r0 = get_radius_for_eigenvalues(eigenvalues)
    # stretched_radiuses, stretched_angles = compute_stretched_radiuses_angles_one_critical_point(r0, index)
    f_b = calculate_fb(r0)
    f_n_around = 0.5 * f_b  # 设定 f_n ≈ 1/2 f

    n = int(calculate_n(f_n_around, surface_tension, RHO_WATER, r0))

    # selected_indices = [(n - 2, 0), (n - 1, 0), (n, 0), (n + 1, 0), (n + 2, 0), (n + 3, 0),(n+4,0),
    #                     (n+5,0), (n+6,0), (n+7,0), (n+8,0), (n+9,0)]

    selected_indices=[(n - 9, 0), (n - 8, 0), (n - 7, 0), (n - 6, 0), (n - 5, 0), (n - 4, 0), (n - 3, 0), (n - 2, 0), (n - 1, 0), (n, 0),
     (n + 1, 0), (n + 2, 0), (n + 3, 0), (n + 4, 0),
     (n + 5, 0), (n + 6, 0), (n + 7, 0), (n + 8, 0), (n + 9, 0)]

    bubble_samples_after = []
    bubble_samples_after_spherical = []
    selected_coefficients_list = []

    for j, bubble_sample_cartesian in enumerate(bubble_samples_cartesian):
        position_after_time = integrate_position_over_time(image_data, bubble_sample_cartesian, [0, 1])
        # bubble_samples_cartesian[j] = position_after_time
        bubble_samples_after.append(position_after_time)

        # Shift the position so the critical point becomes the origin
        shifted_x = position_after_time[0] - critical_point[0]
        shifted_y = position_after_time[1] - critical_point[1]
        shifted_z = position_after_time[2] - critical_point[2]

        # Convert the shifted Cartesian coordinates to spherical coordinates
        r, theta, phi = cartesian_to_spherical(shifted_x, shifted_y, shifted_z)

        bubble_samples_after_spherical.append((r, theta, phi))
        bubble_samples_after_spherecal_all[index].append(bubble_samples_after_spherical)

        #
    selected_coefficients_list = compute_selected_coefficients(bubble_samples_after_spherical, 16,16,
                                                                   selected_indices)

    freq_ns.append([])
    beta_ns.append([])
    for j,coefficient in enumerate(selected_coefficients_list):
        n_mode = selected_indices[j][0]
        f_n = calculate_f_n(n=n_mode, sigma=surface_tension,rho=RHO_WATER,r0=r0)
        omega_n = calculate_omega_n(f_n=f_n)

        freq_ns[index].append(omega_n)
        # omega_b=calculate_omega_b(r0)
        beta_n = calculate_beta_n(n=n_mode, nu=nu, rho=RHO_WATER, r0=r0)
        beta_ns[index].append(beta_n)

flat_freq_ns = [freq for sublist in freq_ns for freq in sublist]

lower_bound = 100*np.pi
upper_bound = 1200*np.pi

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

print(reshaped_mapped_freq_ns)

# reshaped_mapped_freq_ns =freq_ns

# Flatten the freq_ns array to avoid the issue with numpy's min/max operations
flat_beta_ns = [beta_n for sublist in beta_ns for beta_n in sublist]

lower_bound = 1
upper_bound = 1

# Apply the mapping to the flattened array
mapped_flat_beta_ns = map_arr(flat_beta_ns, lower_bound, upper_bound)

# Reshape back into original nested structure
reshaped_mapped_beta_ns = []
index = 0
for sublist in beta_ns:
    reshaped_mapped_beta_ns.append(mapped_flat_beta_ns[index:index + len(sublist)])
    index += len(sublist)

# reshaped_mapped_beta_ns = beta_ns

selected_coefficients_list_all = []
stretched_radiuses_all = []
stretched_angles_all = []

for i, critical_point in enumerate(critical_points):
    p_signals_bubble.append(np.zeros_like(t))
    eigenvalues = eigenvalues_list[i]
    eigenvector = eigenvectors_list[i]
    r0 = get_radius_for_eigenvalues(eigenvalues)

    f_b = calculate_fb(r0)
    f_n_around = 0.5 * f_b  #  f_n ≈ 1/2 f

    n = int(calculate_n(f_n_around, surface_tension, RHO_WATER, r0))

    selected_indices = [(n - 9, 0),(n - 8, 0),(n - 7, 0),(n - 6, 0),(n - 5, 0),(n-4,0),(n-3,0),(n - 2, 0), (n - 1, 0), (n, 0), (n + 1, 0), (n + 2, 0), (n + 3, 0), (n + 4, 0),
                        (n + 5, 0), (n + 6, 0), (n + 7, 0), (n + 8, 0), (n + 9, 0)]

    bubble_samples_cartesian = tf.generate_spherical_coordinates_cartesian(0.005, 16, 16, critical_point)

    bubble_samples_after = []
    bubble_samples_after_spherical = []
    selected_coefficients_list = []
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

        #
    selected_coefficients_list = compute_selected_coefficients(bubble_samples_after_spherical, 16, 16,
                                                               selected_indices)

    selected_coefficients_list_all.append(selected_coefficients_list)

    # freq_ns.append([])
    for j, coefficient in enumerate(selected_coefficients_list):
        n_mode = selected_indices[j][0]
        f_n = calculate_f_n(n=n_mode, sigma=surface_tension, rho=RHO_WATER, r0=r0)
        omega_n = calculate_omega_n(f_n=f_n)

        # freq_ns[index].append(f_n)
        omega_b = calculate_omega_b(r0)
        beta_n = calculate_beta_n(n=n_mode, nu=nu, rho=RHO_WATER, r0=r0)
        p_n = calculate_p_n(t=t, n=n_mode, sigma=surface_tension, r0=r0, omega_n=omega_n, l=l,
                            c_n=coefficient, omega_b=omega_b, beta_n=beta_n).real
        p_signals_bubble[i] += p_n
        p_signals_bubble[i] = p_signals_bubble[i] / np.max(p_signals_bubble[i])


average_coefficients = []
for coeff_list in selected_coefficients_list_all:
    #
    avg_coeff = np.mean([np.abs(coeff) for coeff in coeff_list])
    average_coefficients.append(avg_coeff)


def map_to_range(arr, lower_bound, upper_bound):
    min_val = np.min(arr)
    max_val = np.max(arr)

    def map_value(value):
        return lower_bound + (value - min_val) * (upper_bound - lower_bound) / (max_val - min_val)

    return [map_value(val) for val in arr]


mapped_average_coefficients = map_to_range(average_coefficients, 0.1, 1)

for i in range(len(p_signals_bubble)):
    p_signals_bubble[i] = p_signals_bubble[i] * mapped_average_coefficients[i]

critical_points_jacobians_data = {}

# Create the folder 'bubble_signal_2' if it does not exist
output_folder = 'original_model'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Save each signal in p_signals_bubble as a .wav file
# for k, p_signal in enumerate(p_signals_bubble_normalized):
for k, p_signal in enumerate(p_signals_bubble):
    filename = os.path.join(output_folder, f"bubble_signal_{k}.wav")

    p_signal_normalized = p_signal / np.max(np.abs(p_signal))
    p_signal_pcm = np.int16(p_signal_normalized * 32767)  #

    # Save the file with the desired sampling rate
    wavfile.write(filename, sampling_rate, p_signal_normalized)












































