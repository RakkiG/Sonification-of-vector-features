import math
import os
from scipy.io.wavfile import write
import numpy as np

# Sampling parameters
fs = 48000
duration = 0.8
t = np.linspace(0, duration, int(fs*duration),endpoint = False)

#define constants
SIGMA = 0.072
GAMMA = 1.4
GTH = 1.6e6
KSI = 0.1
RHO_WATER = 998.
PATM = 101325.
G = 9.81
#Assume the bubble is in 1m depth in water
h = 2.
p0 = PATM + RHO_WATER*G*h
CF = 1500. # the speed of sound in water
pi = math.pi
def delta_th(f0):
    delta_th = math.sqrt((9 * (GAMMA - 1)**2) / (4 * GTH) * f0)
    return delta_th

delta_rad = math.sqrt((3 * GAMMA * p0) / (RHO_WATER * CF**2))

# total damping
def delta_tol(f0):
    delta_tol = delta_th(f0) + delta_rad
    return delta_tol

# f0
def f0(r0):
    f0 = 3/r0
    return f0

# def f0( r0):
#     omega = math.sqrt(3 * GAMMA * PATM - 2 * SIGMA * r0) / (r0 * math.sqrt(RHO_WATER))
#     f = omega / 2 / math.pi
#     return f

# beta_0
def beta_0(r0):
    beta_0 = pi*f0(r0)*delta_tol(f0(r0))
    return beta_0
def f_t(r0,t):
    f_t = f0(r0)*(1+KSI*beta_0(r0)*t)
    return f_t

def p_t(r0,t):
    p_t = SIGMA*r0*np.sin(2*math.pi*f_t(r0,t)*t)*np.exp(-beta_0(r0)*t)
    return p_t

def p_t_non_at(r0,t):
    p_t = SIGMA*r0*np.sin(2*math.pi*f_t(r0,1)*t)
    return p_t

# Assume a bubble radius
r0 = 5/1000

# Generate the sound signal
sound_signal = p_t(r0, t)

# Normalize the sound signal to the range of int16
sound_signal = sound_signal / np.max(np.abs(sound_signal))
sound_signal *= 32767
sound_signal = sound_signal.astype(np.int16)

# 创建“spherical bubble sounds”文件夹
output_dir = "spherical bubble sounds"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#Save the sound signal to a file
output_path = os.path.join(output_dir, f'sound_signal_{r0}.wav')
write(output_path, fs, sound_signal)
































