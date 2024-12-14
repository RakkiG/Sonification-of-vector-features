from scipy.io.wavfile import write

import numpy as np

class Source:
    def __init__(self,signal, positions, sample_rate, timeline,label,initial_pos,radius,jacobian_eigenvalues):
        self.signal = signal
        self.positions = positions
        self.sample_rate = sample_rate
        self.timeline = timeline
        self.label = label
        self.initial_pos = initial_pos
        self.radius = radius
        self.jacobian_eigenvalues = jacobian_eigenvalues


class Microphone:
    def __init__(self,label,initial_pos,trajectory):
        self.label = label
        self.initial_pos = initial_pos
        self.trajectory = trajectory


