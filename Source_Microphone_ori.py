# for testing of echo
from scipy.io.wavfile import write

import numpy as np

class Source:
    def __init__(self,signal, positions, sample_rate, timeline,label,initial_points_pos, triangles):
        self.signal = signal
        self.positions = positions
        self.sample_rate = sample_rate
        self.timeline = timeline
        self.label = label
        self.initial_points_pos = initial_points_pos
        self.triangles = triangles


    def addTrajectory(self, trajectory):
        self.trajectory = trajectory

class Microphone:
    def __init__(self,signal, positions, sample_rate, timeline,label,initial_points_pos):
        self.signal = signal
        self.positions = positions
        self.sample_rate = sample_rate
        self.timeline = timeline
        self.label = label
        self.initial_points_pos = initial_points_pos

    def addTrajectory(self, trajectory):
        self.trajectory = trajectory