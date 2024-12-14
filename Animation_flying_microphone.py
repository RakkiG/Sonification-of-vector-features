import  Tool_Functions as tf
import numpy as np
import vtk

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))


class AnimationCallback:
    def __init__(self, vector_field_actors, microphone_object, microphone_actor,iren, w2i, movieWriter,
                num_steps,renderer,camera_distance,sample_rate):

        self.microphone_object = microphone_object
        self.iren = iren
        self.w2i = w2i
        self.movieWriter = movieWriter
        self.vector_field_actors = vector_field_actors
        self.microphone_actor = microphone_actor
        self.num_steps = num_steps
        self.current_step = 0
        self.renderer = renderer
        self.camera_distance = camera_distance
        # self.critical_points = critical_points
        # self.sphere_radius = sphere_radius
        # self.critical_points_signals = critical_points_signals
        self.sample_rate = sample_rate
        # self.time_interval = time_interval
        # self.source_actors_initial_positions = [actor.GetPosition() for actor in self.vector_field_actors]


    def execute(self, obj, event):
        if self.current_step < self.num_steps:
            # update microphone's position
            microphone_position = self.microphone_object.trajectory[self.current_step]
            self.microphone_actor.SetPosition(microphone_position)

            #Update camera to follow the microphone
            camera = self.renderer.GetActiveCamera()
            # Calculate the camera position (opposite direction of movement)
            camera_position = [
                microphone_position[0] - self.camera_distance,
                microphone_position[1] - self.camera_distance,
                microphone_position[2] + self.camera_distance,
            ]

            camera.SetPosition(camera_position)
            camera.SetFocalPoint(microphone_position)

            # Update and write the frame
            self.w2i.Modified()
            self.w2i.Update()
            self.movieWriter.Write()

            # Render and increment time
            obj.GetRenderWindow().Render()
            self.current_step += 1
        else:
            self.iren.DestroyTimer(self.timerId)





















