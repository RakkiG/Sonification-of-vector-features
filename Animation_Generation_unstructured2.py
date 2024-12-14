# Animation generation code for three sound effects in scalar field for echo testing
import Tool_Functions as tf


class AnimationCallback:
    def __init__(self, source_objects, source_actors, microphone_objects, microphone_actors,num_steps,iren, trajectory_manager, w2i, movieWriter,
                 time_interval, animation_duration,initial_points_pos,microphone_positions, renderer):
        self.source_objects = source_objects
        self.microphone_objects = microphone_objects
        self.iren = iren
        self.trajectory_manager = trajectory_manager
        self.w2i = w2i
        self.movieWriter = movieWriter
        self.time_interval = time_interval
        self.animation_duration = animation_duration
        self.current_time = 0
        self.source_actors = source_actors
        self.microphone_actors = microphone_actors
        self.num_steps = num_steps
        self.initial_points_pos = initial_points_pos
        self.microphone_positions = microphone_positions
        self.experiment_interpolation = []
        self.renderer = renderer

    def execute(self, obj, event):
        #To save the last position to cal the difference between two positions to
        # set the position of the actor
        if self.current_time < self.animation_duration:
            for object_label,source_object in self.source_objects.items():
                # The first last_pos is the initial position for 0th point
                actor_property = self.source_actors[object_label].GetProperty()
                # If source_object.positions["point0"] is empty
                if not source_object.positions["point0"]:
                    # Set the transparency to 0 to make the object completely transparent
                    actor_property.SetOpacity(0)
                    # Add a track point containing the None position to the track manager
                    for point_label, point_positions in source_object.positions.items():
                        if point_label == 'triangle':
                            for triangle_id, triangle_dict in source_object.positions[point_label].items():
                                for triangle_label, triangle_point_positions in triangle_dict.items():
                                    if triangle_label == 'center':
                                        for i in range(len(triangle_point_positions)):
                                            self.trajectory_manager.add_trajectory_point(object_label=object_label,
                                                point_label=point_label, time=self.current_time / 1000, position=None,
                                                triangle_id = triangle_id, center=True)

                                    if triangle_label == 'points':
                                        for triangle_point_label, positions_triangle_points in triangle_point_positions.items():
                                            self.trajectory_manager.add_trajectory_point(object_label=object_label,
                                                point_label=point_label, time=self.current_time / 1000, position=None,
                                                triangle_id = triangle_id, triangle_point_id=triangle_point_label)
                        else:
                            self.trajectory_manager.add_trajectory_point(object_label=object_label,point_label=point_label, time=self.current_time / 1000,
                                 position=None)
                    continue
                # The first last_pos is the initial position for 0th point
                last_pos = source_object.initial_points_pos['point0']

                # If the current time exceeds the last time point of the object's time_array
                if self.current_time/1000 > source_object.positions["point0"][-1][0]:
                    #Set the transparency to 0 to make the object completely transparent
                    actor_property.SetOpacity(0)

                    # Add a track point to the track manager with a position of None
                    for point_label, point_positions in source_object.positions.items():
                        if point_label == 'triangle':
                            for triangle_id, triangle_dict in source_object.positions[point_label].items():
                                for triangle_label, triangle_point_positions in triangle_dict.items():
                                    if triangle_label == 'center':
                                        for i in range(len(triangle_point_positions)):
                                            self.trajectory_manager.add_trajectory_point(object_label=object_label,
                                                                                         point_label=point_label,
                                                                                         time=self.current_time / 1000,
                                                                                         position=None,
                                                                                         triangle_id=triangle_id,
                                                                                         center=True)

                                    if triangle_label == 'points':
                                        for triangle_point_label, positions_triangle_points in triangle_point_positions.items():
                                            self.trajectory_manager.add_trajectory_point(object_label=object_label,
                                                                                         point_label=point_label,
                                                                                         time=self.current_time / 1000,
                                                                                         position=None,
                                                                                         triangle_id=triangle_id,
                                                                                         triangle_point_id=triangle_point_label)
                        else:
                            self.trajectory_manager.add_trajectory_point(object_label=object_label,
                                                                         point_label=point_label,
                                                                         time=self.current_time / 1000,
                                                                         position=None)
                else:
                    # Make sure the object is visible if it was previously set to transparent
                    actor_property.SetOpacity(1)
                    found_position_for_current_time = False
                    for point_label, point_positions in source_object.positions.items():
                        if point_label == 'triangle':
                            for triangle_id, triangle_dict in source_object.positions[point_label].items():
                                for triangle_label,triangle_point_positions in triangle_dict.items():
                                    if triangle_label == 'center':
                                        for i in range(len(triangle_point_positions)):
                                            if triangle_point_positions[i][0] <= self.current_time / 1000 <triangle_point_positions[i+1][0]:
                                                found_position_for_current_time = True
                                                time_start = triangle_point_positions[i][0]
                                                time_end = triangle_point_positions[i+1][0]

                                                pos_start = triangle_point_positions[i][1]
                                                pos_end = triangle_point_positions[i+1][1]

                                                interpolated_pos = tf.linear_interpolation(time_start, pos_start, time_end, pos_end,
                                                                                            self.current_time / 1000)

                                                self.trajectory_manager.add_trajectory_point(object_label=object_label,
                                                             point_label=point_label,
                                                             time=self.current_time/1000,
                                                             position=interpolated_pos,
                                                             triangle_id=triangle_id,
                                                             center=True)

                                    if triangle_label == 'points':
                                        for triangle_point_label, positions_triangle_points in triangle_point_positions.items():
                                            for i in range(len(positions_triangle_points)-1):
                                                if positions_triangle_points[i][0] <= self.current_time/1000 < positions_triangle_points[i+1][0]:
                                                    found_position_for_current_time = True
                                                    time_start = positions_triangle_points[i][0]
                                                    time_end = positions_triangle_points[i+1][0]

                                                    pos_start = positions_triangle_points[i][1]
                                                    pos_end = positions_triangle_points[i+1][1]

                                                    interpolated_pos = tf.linear_interpolation(time_start, pos_start, time_end, pos_end,
                                                                                                self.current_time / 1000)

                                                    self.trajectory_manager.add_trajectory_point(object_label=object_label,
                                                                 point_label=point_label,
                                                                 time=self.current_time/1000,
                                                                 position=interpolated_pos,
                                                                 triangle_id=triangle_id,
                                                                 triangle_point_id=triangle_point_label)
                        else:
                            for i in range(len(source_object.positions[point_label])-1):
                                if source_object.positions[point_label][i][0] <= self.current_time/1000 < source_object.positions[point_label][i+1][0]:
                                    found_position_for_current_time = True
                                    time_start = source_object.positions[point_label][i][0]
                                    time_end = source_object.positions[point_label][i+1][0]

                                    pos_start = source_object.positions[point_label][i][1]
                                    pos_end = source_object.positions[point_label][i+1][1]

                                    interpolated_pos = tf.linear_interpolation(time_start, pos_start, time_end, pos_end,
                                                                                self.current_time / 1000)

                                    self.trajectory_manager.add_trajectory_point(object_label=object_label,
                                                                                 point_label=point_label,
                                                                                 time=self.current_time/1000,
                                                                                 position=interpolated_pos)
                                    # update actor's position
                                    if point_label == 'point0':
                                        self.source_actors[object_label].SetPosition(interpolated_pos)
                                        diff = tuple(a - b for a, b in zip(interpolated_pos, last_pos))
                                        current_actor_position = self.source_actors[object_label].GetPosition()
                                        new_position = tuple(a + b for a, b in zip(current_actor_position, diff))
                                        self.source_actors[object_label].SetPosition(new_position)
                                    # Save each interpolated_pos, in order to find the difference between the two times,
                                    # so that the position of the actor can be adjusted accordingly
                                    last_pos = interpolated_pos
                                    break
                        if not found_position_for_current_time:
                            actor_property.SetOpacity(0)
                            # The object has not entered yet, and the amplitude will be set to 0 after marking None
                            self.trajectory_manager.add_trajectory_point(object_label=object_label,
                                                                     point_label=point_label,
                                                                     time=self.current_time / 1000,
                                                                     position=None)

            for microphone_label,microphone_pos in self.microphone_positions.items():
                self.trajectory_manager.add_trajectory_point(object_label=microphone_label, point_label='point0',
                                                             time=self.current_time / 1000, position=microphone_pos)
                self.microphone_actors[microphone_label].SetPosition(microphone_pos)

            # call ResetCamera
            # self.renderer.ResetCamera()
            #Update and write the frame
            self.w2i.Modified()
            self.w2i.Update()
            self.movieWriter.Write()

            #Render and increment time
            obj.GetRenderWindow().Render()
            self.current_time += self.time_interval
        else:
            self.iren.DestroyTimer(self.timerId)

    def get_experiment_interpolation(self):
        return self.experiment_interpolation






































































































