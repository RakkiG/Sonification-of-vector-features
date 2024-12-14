# ---visulize critical points and click
import numpy as np
import vtk
from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballActor

import Source_Microphone
import Tool_Functions as tf
import pygame  # For playing sound

# Initialize pygame mixer for sound playback
pygame.mixer.init()

# Define a dictionary to store the mapping between critical points and sound files
critical_point_sounds = {}

def get_color_for_eigenvalues(eigenvalues):
    # Sort eigenvalues by real part for easier classification
    sorted_eigenvalues = sorted(eigenvalues,key=lambda x: x.real)

    # check if all eigenvalues are real(no imaginary part)
    if np.all(np.isreal(sorted_eigenvalues)):
        # convert to real parts for consistent comparison
        sorted_real_eigenvalues = [e.real for e in sorted_eigenvalues]

        if np.all(np.array(sorted_real_eigenvalues) > 0):
            return (234 / 255, 67 / 255, 53 / 255)  # Red for source (all positive)
        elif np.all(np.array(sorted_real_eigenvalues) < 0):
            return (66 / 255, 133 / 255, 244 / 255)  # Blue for sink (all negative)
        elif sorted_real_eigenvalues[0] < 0 and sorted_real_eigenvalues[1] > 0 and sorted_real_eigenvalues[2] > 0:
            return (52 / 255, 168 / 255, 83 / 255)  # Green for 1:2 saddle (two positive, one negative)
        elif sorted_real_eigenvalues[0] < 0 and sorted_real_eigenvalues[1] < 0 and sorted_real_eigenvalues[2] > 0:
            return (52 / 255, 168 / 255, 83 / 255)  # Cyan for 2:1 saddle (one positive, two negative)

    # if eigenvalues are complex, handle the real parts
    real_parts = [e.real for e in eigenvalues]
    # positive_real = any([r > 0 for r in real_parts])
    # negative_real = any([r < 0 for r in real_parts])
    positive_real = []
    negative_real = []
    for r in real_parts:
        if r > 0:
            positive_real.append(r)
        elif r < 0:
            negative_real.append(r)

    if len(positive_real) == 3:
        return (234 / 255, 67 / 255, 53 / 255)  # Orange for spiral source
    elif len(negative_real) == 3:
        return (66 / 255, 133 / 255, 244 / 255)  # Purple for spiral sink
    elif len(positive_real) == 2 and len(negative_real) == 1:
        return (52 / 255, 168 / 255, 83 / 255)  # Light Blue for 1:2 spiral saddle
    elif len(positive_real) == 1 and len(negative_real) == 2:
        return (52 / 255, 168 / 255, 83 / 255)  # Pink for 2:1 spiral saddle
    return (1.0, 1.0, 1.0)  # White if none of the conditions matche


def create_text_actor(text, position, color):
    text_actor = vtk.vtkTextActor()
    text_actor.SetTextScaleModeToNone()
    text_actor.SetDisplayPosition(position[0], position[1])
    text_actor.SetInput(text)

    text_property = text_actor.GetTextProperty()
    text_property.SetFontFamilyToArial()
    text_property.SetFontSize(24)
    text_property.SetColor(color)

    return text_actor

class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, renderer, render_window, actors):
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.AddObserver("MouseWheelForwardEvent", self.mouse_wheel_forward_event)
        self.AddObserver("MouseWheelBackwardEvent", self.mouse_wheel_backward_event)
        self.renderer = renderer
        self.render_window = render_window
        self.actors = actors

    def left_button_press_event(self, obj, event):
        click_pos = self.GetInteractor().GetEventPosition()
        picker = vtk.vtkPropPicker()
        picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)

        picked_actor = picker.GetActor()
        if picked_actor in self.actors:
            sound_file = critical_point_sounds.get(picked_actor)
            if sound_file:
                pygame.mixer.music.load(sound_file)
                pygame.mixer.music.play()

        # Forward events
        self.OnLeftButtonDown()

    def mouse_wheel_forward_event(self, obj, event):
        self.renderer.GetActiveCamera().Dolly(1.1)  # 放大
        self.renderer.ResetCameraClippingRange()
        self.OnMouseWheelForward()

    def mouse_wheel_backward_event(self, obj, event):
        self.renderer.GetActiveCamera().Dolly(0.9)  #
        self.renderer.ResetCameraClippingRange()
        self.OnMouseWheelBackward()


def visualize_vector_field(filename,indices_to_remove=[]):
    #read .vti file
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()

    image_data = reader.GetOutput()
    print('image_data.GetDimensions()', image_data.GetDimensions())

    # output the range of .vti dataset
    # print_dataset_bounds(image_data, ".vti")

    # visualize critical points
    critical_points, jacobians, eigenvalues_list, eigenvectors_list = tf.extract_critical_points_and_jacobians(image_data)

    # # define indices to be removed
    # # indices_to_remove = {3, 4}
    #
    # #filter elements whose indices are 3 and 4
    # critical_points = [point for i, point in enumerate(critical_points) if i not in indices_to_remove]
    # jacobians = [jacobian for i, jacobian in enumerate(jacobians) if i not in indices_to_remove]
    # eigenvalues_list = [eigenvalues for i, eigenvalues in enumerate(eigenvalues_list) if i not in indices_to_remove]
    # eigenvectors_list = [eigenvectors for i, eigenvectors in enumerate(eigenvectors_list) if i not in indices_to_remove]

    indices_to_remove = {0, 1, 2, 3, 8, 11, 13, 14, 16, 20, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                         39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51}

    critical_points = [point for i, point in enumerate(critical_points) if i not in indices_to_remove]
    jacobians = [jacobian for i, jacobian in enumerate(jacobians) if i not in indices_to_remove]
    eigenvalues_list = [eigenvalues for i, eigenvalues in enumerate(eigenvalues_list) if i not in indices_to_remove]
    eigenvectors_list = [eigenvectors for i, eigenvectors in enumerate(eigenvectors_list) if i not in indices_to_remove]

    # create renderer
    renderer = vtk.vtkRenderer()

    # Create a dictionary of actors for the critical points
    critical_point_actors = []

    for idx, point in enumerate(critical_points):
        eigenvalues = eigenvalues_list[idx]
        color = get_color_for_eigenvalues(eigenvalues)

        # create a sphere to represent a critical point
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetCenter(point[0], point[1], point[2])
        sphere_source.SetRadius(0.5)  # radius of the sphere

        # create a mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere_source.GetOutputPort())

        # create an actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)  # set colors for spheres

        # add actor to the renderer
        renderer.AddActor(actor)
        eigenvalue_str = "_".join([str(int(np.round(val))) for val in eigenvalues])

        # sound_file = f"/Users/gongyaqi/Desktop/MasterThesis/Sonification_of_vector_field_features/Method_1_3_sins/source_{idx}_({eigenvalue_str})_3sins.wav"
        # sound_file = f"/Users/gongyaqi/Desktop/MasterThesis/Sonification_of_vector_field_features/Method2_deformed_bubble_repeated_signals/bubble{idx}_({eigenvalue_str}).wav"
        sound_file = f"/Users/gongyaqi/Desktop/MasterThesis/Sonification_of_vector_field_features/Method3_spherical_bubble_signals/source{idx}_({eigenvalue_str})_bubble_sound.wav"
        # sound_file = f"/Users/gongyaqi/Desktop/MasterThesis/Sonification_of_vector_field_features/original_model/bubble_signal_{idx}.wav"


        critical_point_sounds[actor] = sound_file

        # Store the actor for interaction
        critical_point_actors.append(actor)

    tetra_filename = 'vortex_core_line.vtu'
    tvtu_file = tf.hexahedron_to_tetrahedra(filename, tetra_filename)

    # use streamline to visualize
    lines = tf.visualize_vti_file_with_streamlines(filename)

    renderer.AddActor(lines)

    # visualize vortex core line
    vortex_lines, vortex_line_points, vortex_line_cells = tf.read_vtu_file_and_process_cells2(tetra_filename)

    # 1. extract and store vortex core lines
    vortex_lines, eigenvalues_for_each_core_line = tf.read_vtu_file_and_process_cells(tetra_filename)
    sources = {}

    # Remove duplicates from vortex lines and corresponding eigenvalues
    unique_vortex_lines, unique_eigenvalues = tf.remove_duplicate_vortex_lines(vortex_lines,
                                                                               eigenvalues_for_each_core_line)


    def euclidean_distance(point1, point2):
        return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

    # Group the vortex lines based on connectivity
    connected_vortex_groups, connected_vortex_groups_eigenvalues = tf.group_connected_vortex_lines(
        unique_vortex_lines, unique_eigenvalues)

    # calculate distances between critical points
    distances = []

    for i in range(len(critical_points)):
        min_distance = float('inf')
        for j in range(len(critical_points)):
            if i != j:
                # Calculate the distance between the current point (i) and another point (j)
                distance = np.linalg.norm(np.array(critical_points[i]) - np.array(critical_points[j]))
                # Update the minimum distance if the current distance is smaller
                if distance < min_distance:
                    min_distance = distance
        # Store the minimum distance for the current point
        distances.append(min_distance)

    # find the maximum and minimum distances
    max_distance = max(distances) if distances else None
    min_distance = min(distances) if distances else None
    average_distance = np.mean(distances) if distances else None

    # create critical points' sources
    for index, critical_point in enumerate(critical_points):
        sources[f"source{index}"] = Source_Microphone.Source(signal=None, positions=[critical_point], sample_rate=None,
                                                             timeline=None, label=f"source{index}",
                                                             initial_pos=critical_point,
                                                             radius=average_distance / 2,
                                                             jacobian_eigenvalues=eigenvalues_list[index])

    # Traverse connected_vortex_groups. For each group, use every 6 lines as a sound source.
    # The sound source point is at the end of the third point.
    # First select the third point from bottom to top, and then use every 6 points as a sound source.
    # If there are less than 6 points, select the midpoint in the z direction as the sound source point.
    num_critical_sources = len(sources)
    length_per_source = 100
    i = num_critical_sources
    for index, group in enumerate(connected_vortex_groups):
        num_lines = len(group)
        if num_lines < 3:
            continue
        elif 3 <= num_lines < length_per_source:

            line_th = num_lines // 2
            # source_pos
            source_pos = group[line_th][0]
            # Subtract the positions of the first and last points of this group and take half of their length as the radius
            group_1st_point = group[0][0]
            group_last_point = group[len(group) - 1][1]

            radius = euclidean_distance(group_1st_point, group_last_point)
            jacobian_eigenvalues = connected_vortex_groups_eigenvalues[index][line_th][0]


            sources[f"source{i}"] = Source_Microphone.Source(signal=None, positions=[source_pos], sample_rate=None,
                                                             timeline=None, label=f"source{i}", initial_pos=source_pos,
                                                             radius=radius, jacobian_eigenvalues=jacobian_eigenvalues)

            i += 1
        else:
            j = 0
            remain_num_lines = len(group) - j * length_per_source
            while remain_num_lines >= length_per_source:
                source_pos = group[j * length_per_source + length_per_source // 2][0]

                start_source_point = group[j * length_per_source][0]
                end_source_point = group[j * length_per_source + length_per_source - 1][1]

                radius = euclidean_distance(start_source_point, end_source_point)
                jacobian_eigenvalues = \
                connected_vortex_groups_eigenvalues[index][j * length_per_source + length_per_source // 2][0]  # 取中间

                sources[f"source{i}"] = Source_Microphone.Source(signal=None, positions=[source_pos], sample_rate=None,
                                                                 timeline=None, label=f"source{i}",
                                                                 initial_pos=source_pos, radius=radius,
                                                                 jacobian_eigenvalues=jacobian_eigenvalues)
                j += 1
                remain_num_lines = len(group) - j * length_per_source
                i += 1

            if remain_num_lines:
                line_th = remain_num_lines // 2
                source_pos = group[j * length_per_source + line_th][0]

                start_source_point = group[j * length_per_source][0]
                end_source_point = group[j * length_per_source + remain_num_lines - 1][1]

                radius = euclidean_distance(start_source_point, end_source_point)
                jacobian_eigenvalues = \
                    connected_vortex_groups_eigenvalues[index][j * length_per_source + remain_num_lines // 2][0]

                sources[f"source{i}"] = Source_Microphone.Source(signal=None, positions=[source_pos], sample_rate=None,
                                                                 timeline=None, label=f"source{i}",
                                                                 initial_pos=source_pos, radius=radius,
                                                                 jacobian_eigenvalues=jacobian_eigenvalues)
                i += 1

    # Create a polydata for the vortex lines
    vortex_lines_polydata = vtk.vtkPolyData()
    vortex_lines_polydata.SetPoints(vortex_line_points)
    vortex_lines_polydata.SetLines(vortex_line_cells)

    # Create a mapper and actor for the vortex lines
    vortex_lines_mapper = vtk.vtkPolyDataMapper()
    vortex_lines_mapper.SetInputData(vortex_lines_polydata)
    vortex_lines_actor = vtk.vtkActor()
    vortex_lines_actor.SetMapper(vortex_lines_mapper)
    vortex_lines_actor.GetProperty().SetColor(1, 1, 0)  # Yellow color for vortex lines
    vortex_lines_actor.GetProperty().SetLineWidth(5)

    # Add the actors to the renderer
    renderer.AddActor(vortex_lines_actor)
    renderer.SetBackground(1, 1, 1)

    vtu_reader = vtk.vtkXMLUnstructuredGridReader()
    vtu_reader.SetFileName(tetra_filename)
    vtu_reader.Update()
    vtu_data = vtu_reader.GetOutput()
    # print_dataset_bounds(vtu_data, ".vtu")

    #  legend_positions，only keep positions of three labels
    legend_positions = [
        (50, 1200),  # Position for "Source"
        (50, 1160),  # Position for "Sink"
        (50, 1120),  # Position for "Saddle"
    ]

    # create legend_actors，only add "Source", "Sink", 和 "Saddle"
    legend_actors = []
    selected_labels = ["Source", "Sink", "Saddle"]
    selected_colors = {
        "Source": (234 / 255, 67 / 255, 53 / 255),  # Red
        "Sink": (66 / 255, 133 / 255, 244 / 255),  # Blue
        "Saddle": (52 / 255, 168 / 255, 83 / 255),  # Green
    }

    for i, label in enumerate(selected_labels):
        color = selected_colors[label]  # get the corresponding color
        legend_actor = create_text_actor(label, legend_positions[i], color)
        legend_actors.append(legend_actor)
        renderer.AddActor(legend_actor)

    camera = vtk.vtkCamera()
    renderer.SetActiveCamera(camera)
    renderer.ResetCamera()

    renderer.TwoSidedLightingOn()  # 开启两面光照

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1600, 1400)


    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Create the interactor style with custom actor picking
    interactor_style = CustomInteractorStyle(renderer, render_window, critical_point_actors)

    # Create render window interactor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Set the custom interactor style
    render_window_interactor.SetInteractorStyle(interactor_style)

    # Start rendering and interaction
    render_window.Render()
    render_window_interactor.Start()

# visualize_vector_field('/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/vector_field_process/kitchen_converted.vti')
# visualize_vector_field('/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/data_generation/vector_field_topology.vti')

# visualize_vector_field('/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/vector_field_process5/vector_field_complex.vti')
filename_vec = "/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/vector_field_process5/vector_field_complex_critical_points.vti"
visualize_vector_field(filename_vec,)












































