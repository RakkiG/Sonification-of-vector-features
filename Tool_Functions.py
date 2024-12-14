import json
import math
import os
import wave
from scipy.io import wavfile
from scipy.optimize import fsolve
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor, vtkRenderWindowInteractor
import matplotlib
import vtk
import Amplitude_distance2
import Doppler_effect_re_formular_ori as Doppler_effect_re_formular
import Echo
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, concatenate_audioclips, ImageClip
from scipy.interpolate import RegularGridInterpolator, CubicSpline
matplotlib.use('MacOSX')

from scipy.signal import resample
import soundfile as sf
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, concatenate_audioclips


class TrajectoryManager:
    def __init__(self):
        self.trajectories = {}

    def add_trajectory_point(self, object_label,point_label, time, position, triangle_id=None,center=None,triangle_point_id=None,velocity=0):
        if object_label not in self.trajectories:
            self.trajectories[object_label] = {}

        if point_label not in self.trajectories[object_label]:
            if point_label != 'triangle': #'point0' or 'centroid'
                self.trajectories[object_label][point_label] = []
            else:
                self.trajectories[object_label][point_label]={}
        if point_label == 'triangle':
            if triangle_id not in self.trajectories[object_label]['triangle']:
                self.trajectories[object_label]['triangle'][triangle_id] = {}

            if triangle_point_id is not None:
                if 'points' not in self.trajectories[object_label]['triangle'][triangle_id]:
                    self.trajectories[object_label]['triangle'][triangle_id]['points'] = {}
                if triangle_point_id not in self.trajectories[object_label]['triangle'][triangle_id]['points']:
                    self.trajectories[object_label]['triangle'][triangle_id]['points'][triangle_point_id] = []
                self.trajectories[object_label]['triangle'][triangle_id]['points'][triangle_point_id].append((time, position, velocity))

            if center is not None:
                if 'center' not in self.trajectories[object_label]['triangle'][triangle_id]:
                    self.trajectories[object_label]['triangle'][triangle_id]['center'] = []
                self.trajectories[object_label]['triangle'][triangle_id]['center'].append((time, position, velocity))

        else:
            self.trajectories[object_label][point_label].append((time, position, velocity))

    def get_trajectory(self, object_label):
        return self.trajectories.get(object_label, {})


def video_generator(video_path, audio_path, output_path, is_echo=False):
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)

    if audio_clip.duration > video_clip.duration:
        if is_echo:
            # Create a clip that freezes the last frame
            last_frame = video_clip.get_frame(video_clip.duration - 1 / video_clip.fps)
            freeze_clip = ImageClip(last_frame).set_duration(audio_clip.duration - video_clip.duration)

            # Concatenate the original video with the freeze clip
            video_clip = concatenate_videoclips([video_clip, freeze_clip])
        else:
            audio_clip = audio_clip.subclip(0, video_clip.duration)
    elif audio_clip.duration < video_clip.duration:
        number_of_loops = video_clip.duration // audio_clip.duration + 1
        audio_clips = [audio_clip] * int(number_of_loops)
        concatenated_clip = concatenate_audioclips(audio_clips)
        audio_clip = concatenated_clip.set_duration(video_clip.duration)

    video_clip_with_audio = video_clip.set_audio(audio_clip)
    video_clip_with_audio.write_videofile(output_path, codec='libx264', audio_codec='aac')


#Create a VTK actor and add a label
def create_actor(shape_source,label):
    #make sure that shape_source is not null
    if shape_source.GetOutputPort() is None:
        raise ValueError("shape_source is None")

    #Create a mapper and map to the source of shapes
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(shape_source.GetOutputPort())

    #Create an actor and set the mapper
    actor = vtkActor()
    actor.SetMapper(mapper)

    #Add self-defined label to the actor
    actor.label = label

    #return the actor object
    return actor

def divide_and_round_up(f, a):
    if f % a == 0:
        r = f // a
    else:
        r = f // a + 1
    return r

def linear_interpolation(time_start,pos_start, time_end,pos_end, current_time):
    #Calculate the ratio of the current time within the interval
    ratio = (current_time-time_start)/(time_end-time_start)

    #Calculate the interpolated position
    interpolated_pos = tuple(pos_start[i] + ratio*(pos_end[i]-pos_start[i]) for i in range(3))

    return interpolated_pos

def generate_positions_for_all_points(initial_points_positions,zeroth_point_positions,time_array):
    #create a dictionary to store positions of 0th and centroid
    all_positions = {}
    all_positions['point0'] = []
    all_positions['centroid'] = []

    print("zeroth_point_positions:",zeroth_point_positions)

    for i in range(len(zeroth_point_positions)):
        all_positions['point0'].append((time_array[i],zeroth_point_positions[i]))

    for k in range(0,len(zeroth_point_positions)):

        difference = tuple(a - b for a, b in zip(zeroth_point_positions[k], initial_points_positions['point0'])) #三维坐标之间的差值
        position = tuple(a + b for a, b in zip(initial_points_positions['centroid'], difference))
        all_positions['centroid'].append((time_array[k],position))

    all_positions['triangle'] = {}

    if 'triangle' in initial_points_positions:
        for triangle_id, triangle_dict in initial_points_positions['triangle'].items():
            all_positions['triangle'][triangle_id] = {}
            for point_type, point_positions in triangle_dict.items():
                if point_type == 'center':
                    all_positions['triangle'][triangle_id]['center'] = []
                    for k in range(0,len(zeroth_point_positions)):
                        difference = tuple(a - b for a, b in zip(zeroth_point_positions[k], initial_points_positions['point0']))
                        position = tuple(a + b for a, b in zip(point_positions, difference))
                        all_positions['triangle'][triangle_id]['center'].append((time_array[k],position))

                else:
                    all_positions['triangle'][triangle_id][point_type] = {}
                    for i in range(3):
                        all_positions['triangle'][triangle_id][point_type][f'point{i}'] = []
                        for k in range(0,len(zeroth_point_positions)):
                            difference = tuple(a - b for a, b in zip(zeroth_point_positions[k], initial_points_positions['point0']))
                            position = tuple(a + b for a, b in zip(initial_points_positions['triangle'][triangle_id]['points'][i], difference))
                            all_positions['triangle'][triangle_id][point_type][f'point{i}'].append((time_array[k],position))

    print("all_positions:",all_positions)
    return all_positions


def generate_positions_for_microphone(initial_microphones_positions,time_array):
#create a dictionary to store the positions of the microphone
    microphone_positions = {}
    for microphone_label, microphone_position in initial_microphones_positions.items():
        microphone_positions[microphone_label] = {}
        microphone_positions[microphone_label]['point0'] = []
        for i in range(len(time_array)):
            microphone_positions[microphone_label]['point0'].append((time_array[i], initial_microphones_positions[microphone_label]))
    return microphone_positions

def unified_sample_rate(source_objects):
    def resample_audio( input_signal=None, input_sr=None, target_sr=48000):
        signal, sr = input_signal, input_sr
        target_length = int(len(signal) * target_sr / sr)

        resampled_signal = resample(signal, target_length)

        print(f"Resampled to {target_sr} Hz, new signal length: {len(resampled_signal)}")

        return resampled_signal,target_sr

    for object_label, source_object in source_objects.items():
        if source_object.signal is None:
            continue
        signal = source_object.signal
        source_object.signal,source_object.sample_rate = resample_audio(input_signal=signal, input_sr=source_object.sample_rate)


def Unify_audio_len(source_objects, animation_duration_ms):
    def adjust_signal_duration(signal, sample_rate, duration_ms):
        num_samples_required = int(sample_rate * (duration_ms / 1000.0))
        if len(signal) < num_samples_required:
            repeat_count = math.ceil(num_samples_required / len(signal))
            processed_signal = np.tile(signal, repeat_count)  #
            print(f"Repeated signal length: {len(processed_signal)}")
            processed_signal = processed_signal[:num_samples_required]  #
        else:
            processed_signal = signal[:num_samples_required]  #
            print(f"Signal was long enough, no need to repeat.")

        print(f"Original signal length: {len(signal)}")
        print(f"Final signal length: {len(processed_signal)}")

        timeline = np.linspace(0, duration_ms / 1000.0, num_samples_required, endpoint=False)
        return processed_signal, timeline

    for object_label, source_object in source_objects.items():
        if source_object.signal is None:
            continue
        signal = source_object.signal
        adjusted_signal, timeline = adjust_signal_duration(signal, source_object.sample_rate, animation_duration_ms)
        # normalized_signal = np.int16((adjusted_signal / np.max(np.abs(adjusted_signal))) * 32767)

        source_object.signal = adjusted_signal
        source_object.timeline = timeline

        # print(f"The length of the adjusted signal for {object_label} is {len(normalized_signal)}")


def cal_centroid_unstructured(input_file):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(input_file)
    reader.Update()

    unstructuredGrid = reader.GetOutput()

    centerOfMassFilter = vtk.vtkCenterOfMass()
    centerOfMassFilter.SetInputData(unstructuredGrid)
    centerOfMassFilter.Update()

    center = centerOfMassFilter.GetCenter()
    rounded_center = (round(center[0], 2), round(center[1], 2), round(center[2], 2))

    return rounded_center


def filter_and_save_vtu(input_grid,output_filename):

    used_points = set()
    for i in range(input_grid.GetNumberOfCells()):
        cell = input_grid.GetCell(i)
        point_ids = cell.GetPointIds()
        for j in range(point_ids.GetNumberOfIds()):
            used_points.add(point_ids.GetId(j))


    output_grid = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints()


    point_map = {}


    for old_id in sorted(used_points):
        point_coordinates = input_grid.GetPoint(old_id)
        new_id = points.InsertNextPoint(point_coordinates)
        point_map[old_id] = new_id

    output_grid.SetPoints(points)


    for i in range(input_grid.GetNumberOfCells()):
        cell = input_grid.GetCell(i)
        new_cell_points = vtk.vtkIdList()
        for j in range(cell.GetPointIds().GetNumberOfIds()):
            new_cell_points.InsertNextId(point_map[cell.GetPointId(j)])

        output_grid.InsertNextCell(cell.GetCellType(),new_cell_points)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(output_grid)
    writer.Write()

    return [points.GetPoint(i) for i in range(points.GetNumberOfPoints())]


def get_all_connected_components(filename,output_dir,color_config_json):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()

    # vtkUnstructuredGrid
    input_grid = reader.GetOutput()

    connectivityFilter = vtk.vtkConnectivityFilter()
    connectivityFilter.SetInputData(input_grid)
    connectivityFilter.SetExtractionModeToAllRegions()
    connectivityFilter.ColorRegionsOn()
    connectivityFilter.Update()

    numRegions = connectivityFilter.GetNumberOfExtractedRegions()

    with open(color_config_json, 'r') as file:
        original_config = json.load(file)
        original_point_colors = original_config['point_colors']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for region_index in range(numRegions):
        connectivityFilter.InitializeSpecifiedRegionList()
        connectivityFilter.AddSpecifiedRegion(region_index)
        connectivityFilter.SetExtractionModeToSpecifiedRegions()
        connectivityFilter.Update()


        filename_vtu = os.path.join(output_dir,f'connected_component_{region_index}.vtu')


        filtered_points = filter_and_save_vtu(connectivityFilter.GetOutput(),filename_vtu)

        filtered_points_indices = []
        filtered_colors = []
        for point in filtered_points:
            idx = input_grid.FindPoint(point)
            if idx != -1:
                filtered_points_indices.append(idx)
                filtered_colors.append(original_point_colors[idx])


        filename_json = os.path.join(output_dir,f'connected_component_{region_index}_config.json')
        with open(filename_json,'w') as f_json:
            json_config = {
                "point_colors":filtered_colors,
                "point_indices_in_original_grid":filtered_points_indices
            }
            json.dump(json_config, f_json, indent=4)
    print(f"Extracted {numRegions} components and saved to separate files along with their color configurations.")
    return numRegions

def get_point_coordinate_from_vtu(file_path,point_index):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()

    unstructured_grid = reader.GetOutput()

    if point_index < unstructured_grid.GetNumberOfPoints():
        point_coordinates = unstructured_grid.GetPoint(point_index)
        print(f"Coordinates of point at index {point_index}: {point_coordinates}")
        return point_coordinates
    else:
        print("Point index is out of the range of available points.")
        return None

def combine_to_stereo(left_channel, right_channel, sample_rate, output_filename):

    if left_channel.ndim > 1:
        left_channel = left_channel.flatten()
    if right_channel.ndim > 1:
        right_channel = right_channel.flatten()

    min_length = min(len(left_channel), len(right_channel))
    left_channel = left_channel[:min_length].astype(np.int16)
    right_channel = right_channel[:min_length].astype(np.int16)

    stereo_signal = np.empty((2 * min_length,), dtype=np.int16)
    stereo_signal[0::2] = left_channel
    stereo_signal[1::2] = right_channel


    with wave.open(output_filename, 'w') as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(stereo_signal.tobytes())


def result_generation(microphone_objects,source_objects,is_amplitude_distance=False, is_doppler=False,is_echo=False):
    signals = {}
    sum_signals = {}
    average_signals = {}
    echo_signals = {}
    for microphone_label, microphone_object in microphone_objects.items():
        signals[microphone_label] = {}
        num_signals = 0
        sum_signals[microphone_label]=None

        for source_label, source_object in source_objects.items():
            if source_object.signal is None:
                continue

            #create a dictionary to store the results
            centroid_trajectory = source_object.trajectory['centroid']
            if centroid_trajectory is not None:
                if is_amplitude_distance:
                    signals[microphone_label][source_label] = Amplitude_distance2.amplitude_distance(
                        sample_rate=source_object.sample_rate, source_trajectory=source_object.trajectory['centroid'],
                        receiver_trajectory=microphone_object.trajectory['point0'], signal=source_object.signal)

                    if is_doppler:
                        signals[microphone_label][source_label] = Doppler_effect_re_formular.Doppler_effect_re_formular(
                                                        sample_rate=source_object.sample_rate,
                                                         timeline=source_object.timeline,
                                                         source_trajectory=source_object.trajectory['centroid'],
                                                         receiver_trajectory=microphone_object.trajectory['point0'],
                                                         signal=signals[microphone_label][source_label])
                elif is_doppler:
                    signals[microphone_label][source_label] = Doppler_effect_re_formular.Doppler_effect_re_formular(
                                                        sample_rate=source_object.sample_rate,
                                                         timeline=source_object.timeline,
                                                         source_trajectory=source_object.trajectory['centroid'],
                                                         receiver_trajectory=microphone_object.trajectory['point0'],
                                                         signal=source_object.signal)
                else:
                    signals[microphone_label][source_label] = source_object.signal

            for source_label,source_signal in signals[microphone_label].items():
                if sum_signals[microphone_label] is None:
                    sum_signals[microphone_label] = np.zeros_like(source_signal, dtype=np.float64)
                sum_signals[microphone_label] += source_signal
                num_signals += 1

            # average_signals[microphone_label] = sum_signals[microphone_label] / num_signals
            average_signals[microphone_label] = sum_signals[microphone_label]/num_signals


        if is_echo:

            echo_signals[microphone_label] = Echo.Echo(source_objects, microphone_object)
            average_signals = echo_signals


    return average_signals

def get_surface_triangles(vtu_file_path, max_triangle_area):
    def calculate_triangle_area(points):
        p0, p1, p2 = points
        vec1 = np.array(p1) - np.array(p0)
        vec2 = np.array(p2) - np.array(p0)
        cross_product = np.cross(vec1, vec2)
        area = np.linalg.norm(cross_product) / 2.0
        return area

    def subdivide_triangle(points):
        p0, p1, p2 = points
        mid01 = (p0 + p1) / 2.0
        mid12 = (p1 + p2) / 2.0
        mid20 = (p2 + p0) / 2.0
        return [
            (p0, mid01, mid20),
            (mid01, p1, mid12),
            (mid01, mid12, mid20),
            (mid20, mid12, p2)
        ]

    def subdivide_large_triangles(polydata, max_area):
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        centers = vtk.vtkPoints()

        for i in range(polydata.GetNumberOfCells()):
            cell = polydata.GetCell(i)
            if cell.GetCellType() == vtk.VTK_TRIANGLE:
                pts = [np.array(polydata.GetPoint(cell.GetPointId(j))) for j in range(3)]
                queue = [pts]
                while queue:
                    tri_pts = queue.pop()
                    if calculate_triangle_area(tri_pts) > max_area:
                        queue.extend(subdivide_triangle(tri_pts))
                    else:
                        ids = [points.InsertNextPoint(pt) for pt in tri_pts]
                        cells.InsertNextCell(3, ids)
                        center = np.mean(tri_pts, axis=0)
                        centers.InsertNextPoint(center)

        output_polydata = vtk.vtkPolyData()
        output_polydata.SetPoints(points)
        output_polydata.SetPolys(cells)

        center_points_polydata = vtk.vtkPolyData()
        center_points_polydata.SetPoints(centers)

        return output_polydata, center_points_polydata


    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_file_path)
    reader.Update()
    unstructured_grid = reader.GetOutput()

    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(unstructured_grid)
    geometry_filter.Update()

    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputConnection(geometry_filter.GetOutputPort())
    triangle_filter.Update()

    triangle_polydata = vtk.vtkPolyData()
    triangle_polydata.SetPoints(triangle_filter.GetOutput().GetPoints())
    cells = vtk.vtkCellArray()
    has_triangles = False
    for i in range(triangle_filter.GetOutput().GetNumberOfCells()):
        cell = triangle_filter.GetOutput().GetCell(i)
        if cell.GetCellType() == vtk.VTK_TRIANGLE:
            has_triangles = True
            ids = cell.GetPointIds()
            cells.InsertNextCell(3)
            cells.InsertCellPoint(ids.GetId(0))
            cells.InsertCellPoint(ids.GetId(1))
            cells.InsertCellPoint(ids.GetId(2))

    if not has_triangles:
        return None, None

    triangle_polydata.SetPolys(cells)

    subdivided_surface, center_points_polydata = subdivide_large_triangles(triangle_polydata, max_triangle_area)

    return subdivided_surface, center_points_polydata


def get_center_points_array(vtu_file_path, max_triangle_area):
    subdivided_surface, center_points_polydata = get_surface_triangles(vtu_file_path, max_triangle_area)
    if subdivided_surface is None:
        return None
    center_points = []
    for i in range(center_points_polydata.GetNumberOfPoints()):
        center_points.append(center_points_polydata.GetPoint(i))

    return center_points

def get_triangle_points_array(vtu_file_path, max_triangle_area):
    subdivided_surface, _ = get_surface_triangles(vtu_file_path, max_triangle_area)
    if subdivided_surface is None:
        return None
    points = subdivided_surface.GetPoints()
    triangles = []

    for i in range(subdivided_surface.GetNumberOfCells()):
        cell = subdivided_surface.GetCell(i)
        if cell.GetCellType() == vtk.VTK_TRIANGLE:
            triangle = []
            for j in range(3):
                pt = points.GetPoint(cell.GetPointId(j))
                triangle.append((pt[0], pt[1], pt[2]))
            triangles.append(triangle)

    return triangles

def get_triangles_with_centers(vtu_file_path, max_triangle_area):
    subdivided_surface, center_points_polydata = get_surface_triangles(vtu_file_path, max_triangle_area)
    if subdivided_surface is None:
        return None
    points = subdivided_surface.GetPoints()
    triangles = {}

    for i in range(subdivided_surface.GetNumberOfCells()):
        cell = subdivided_surface.GetCell(i)
        if cell.GetCellType() == vtk.VTK_TRIANGLE:
            triangle_points = []
            for j in range(3):
                pt = points.GetPoint(cell.GetPointId(j))
                triangle_points.append((pt[0], pt[1], pt[2]))
            center = np.mean(triangle_points, axis=0)
            triangles[f'triangle{i}'] = {
                'points': triangle_points,
                'center': (center[0], center[1], center[2])
            }
    return triangles



def line_segment_intersects_triangle(p1, p2, triangle):
    def point_in_triangle(pt, v0, v1, v2):
        # Barycentric coordinate system to check if point pt is inside the triangle v0, v1, v2
        d00 = np.dot(v2 - v0, v2 - v0)
        d01 = np.dot(v2 - v0, v1 - v0)
        d11 = np.dot(v1 - v0, v1 - v0)
        d20 = np.dot(pt - v0, v2 - v0)
        d21 = np.dot(pt - v0, v1 - v0)

        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        return (u >= 0) and (v >= 0) and (w >= 0)

    def ray_intersects_triangle(ray_origin, ray_vector, v0, v1, v2):
        EPSILON = 1e-9
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(ray_vector, edge2)
        a = np.dot(edge1, h)
        if -EPSILON < a < EPSILON:
            return False, None  # This ray is parallel to this triangle.
        f = 1.0 / a
        s = ray_origin - v0
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return False, None
        q = np.cross(s, edge1)
        v = f * np.dot(ray_vector, q)
        if v < 0.0 or u + v > 1.0:
            return False, None
        # At this stage we can compute t to find out where the intersection point is on the line.
        t = f * np.dot(edge2, q)
        if t > EPSILON:  # ray intersection
            intersect_point = ray_origin + ray_vector * t
            return True, intersect_point
        else:  # This means that there is a line intersection but not a ray intersection.
            return False, None

    v0 = np.array(triangle[0])
    v1 = np.array(triangle[1])
    v2 = np.array(triangle[2])

    ray_origin = np.array(p1)
    ray_vector = np.array(p2) - np.array(p1)

    intersects, intersect_point = ray_intersects_triangle(ray_origin, ray_vector, v0, v1, v2)
    if not intersects:
        return False
    return point_in_triangle(intersect_point, v0, v1, v2)


def set_actor_color(actor, colors):
    #Set the color of the actor based on the provided color list.
    points = actor.GetMapper().GetInput().GetPoints()
    point_data = actor.GetMapper().GetInput().GetPointData()
    colors_array = vtk.vtkUnsignedCharArray()
    colors_array.SetNumberOfComponents(3)
    colors_array.SetName("Colors")

    for i in range(points.GetNumberOfPoints()):
        colors_array.InsertNextTuple(colors[i])

    point_data.SetScalars(colors_array)


def uniform_sphere_sampling(center, radius, num_samples):
    phi = np.random.uniform(0, 2*np.pi, num_samples)
    costheta = np.random.uniform(-1, 1, num_samples)
    u = np.random.uniform(0, 1, num_samples)

    theta = np.arccos(costheta)
    r = radius * np.cbrt(u)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.column_stack((x, y, z)) + center



def load_vti_file(filename):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def trilinear_interpolation_coefficients(vectors, coords):

    # Calculate the coefficients for the trilinear interpolation function.
    # vectors: List of 8 vectors at the corners of a cell.
    # coords: List of 8 (x, y, z) coordinates of the corners.

    A = np.zeros((8, 8))
    B_u = np.zeros(8)
    B_v = np.zeros(8)
    B_w = np.zeros(8)

    for idx in range(8):
        x, y, z = coords[idx]
        A[idx] = [1, x, y, z, x*y, x*z, y*z, x*y*z]
        B_u[idx] = vectors[idx][0]
        B_v[idx] = vectors[idx][1]
        B_w[idx] = vectors[idx][2]

    coeffs_u = np.linalg.solve(A, B_u)
    coeffs_v = np.linalg.solve(A, B_v)
    coeffs_w = np.linalg.solve(A, B_w)

    return coeffs_u, coeffs_v, coeffs_w

def trilinear_interpolation_function(coeffs, x, y, z):
    return coeffs[0] + coeffs[1]*x + coeffs[2]*y + coeffs[3]*z + coeffs[4]*x*y + coeffs[5]*x*z + coeffs[6]*y*z + coeffs[7]*x*y*z

def find_zero_points(coeffs_u, coeffs_v, coeffs_w, bounds):
    def equations(vars):
        x, y, z = vars
        return (
            trilinear_interpolation_function(coeffs_u, x, y, z),
            trilinear_interpolation_function(coeffs_v, x, y, z),
            trilinear_interpolation_function(coeffs_w, x, y, z)
        )

    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    initial_guess = [(x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2]
    initial_guess = np.array(initial_guess)
    solution = fsolve(equations, initial_guess)

    if x_min <= solution[0] <= x_max and y_min <= solution[1] <= y_max and z_min <= solution[2] <= z_max:
        return solution
    else:
        return None

def compute_jacobian(coeffs_u, coeffs_v, coeffs_w, x, y, z):

    J = np.zeros((3, 3))

    J[0, 0] = coeffs_u[1] + coeffs_u[4]*y + coeffs_u[5]*z + coeffs_u[7]*y*z
    J[0, 1] = coeffs_u[2] + coeffs_u[4]*x + coeffs_u[6]*z + coeffs_u[7]*x*z
    J[0, 2] = coeffs_u[3] + coeffs_u[5]*x + coeffs_u[6]*y + coeffs_u[7]*x*y

    J[1, 0] = coeffs_v[1] + coeffs_v[4]*y + coeffs_v[5]*z + coeffs_v[7]*y*z
    J[1, 1] = coeffs_v[2] + coeffs_v[4]*x + coeffs_v[6]*z + coeffs_v[7]*x*z
    J[1, 2] = coeffs_v[3] + coeffs_v[5]*x + coeffs_v[6]*y + coeffs_v[7]*x*y

    J[2, 0] = coeffs_w[1] + coeffs_w[4]*y + coeffs_w[5]*z + coeffs_w[7]*y*z
    J[2, 1] = coeffs_w[2] + coeffs_w[4]*x + coeffs_w[6]*z + coeffs_w[7]*x*z
    J[2, 2] = coeffs_w[3] + coeffs_w[5]*x + coeffs_w[6]*y + coeffs_w[7]*x*y

    return J

def extract_critical_points_and_jacobians(image_data):
    dims = image_data.GetDimensions()
    spacing = image_data.GetSpacing()
    origin = image_data.GetOrigin()
    critical_points = []
    jacobians = []
    eigenvalues_list = []
    eigenvectors_list = []

    vector_data = image_data.GetPointData().GetVectors()

    for k in range(dims[2] - 1):
        for j in range(dims[1] - 1):
            for i in range(dims[0] - 1):
                vectors = []
                coords = []
                for dz in range(2):
                    for dy in range(2):
                        for dx in range(2):
                            x = i + dx
                            y = j + dy
                            z = k + dz

                            #Get the vector at the current grid point
                            vector_index = x + dims[0] * (y+dims[1]*z)
                            vector = vector_data.GetTuple3(vector_index)
                            vectors.append(vector)
                            coords.append((x * spacing[0] + origin[0], y * spacing[1] + origin[1], z * spacing[2] + origin[2]))

                u_signs = [np.sign(v[0]) for v in vectors]
                v_signs = [np.sign(v[1]) for v in vectors]
                w_signs = [np.sign(v[2]) for v in vectors]

                if len(set(u_signs)) > 1 and len(set(v_signs)) > 1 and len(set(w_signs)) > 1:
                    coeffs_u, coeffs_v, coeffs_w = trilinear_interpolation_coefficients(vectors, coords)
                    bounds = (
                        coords[0][0], coords[1][0],
                        coords[0][1], coords[2][1],
                        coords[0][2], coords[4][2]
                    )
                    zero_point = find_zero_points(coeffs_u, coeffs_v, coeffs_w, bounds)
                    if zero_point is not None:
                        critical_points.append(zero_point)
                        jacobian = compute_jacobian(coeffs_u, coeffs_v, coeffs_w, *zero_point)
                        jacobians.append(jacobian)

                        # Compute eigenvalues and eigenvectors
                        eigenvalues, eigenvectors = np.linalg.eig(jacobian)
                        eigenvalues_list.append(eigenvalues)
                        eigenvectors_list.append(eigenvectors)

    # Remove duplicate points and corresponding Jacobians by rounding to a precision of 5 decimal places
    unique_critical_points = []
    unique_jacobians = []
    unique_eigenvalues_list = []
    unique_eigenvectors_list = []
    tolerance = 0.1

    for point, jacobian, eigenvalues, eigenvectors in zip(critical_points, jacobians, eigenvalues_list,
                                                          eigenvectors_list):
        is_unique = True
        for up in unique_critical_points:
            distance = np.linalg.norm(np.array(point) - np.array(up))
            print(distance)
            if distance < tolerance:
                is_unique = False
                break
        if is_unique:
            unique_critical_points.append(point)
            unique_jacobians.append(jacobian)
            unique_eigenvalues_list.append(eigenvalues)
            unique_eigenvectors_list.append(eigenvectors)

    return unique_critical_points, unique_jacobians, unique_eigenvalues_list, unique_eigenvectors_list

def linear_interpolation_coefficients(vectors,coords):

    # Calculate the coefficients for the linear interpolation function.

    A = np.zeros((4, 4))
    B_u = np.zeros(4)
    B_v = np.zeros(4)
    B_w = np.zeros(4)

    for idx in range(4):
        x, y, z = coords[idx]
        A[idx] = [1, x, y, z]

        B_u[idx] = vectors[idx][0]
        B_v[idx] = vectors[idx][1]
        B_w[idx] = vectors[idx][2]

    coeffs_u = np.linalg.solve(A, B_u)
    coeffs_v = np.linalg.solve(A, B_v)
    coeffs_w = np.linalg.solve(A, B_w)

    return coeffs_u, coeffs_v, coeffs_w

def linear_interpolation_function(coeffs, x, y, z):
    return coeffs[0] + coeffs[1]*x + coeffs[2]*y + coeffs[3]*z

def compute_jacobian2(coeffs_u, coeffs_v, coeffs_w):

    # Compute the Jacobian matrix from interpolation coefficients.


    jacobian = np.array([
        [coeffs_u[1], coeffs_u[2], coeffs_u[3]],
        [coeffs_v[1], coeffs_v[2], coeffs_v[3]],
        [coeffs_w[1], coeffs_w[2], coeffs_w[3]]
    ])
    return jacobian

def check_eigenvalues(J):
    # Check if the matrix A has one real eigenvalue and a pair of complex-conjugate eigenvalues.

    eigenvalues, eigenvectors = np.linalg.eig(J)
    real_eigenvalues = [val.real for val in eigenvalues if np.isreal(val)]
    complex_eigenvalues = [val for val in eigenvalues if np.iscomplex(val)]

    if len(real_eigenvalues) == 1 and len(complex_eigenvalues) == 2:
        if np.isclose(complex_eigenvalues[0], np.conj(complex_eigenvalues[1])):
            real_eigenvalue_index = np.where(eigenvalues == real_eigenvalues[0])[0][0]
            return True, eigenvectors[:, real_eigenvalue_index].real,eigenvalues
    return False, None, None

def reduced_velocity_function(x,y,z, coeffs_u, coeffs_v, coeffs_w, normal_vector):
    u = linear_interpolation_function(coeffs_u, x,y,z)
    v = linear_interpolation_function(coeffs_v, x,y,z)
    w = linear_interpolation_function(coeffs_w, x,y,z)
    velocity = np.array([u, v, w])
    reduced_velocity = velocity - np.dot(velocity, normal_vector) * normal_vector
    return reduced_velocity

def point_in_triangle(pt, v0, v1, v2):
    # Barycentric coordinate system to check if point pt is inside the triangle v0, v1, v2
    d00 = np.dot(v2 - v0, v2 - v0)
    d01 = np.dot(v2 - v0, v1 - v0)
    d11 = np.dot(v1 - v0, v1 - v0)
    d20 = np.dot(pt - v0, v2 - v0)
    d21 = np.dot(pt - v0, v1 - v0)

    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return (u >= 0) and (v >= 0) and (w >= 0)

def find_zero_reduced_velocity_points(coeffs_u, coeffs_v, coeffs_w, face_coords, normal_vector):

    # Find the point on the face where the reduced velocity is zero.

    # Convert face_coords from tuples to numpy arrays
    face_coords = [np.array(vertex) for vertex in face_coords]

    n = normal_vector / np.linalg.norm(normal_vector)

    def func(vars):
        x,y,z = vars
        return reduced_velocity_function(x,y,z, coeffs_u, coeffs_v, coeffs_w, n)

    # Use initial guess as the centroid of the face
    initial_guess = np.mean(face_coords, axis=0)

    point, _, ier, _ = fsolve(func, initial_guess, full_output=True)

    if ier == 1 and point_in_triangle(point, face_coords[0], face_coords[1],face_coords[2]):
        # check if the point is inside the triangle using barycentric coordinates
        return point
    else:
        return None

def remove_duplicate_points(points, tolerance=1e-5):
    unique_points = []
    for point in points:
        if not any(np.linalg.norm(np.array(point) - np.array(up)) < tolerance for up in unique_points):
            unique_points.append(point)
    return unique_points
faces = [
    [0, 1, 2],
    [0, 1, 3],
    [0, 2, 3],
    [1, 2, 3]
]

def check_tetrahedron_for_zero_reduced_velocity(vectors, coords):
    # Calculate interpolation coefficients
    coeffs_u, coeffs_v, coeffs_w = linear_interpolation_coefficients(vectors, coords)

    # Compute the Jacobian matrix
    jacobian = compute_jacobian2(coeffs_u, coeffs_v, coeffs_w)

    # Check eigenvalues and find the eigenvector corresponding to the real eigenvalue
    has_valid_eigenvalues, real_eigenvector,eigenvalues_for_one_line = check_eigenvalues(jacobian)

    zero_points = []
    if has_valid_eigenvalues:
        # Check each face for zero reduced velocity point
        for face in faces:
            face_coords = [coords[i] for i in face]
            point = find_zero_reduced_velocity_points(coeffs_u, coeffs_v, coeffs_w, face_coords, real_eigenvector)

            if point is not None:
                zero_points.append(point)

        zero_points = remove_duplicate_points(zero_points)

    if len(zero_points) == 2:
        point1, point2 = zero_points
        return np.array([point1, point2]),eigenvalues_for_one_line
    else:
        return None,None

def read_vtu_file_and_process_cells(file_path):
    # Compared to the following 2, only the return is different
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()
    data = reader.GetOutput()

    points = data.GetPoints()
    # cells = data.GetCells()
    velocity_array = data.GetPointData().GetScalars("Vectors")

    vortex_lines = []
    vortex_line_points = vtk.vtkPoints()
    vortex_line_cells = vtk.vtkCellArray()
    eigenvalues_for_each_core_line = []

    for i in range(data.GetNumberOfCells()):
        cell = data.GetCell(i)
        if cell.GetCellType() != vtk.VTK_TETRA:
            continue

        cell_points = cell.GetPoints()
        coords = [points.GetPoint(cell.GetPointId(j)) for j in range(cell_points.GetNumberOfPoints())]
        vectors = [velocity_array.GetTuple(cell.GetPointId(j)) for j in range(cell_points.GetNumberOfPoints())]

        # try:
        vortex_line, eigenvalues_for_one_line = check_tetrahedron_for_zero_reduced_velocity(vectors, coords)
        if vortex_line is not None:
            vortex_lines.append(vortex_line)
            eigenvalues_for_each_core_line.append([eigenvalues_for_one_line,eigenvalues_for_one_line])
            # Add points to vtkPoints and create a line cell
            id1 = vortex_line_points.InsertNextPoint(vortex_line[0])
            id2 = vortex_line_points.InsertNextPoint(vortex_line[1])
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, id1)
            line.GetPointIds().SetId(1, id2)
            vortex_line_cells.InsertNextCell(line)
            print(f"Found vortex line at cell {i}: {vortex_line}")

    return vortex_lines, eigenvalues_for_each_core_line

def hexahedron_to_tetrahedra(input_filename, output_filename):
    # Read the .vti file
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(input_filename)
    reader.Update()

    # Get the structured grid
    structured_grid = reader.GetOutput()

    # Prepare a new unstructured grid to hold tetrahedra
    tetra_grid = vtk.vtkUnstructuredGrid()
    tetra_points = vtk.vtkPoints()

    # Create an array to store the vector data
    vector_data = vtk.vtkDoubleArray()
    vector_data.SetNumberOfComponents(3)
    vector_data.SetName("Vectors")

    # Extract dimensions and spacing from the structured grid
    dims = structured_grid.GetDimensions()
    spacing = structured_grid.GetSpacing()
    origin = structured_grid.GetOrigin()

    # Get the vector array from the structured grid
    vector_array = structured_grid.GetPointData().GetVectors()

    # Generate points manually based on dimensions, spacing, and origin
    point_ids = {}
    for k in range(dims[2]):
        for j in range(dims[1]):
            for i in range(dims[0]):
                x = origin[0] + i * spacing[0]
                y = origin[1] + j * spacing[1]
                z = origin[2] + k * spacing[2]
                point_id = tetra_points.InsertNextPoint(x, y, z)
                point_ids[(i, j, k)] = point_id

                # Get vector value from the structured grid and add it to the vector_data array
                vector = vector_array.GetTuple3(i + dims[0] * (j + dims[1] * k))
                vector_data.InsertNextTuple(vector)
                print(f"Vector at ({i}, {j}, {k}): {vector}")

    tetra_grid.SetPoints(tetra_points)
    tetra_grid.GetPointData().SetVectors(vector_data)

    # Define the six tetrahedra in terms of the hexahedron's vertices
    hexa_to_tetra = [
        [0, 1, 3, 4],
        [1, 2, 3, 6],
        [1, 4, 5, 6],
        [3, 4, 6, 7],
        [1, 3, 4, 6],
        [0, 3, 4, 6]
    ]

    # Iterate over all cells in the structured grid and convert them
    for k in range(dims[2] - 1):
        for j in range(dims[1] - 1):
            for i in range(dims[0] - 1):
                # Get the point ids for the hexahedron
                hexa_ids = [
                    point_ids[(i, j, k)],
                    point_ids[(i + 1, j, k)],
                    point_ids[(i + 1, j + 1, k)],
                    point_ids[(i, j + 1, k)],
                    point_ids[(i, j, k + 1)],
                    point_ids[(i + 1, j, k + 1)],
                    point_ids[(i + 1, j + 1, k + 1)],
                    point_ids[(i, j + 1, k + 1)]
                ]

                # Create six tetrahedra from the hexahedron
                for tet in hexa_to_tetra:
                    tetra = vtk.vtkTetra()
                    for i in range(4):
                        tetra.GetPointIds().SetId(i, hexa_ids[tet[i]])
                    tetra_grid.InsertNextCell(tetra.GetCellType(), tetra.GetPointIds())

    # Write the tetrahedral grid to a .vtu file
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(tetra_grid)
    writer.Write()

    return tetra_grid

def visualize_vti_file_with_streamlines(filename):

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()

    image_data = reader.GetOutput()
    critical_points, jacobians, eigenvalues_list, eigenvectors_list = extract_critical_points_and_jacobians(image_data)
    tetra_filename = 'vortex_core_line.vtu'
    tvtu_file = hexahedron_to_tetrahedra(filename, tetra_filename)

    vectors = image_data.GetPointData().GetVectors()

    if vectors is None:
        print("Error: No velocity vector data found in the file.")
        return

    bounds = image_data.GetBounds()
    max_length = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

    append_filter = vtk.vtkAppendPolyData()

    seed_points = critical_points

    vortex_lines, eigenvalues_for_each_core_line = read_vtu_file_and_process_cells(tetra_filename)

    # Remove duplicates from vortex lines and corresponding eigenvalues
    unique_vortex_lines, unique_eigenvalues = remove_duplicate_vortex_lines(vortex_lines,
                                                                               eigenvalues_for_each_core_line)

    # def euclidean_distance(point1, point2):
    #     return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

    # Group the vortex lines based on connectivity
    connected_vortex_groups, connected_vortex_groups_eigenvalues = group_connected_vortex_lines(unique_vortex_lines,
                                                                                                   unique_eigenvalues)
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

    # Traverse connected_vortex_groups. For each group, use every 6 lines as a sound source.
    # The sound source point is at the end of the third point. # First select the third point from bottom to top,
    # and then use every 6 points as a sound source. If there are less than 6 points,
    # select the midpoint in the z direction as the sound source point.
    num_critical_sources = len(seed_points)
    length_per_source = 14
    i = num_critical_sources
    for index, group in enumerate(connected_vortex_groups):
        num_lines = len(group)
        if num_lines < 3:
            continue
        elif 3 <= num_lines < length_per_source:
            line_th = num_lines // 2
            # source_pos
            source_pos = group[line_th][0]
            seed_points.append(source_pos)

            i += 1
        else:

            j = 0
            remain_length_lines = len(group) - j * length_per_source
            while remain_length_lines >= length_per_source:
                source_pos = group[j * length_per_source + length_per_source // 2][0]
                seed_points.append(source_pos)
                j += 1
                remain_length_lines = len(group) - j * length_per_source
                i += 1


            if remain_length_lines:
                line_th = remain_length_lines // 2
                source_pos = group[j * length_per_source + line_th][0]
                seed_points.append(source_pos)
                i += 1

    #
    for seed_center in seed_points:

        seeds = vtk.vtkPointSource()
        seeds.SetCenter(seed_center)
        seeds.SetRadius(max_length * 0.05)
        seeds.SetNumberOfPoints(50)

        seeds.Update()
        seed_points_data = seeds.GetOutput().GetPoints()

        valid_seed_points = vtk.vtkPoints()

        for i in range(seed_points_data.GetNumberOfPoints()):
            point = seed_points_data.GetPoint(i)
            velocity = vectors.GetTuple(i)

            velocity_magnitude = np.linalg.norm(velocity)
            if velocity_magnitude > 1e-4:
                valid_seed_points.InsertNextPoint(point)

        if valid_seed_points.GetNumberOfPoints() == 0:
            print(f"No valid seed points with non-zero velocity near center {seed_center}")
            continue

        seed_center_np = np.array(seed_center)
        distances_to_critical_point = []

        for i in range(valid_seed_points.GetNumberOfPoints()):
            point = np.array(valid_seed_points.GetPoint(i))
            distance = np.linalg.norm(point - seed_center_np)
            distances_to_critical_point.append((distance, point))

        distances_to_critical_point.sort(key=lambda x: x[0])
        selected_seed_points = vtk.vtkPoints()

        for i in range(min(5, len(distances_to_critical_point))):
            selected_seed_points.InsertNextPoint(distances_to_critical_point[i][1])


        polydata = vtk.vtkPolyData()
        polydata.SetPoints(selected_seed_points)

        stream_tracer = vtk.vtkStreamTracer()
        stream_tracer.SetInputConnection(reader.GetOutputPort())
        stream_tracer.SetSourceData(polydata)
        stream_tracer.SetMaximumPropagation(1000)
        stream_tracer.SetInitialIntegrationStep(0.1)
        stream_tracer.SetIntegrationDirectionToForward()


        tube_filter = vtk.vtkTubeFilter()
        tube_filter.SetInputConnection(stream_tracer.GetOutputPort())
        tube_filter.SetRadius(0.03)
        tube_filter.SetNumberOfSides(20)

        append_filter.AddInputConnection(tube_filter.GetOutputPort())

    append_filter.Update()

    stream_mapper = vtk.vtkPolyDataMapper()
    stream_mapper.SetInputConnection(append_filter.GetOutputPort())

    lines = vtk.vtkActor()
    lines.SetMapper(stream_mapper)
    colors = vtk.vtkNamedColors()
    lines.GetProperty().SetColor(colors.GetColor3d("Gray"))

    return lines

def get_jacobian_at_point(image_data, point):

    origin = image_data.GetOrigin()
    spacing = image_data.GetSpacing()

    x_idx = int((point[0] - origin[0]) / spacing[0])
    y_idx = int((point[1] - origin[1]) / spacing[1])
    z_idx = int((point[2] - origin[2]) / spacing[2])

    coords = []
    vectors = []
    for k in range(2):
        for j in range(2):
            for i in range(2):
                x = x_idx + i
                y = y_idx + j
                z = z_idx + k
                coord = (
                    origin[0] + (x_idx + i) * spacing[0],
                    origin[1] + (y_idx + j) * spacing[1],
                    origin[2] + (z_idx + k) * spacing[2]
                )
                vector = [
                    image_data.GetScalarComponentAsDouble(x, y, z, 0),
                    image_data.GetScalarComponentAsDouble(x, y, z, 1),
                    image_data.GetScalarComponentAsDouble(x, y, z, 2)
                ]
                coords.append(coord)
                vectors.append(vector)

    coeffs_u, coeffs_v, coeffs_w = trilinear_interpolation_coefficients(vectors, coords)

    interpolated_u = trilinear_interpolation_function(coeffs_u, point[0], point[1], point[2])
    interpolated_v = trilinear_interpolation_function(coeffs_v, point[0], point[1], point[2])
    interpolated_w = trilinear_interpolation_function(coeffs_w, point[0], point[1], point[2])

    return [interpolated_u, interpolated_v, interpolated_w]


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
        if r>0:
            positive_real.append(r)
        elif r<0:
            negative_real.append(r)

    if len(positive_real)==3:
        return (234 / 255, 67 / 255, 53 / 255)  # Orange for spiral source
    elif len(negative_real)==3:
        return (66 / 255, 133 / 255, 244 / 255)  # Purple for spiral sink
    elif len(positive_real) == 2 and len(negative_real) == 1:
        return (52 / 255, 168 / 255, 83 / 255)  # Light Blue for 1:2 spiral saddle
    elif len(positive_real) == 1 and len(negative_real) == 2:
        return (52 / 255, 168 / 255, 83 / 255)  # Pink for 2:1 spiral saddle
    # else:
    #     return (1.0, 0.75, 0.8)  # Pink for 2:1 spiral saddle

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

def check_eigenvalues2(J):

    # Check if the matrix A has one real eigenvalue and a pair of complex-conjugate eigenvalues.

    eigenvalues, eigenvectors = np.linalg.eig(J)
    real_eigenvalues = [val.real for val in eigenvalues if np.isreal(val)]
    complex_eigenvalues = [val for val in eigenvalues if np.iscomplex(val)]

    if len(real_eigenvalues) == 1 and len(complex_eigenvalues) == 2:
        if np.isclose(complex_eigenvalues[0], np.conj(complex_eigenvalues[1])):
            real_eigenvalue_index = np.where(eigenvalues == real_eigenvalues[0])[0][0]
            return True, eigenvectors[:, real_eigenvalue_index].real
    return False, None

def check_tetrahedron_for_zero_reduced_velocity2(vectors, coords):
    # Calculate interpolation coefficients
    coeffs_u, coeffs_v, coeffs_w = linear_interpolation_coefficients(vectors, coords)

    # Compute the Jacobian matrix
    jacobian = compute_jacobian2(coeffs_u, coeffs_v, coeffs_w)

    # Check eigenvalues and find the eigenvector corresponding to the real eigenvalue
    has_valid_eigenvalues, real_eigenvector = check_eigenvalues2(jacobian)

    zero_points = []
    if has_valid_eigenvalues:
        # Check each face for zero reduced velocity point
        for face in faces:
            face_coords = [coords[i] for i in face]
            point = find_zero_reduced_velocity_points(coeffs_u, coeffs_v, coeffs_w, face_coords, real_eigenvector)

            if point is not None:
                zero_points.append(point)
            #Deduplication
        zero_points = remove_duplicate_points(zero_points)

    if len(zero_points) == 2:
        point1, point2 = zero_points
        return np.array([point1, point2])
    else:
        return None

def read_vtu_file_and_process_cells2(file_path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()
    data = reader.GetOutput()

    points = data.GetPoints()
    # cells = data.GetCells()
    velocity_array = data.GetPointData().GetScalars("Vectors")

    vortex_lines = []
    vortex_line_points = vtk.vtkPoints()
    vortex_line_cells = vtk.vtkCellArray()

    for i in range(data.GetNumberOfCells()):
        cell = data.GetCell(i)
        if cell.GetCellType() != vtk.VTK_TETRA:
            continue

        cell_points = cell.GetPoints()
        coords = [points.GetPoint(cell.GetPointId(j)) for j in range(cell_points.GetNumberOfPoints())]
        vectors = [velocity_array.GetTuple(cell.GetPointId(j)) for j in range(cell_points.GetNumberOfPoints())]

        # try:
        vortex_line = check_tetrahedron_for_zero_reduced_velocity2(vectors, coords)
        if vortex_line is not None:
            vortex_lines.append(vortex_line)
            # Add points to vtkPoints and create a line cell
            id1 = vortex_line_points.InsertNextPoint(vortex_line[0])
            id2 = vortex_line_points.InsertNextPoint(vortex_line[1])
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, id1)
            line.GetPointIds().SetId(1, id2)
            vortex_line_cells.InsertNextCell(line)
            print(f"Found vortex line at cell {i}: {vortex_line}")

    return vortex_lines, vortex_line_points, vortex_line_cells



def vector_field_actors(filename,indices_to_remove={}):
    #read .vti file
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()

    image_data = reader.GetOutput()
    print('image_data.GetDimensions()', image_data.GetDimensions())

    # visualize critical points
    critical_points, jacobians, eigenvalues_list, eigenvectors_list = extract_critical_points_and_jacobians(image_data)

    # define the indexes to be removed
    # indices_to_remove = {3, 4}

    # Filter the elements whose indices are 3 and 4
    critical_points = [point for i, point in enumerate(critical_points) if i not in indices_to_remove]
    jacobians = [jacobian for i, jacobian in enumerate(jacobians) if i not in indices_to_remove]
    eigenvalues_list = [eigenvalues for i, eigenvalues in enumerate(eigenvalues_list) if i not in indices_to_remove]
    eigenvectors_list = [eigenvectors for i, eigenvectors in enumerate(eigenvectors_list) if i not in indices_to_remove]

    # critical_points, jacobians, eigenvalues_list, eigenvectors_list = [],[], [], []

    critical_points_actors = []
    for idx, point in enumerate(critical_points):
        eigenvalues = eigenvalues_list[idx]
        color = get_color_for_eigenvalues(eigenvalues)

        # create a sphere to represent a critical point
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetCenter(point[0], point[1], point[2])
        sphere_source.SetRadius(0.5)

        # create a mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere_source.GetOutputPort())

        # create an actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        critical_points_actors.append(actor)

    tetra_filename = 'vortex_core_line.vtu'
    tvtu_file = hexahedron_to_tetrahedra(filename, tetra_filename)

    #streamline
    lines = visualize_vti_file_with_streamlines(filename)
    # lines = tf.visualize_vti_file_with_streamlines(filename)

    # vortex core line
    vortex_lines, vortex_line_points, vortex_line_cells = read_vtu_file_and_process_cells2(tetra_filename)

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


    legend_positions = [
        (50, 1200),  # Position for "Source"
        (50, 1160),  # Position for "Sink"
        (50, 1120),  # Position for "Saddle"
    ]

    legend_actors = []
    selected_labels = ["Source", "Sink", "Saddle"]
    selected_colors = {
        "Source": (234 / 255, 67 / 255, 53 / 255),  # Red
        "Sink": (66 / 255, 133 / 255, 244 / 255),  # Blue
        "Saddle": (52 / 255, 168 / 255, 83 / 255),  # Green (统一颜色)
    }

    for i, label in enumerate(selected_labels):
        color = selected_colors[label]  # get the corresponding colors
        legend_actor = create_text_actor(label, legend_positions[i], color)
        legend_actors.append(legend_actor)


    return critical_points_actors, lines, vortex_lines_actor, legend_actors


def vector_field_actors_vortex(filename,indices_to_remove={}):
    #read .vti file
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()

    image_data = reader.GetOutput()
    print('image_data.GetDimensions()', image_data.GetDimensions())

    # visualize critical points
    critical_points, jacobians, eigenvalues_list, eigenvectors_list = extract_critical_points_and_jacobians(image_data)
    critical_points, jacobians, eigenvalues_list, eigenvectors_list = [],[], [], []

    critical_points_actors = []
    for idx, point in enumerate(critical_points):
        eigenvalues = eigenvalues_list[idx]
        color = get_color_for_eigenvalues(eigenvalues)


        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetCenter(point[0], point[1], point[2])
        sphere_source.SetRadius(0.5)


        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere_source.GetOutputPort())

        # actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        critical_points_actors.append(actor)

    tetra_filename = 'vortex_core_line.vtu'
    tvtu_file = hexahedron_to_tetrahedra(filename, tetra_filename)

    # visualize using streamline
    lines = visualize_vti_file_with_streamlines(filename)
    # lines = tf.visualize_vti_file_with_streamlines(filename)

    # visualize vortex core line
    vortex_lines, vortex_line_points, vortex_line_cells = read_vtu_file_and_process_cells2(tetra_filename)

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

    # add legend
    legend_positions = [
        (50, 1200),  # Position for the first label (x, y)
        (50, 1160),  # Position for the second label
        (50, 1120),
        (50, 1080),
        (50, 1040),
        (50, 1000),
        (50, 960),
        (50, 920)
    ]

    labels = [
        ("Source", (1.0, 0.0, 0.0)),  # Red
        ("Sink", (0.0, 0.0, 1.0)),  # Blue
        ("1:2 Saddle", (0.0, 1.0, 0.0)),  # Green
        ("2:1 Saddle", (0.0, 1.0, 1.0)),  # Cyan
        ("Spiral Source", (1.0, 0.5, 0.0)),  # Orange
        ("Spiral Sink", (0.5, 0.0, 1.0)),  # Purple
        ("2:1 Spiral Saddle", (1.0, 0.75, 0.8)),  # pink
        ("1:2 Spiral Saddle", (0.5, 0.5, 1.0))  # Light Blue
    ]
    legend_actors = []
    for i, (text, color) in enumerate(labels):
        legend_actor = create_text_actor(text, legend_positions[i], color)
        legend_actors.append(legend_actor)

    return critical_points_actors, lines, vortex_lines_actor, legend_actors


def get_frequency_for_eigenvalues(eigenvalues):
    # define frequency mapping
    frequencies = {
        'Sink': 800,
        '2:1 Saddle': 600,
        '1:2 Saddle': 400,
        'Source': 200
    }

    real_parts = np.real(eigenvalues)
    # Sort real_parts from small to large
    sorted_real_parts = np.sort(real_parts)[::-1]
    # sorted_real_parts = np.sort(real_parts)

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
    a = frequencies[point_type]
    print('frequencies[point_type]:',a)

    return frequencies[point_type]



def get_frequencies_for_eigenvalues2(eigenvalues):
    import numpy as np

    eigenvalues = np.array(eigenvalues)


    real_parts = np.real(eigenvalues)


    pos_min_freq, pos_max_freq = 200, 400
    neg_min_freq, neg_max_freq = 800, 1000
    zero_freq = 600


    positive_real_parts = real_parts[real_parts > 0]
    zero_real_parts = real_parts[real_parts == 0]
    negative_real_parts = real_parts[real_parts < 0]

    if positive_real_parts.size > 0:
        max_pos_real = np.max(positive_real_parts)
        min_pos_real = np.min(positive_real_parts)
    else:
        max_pos_real = min_pos_real = None

    if negative_real_parts.size > 0:
        max_neg_real = np.max(np.abs(negative_real_parts))
        min_neg_real = np.min(np.abs(negative_real_parts))
    else:
        max_neg_real = min_neg_real = None

    frequencies = np.zeros_like(real_parts)

    # Frequency mapping is performed on positive eigenvalues. The larger the absolute value, the lower the frequency.
    if positive_real_parts.size > 0:
        if max_pos_real == min_pos_real:
            frequencies[real_parts > 0] = (pos_min_freq + pos_max_freq) / 2
        else:
            pos_scale = (pos_max_freq - pos_min_freq) / (max_pos_real - min_pos_real)
            frequencies[real_parts > 0] = pos_max_freq - (real_parts[real_parts > 0] - min_pos_real) * pos_scale

    # Frequency mapping perform on negative eigenvalues.
    # The larger the absolute value, the higher the frequency.
    if negative_real_parts.size > 0:
        if max_neg_real == min_neg_real:  # If all negative eigenvalues are equal
            frequencies[real_parts < 0] = (neg_min_freq + neg_max_freq) / 2
        else:
            neg_scale = (neg_max_freq - neg_min_freq) / (max_neg_real - min_neg_real)
            frequencies[real_parts < 0] = neg_min_freq + (np.abs(real_parts[real_parts < 0]) - min_neg_real) * neg_scale

    # Assign a fixed frequency of 600 to eigenvalue 0
    frequencies[real_parts == 0] = zero_freq

    return frequencies


def get_radiuses_for_eigenvalues2(eigenvalues):
    eigenvalues = np.array(eigenvalues)


    real_parts = np.real(eigenvalues)

    pos_min_freq, pos_max_freq = 0.025, 0.017
    neg_min_freq, neg_max_freq = 0.009, 0.001

    positive_real_parts = real_parts[real_parts > 0]
    negative_real_parts = real_parts[real_parts < 0]

    if positive_real_parts.size > 0:
        max_pos_real = np.max(positive_real_parts)
        min_pos_real = np.min(positive_real_parts)
    else:
        max_pos_real = min_pos_real = None

    if negative_real_parts.size > 0:
        max_neg_real = np.max(np.abs(negative_real_parts))
        min_neg_real = np.min(np.abs(negative_real_parts))
    else:
        max_neg_real = min_neg_real = None

    frequencies = np.zeros_like(real_parts)

    if positive_real_parts.size > 0:
        pos_scale = (pos_max_freq - pos_min_freq) / (max_pos_real - min_pos_real)
        frequencies[real_parts > 0] = pos_max_freq - (real_parts[real_parts > 0] - min_pos_real) * pos_scale

    if negative_real_parts.size > 0:
        neg_scale = (neg_max_freq - neg_min_freq) / (max_neg_real - min_neg_real)
        frequencies[real_parts < 0] = neg_min_freq + (np.abs(real_parts[real_parts < 0]) - min_neg_real) * neg_scale

    return frequencies

def get_modes_for_eigenvalues(eigenvalues):
    eigenvalues = np.array(eigenvalues)

    real_parts = np.real(eigenvalues)

    pos_min_freq, pos_max_freq = 8, 14
    neg_min_freq, neg_max_freq = 19, 25

    positive_real_parts = real_parts[real_parts > 0]
    negative_real_parts = real_parts[real_parts < 0]

    if positive_real_parts.size > 0:
        max_pos_real = np.max(positive_real_parts)
        min_pos_real = np.min(positive_real_parts)
    else:
        max_pos_real = min_pos_real = None

    if negative_real_parts.size > 0:
        max_neg_real = np.max(np.abs(negative_real_parts))
        min_neg_real = np.min(np.abs(negative_real_parts))
    else:
        max_neg_real = min_neg_real = None

    frequencies = np.zeros_like(real_parts)

    if positive_real_parts.size > 0:
        pos_scale = (pos_max_freq - pos_min_freq) / (max_pos_real - min_pos_real)
        frequencies[real_parts > 0] = pos_max_freq - (real_parts[real_parts > 0] - min_pos_real) * pos_scale

    if negative_real_parts.size > 0:
        neg_scale = (neg_max_freq - neg_min_freq) / (max_neg_real - min_neg_real)
        frequencies[real_parts < 0] = neg_min_freq + (np.abs(real_parts[real_parts < 0]) - min_neg_real) * neg_scale

    return frequencies



def adjust_length(sound_data, target_length):

    current_length = len(sound_data)

    if current_length < int(target_length):
        repeat_times = int(target_length // current_length + 1)
        sound_data = np.tile(sound_data, repeat_times)
    adjusted_sound_data = sound_data[:int(target_length)]

    return adjusted_sound_data


def resampling(freq_ratio, signal, sampling_rate):
    # get timeline according to signal and its sampling rate
    time_points = np.linspace(0, len(signal) / sampling_rate, len(signal), endpoint=False)

    # Increase/decrease the time size of the time point to 1/freq_ratio times of the original time point,
    # and the corresponding signal value remains unchanged
    new_time_points = time_points / freq_ratio

    # Generate a new timeline with the same sampling rate as the original signal
    new_duration = new_time_points[-1]
    new_length = int(new_duration * sampling_rate)  # the number of sampling points of the new timeline

    # resampling
    resampling_time_points = np.linspace(0, new_duration, new_length)

    spl = CubicSpline(new_time_points, signal)

    resampling_signal = spl(resampling_time_points)

    return resampling_signal

def generate_sound_for_vortex_core_line(eigenvalues, filename, max_ab_imag_eigenvalue, min_ab_imag_eigenvalue,new_min,
                                        new_max, signal_duration):
    sample_rate, sound_data = wavfile.read(filename)

    for number in eigenvalues:
        if np.iscomplex(number):
            chosen_complex_number = number
            break
    imag_part = np.imag(chosen_complex_number)

    imag_abs = np.abs(imag_part)

    mapped_speed = new_min + (imag_abs - min_ab_imag_eigenvalue) * (new_max - new_min) / (max_ab_imag_eigenvalue - min_ab_imag_eigenvalue)

    mapped_speed = round(mapped_speed, 2)
    if max_ab_imag_eigenvalue == min_ab_imag_eigenvalue:
        mapped_speed = 1.0
    # new_sound_data = change_playback_speed(sound_data, mapped_speed)
    new_sound_data = resampling(freq_ratio=mapped_speed,signal=sound_data, sampling_rate=sample_rate)


    target_length_seconds = signal_duration
    target_length = target_length_seconds * sample_rate

    final_sound_data = adjust_length(new_sound_data, target_length)

    return final_sound_data

def generate_spherical_coordinates_cartesian(radius, theta_resolution, phi_resolution, center):

    sphereSource = vtk.vtkSphereSource()

    sphereSource.SetThetaResolution(theta_resolution)  #
    sphereSource.SetPhiResolution(phi_resolution)  #
    sphereSource.SetRadius(radius)  #


    sphereSource.Update()


    points = sphereSource.GetOutput().GetPoints()

    cartesian_coordinates = []

    for i in range(points.GetNumberOfPoints()):
        x, y, z = points.GetPoint(i)


        x += center[0]
        y += center[1]
        z += center[2]

        cartesian_coordinates.append((x, y, z))

    return cartesian_coordinates


import numpy as np


def trilinear_interpolation_coefficients(vectors, coords):
    # Calculate the coefficients for the trilinear interpolation function.

    A = np.zeros((8, 8))
    B_u = np.zeros(8)
    B_v = np.zeros(8)
    B_w = np.zeros(8)

    for idx in range(8):
        x, y, z = coords[idx]
        A[idx] = [1, x, y, z, x * y, x * z, y * z, x * y * z]
        B_u[idx] = vectors[idx][0]
        B_v[idx] = vectors[idx][1]
        B_w[idx] = vectors[idx][2]

    coeffs_u = np.linalg.solve(A, B_u)
    coeffs_v = np.linalg.solve(A, B_v)
    coeffs_w = np.linalg.solve(A, B_w)

    return coeffs_u, coeffs_v, coeffs_w


def calculate_vector_at_point(coeffs_u, coeffs_v, coeffs_w, x, y, z):

    # Perform trilinear interpolation to get the vector value at a given point.
    # coeffs_u, coeffs_v, coeffs_w: Coefficients for trilinear interpolation for each vector component.
    # x, y, z: The coordinates of the point where interpolation is performed.

    interpolated_u = coeffs_u[0] + coeffs_u[1] * x + coeffs_u[2] * y + coeffs_u[3] * z + \
                     coeffs_u[4] * x * y + coeffs_u[5] * x * z + coeffs_u[6] * y * z + coeffs_u[7] * x * y * z
    interpolated_v = coeffs_v[0] + coeffs_v[1] * x + coeffs_v[2] * y + coeffs_v[3] * z + \
                     coeffs_v[4] * x * y + coeffs_v[5] * x * z + coeffs_v[6] * y * z + coeffs_v[7] * x * y * z
    interpolated_w = coeffs_w[0] + coeffs_w[1] * x + coeffs_w[2] * y + coeffs_w[3] * z + \
                     coeffs_w[4] * x * y + coeffs_w[5] * x * z + coeffs_w[6] * y * z + coeffs_w[7] * x * y * z

    return np.array([interpolated_u, interpolated_v, interpolated_w])


def get_interpolated_vector_at_point(image_data, point):

   # Perform trilinear interpolation within a cubical grid to get the vector value at a given point.

    dims = image_data.GetDimensions()
    spacing = image_data.GetSpacing()
    origin = image_data.GetOrigin()

    vector_data = image_data.GetPointData().GetVectors()

    # Calculate indices and ensure they're within bounds
    x_idx = max(0, min(int((point[0] - origin[0]) / spacing[0]), dims[0] - 2))
    y_idx = max(0, min(int((point[1] - origin[1]) / spacing[1]), dims[1] - 2))
    z_idx = max(0, min(int((point[2] - origin[2]) / spacing[2]), dims[2] - 2))

    # Get coordinates and vector values at the 8 corner points of the cell
    coords = []
    vectors = []
    for k in range(2):
        for j in range(2):
            for i in range(2):
                x = x_idx + i
                y = y_idx + j
                z = z_idx + k
                coord = (
                    origin[0] + x * spacing[0],
                    origin[1] + y * spacing[1],
                    origin[2] + z * spacing[2]
                )
                vector_index = x + dims[0] * (y + dims[1] * z)
                vector = vector_data.GetTuple3(vector_index)
                vectors.append(vector)
                coords.append(coord)

    # Calculate trilinear interpolation coefficients
    coeffs_u, coeffs_v, coeffs_w = trilinear_interpolation_coefficients(vectors, coords)

    # Interpolate the vector at the given point
    interpolated_vector = calculate_vector_at_point(coeffs_u, coeffs_v, coeffs_w, point[0], point[1], point[2])

    return interpolated_vector

# ------------------------------------------vortex core line-----------------------------------------------------
def remove_duplicate_vortex_lines(vortex_lines, eigenvalues_for_each_core_line, tolerance=1e-6):

    #Remove duplicate vortex lines. Two lines are considered duplicates if their
    #start and end points are within the tolerance distance.

    unique_vortex_lines = []
    unique_eigenvalues = []

    for i, line in enumerate(vortex_lines):
        start_point, end_point = line[0], line[-1]
        is_duplicate = False

        for j, unique_line in enumerate(unique_vortex_lines):
            unique_start_point, unique_end_point = unique_line[0], unique_line[-1]

            # Check if both start and end points are within the tolerance distance
            if (np.linalg.norm(np.array(start_point) - np.array(unique_start_point)) <= tolerance and
                np.linalg.norm(np.array(end_point) - np.array(unique_end_point)) <= tolerance):
                is_duplicate = True
                break

        # If the line is not a duplicate, add it to the list of unique lines
        if not is_duplicate:
            unique_vortex_lines.append(line)
            unique_eigenvalues.append(eigenvalues_for_each_core_line[i])

    return unique_vortex_lines, unique_eigenvalues

def merge_connected_groups(connected_groups, connected_groups_eigenvalues, tolerance=1e-6):

    # Merge groups of connected vortex lines. Starting from the last group, check if the first line in the group
    # connects with the last line in the previous group. If connected, merge the groups. Continue this process
    # for all groups.

    def are_lines_connected(line1, line2, tolerance):
        #Check if two lines are connected based on their start and end points.
        start1, end1 = line1[0], line1[-1]
        start2, end2 = line2[0], line2[-1]

        return (np.linalg.norm(np.array(start1) - np.array(end2)) <= tolerance or
                np.linalg.norm(np.array(end1) - np.array(start2)) <= tolerance)

    merged_groups = connected_groups[:]  # Make a copy of the groups
    merged_groups_eigenvalues = connected_groups_eigenvalues[:]  # Make a copy of the eigenvalues

    # Start merging from the last group
    i = len(merged_groups) - 1
    while i>0:
        current_group = merged_groups[i]
        j = i-1
        # is_merged = False
        while j>=0:
            previous_group = merged_groups[j]
            first_line_in_current = current_group[0]
            last_line_in_previous = previous_group[-1]

            # Check if the first line in the current group connects with the last line in the previous group
            if are_lines_connected(first_line_in_current, last_line_in_previous, tolerance):
                # Merge the current group with the previous group
                merged_groups[j] += merged_groups[i]
                merged_groups_eigenvalues[j] += merged_groups_eigenvalues[i]  # Add the eigenvalues of the current group to the eigenvalues of the previous group
                # Remove the current group as it has been merged
                merged_groups.pop(i)
                merged_groups_eigenvalues.pop(i)
                # is_merged = True
                break
            else:
                j -= 1  # Move to the next pair of groups
        i = i-1

    # Do it again from bottom to top. This time, if the end point of the last line of the group being processed coincides
    # with the starting point of the first line of the previous group, add the current line to the front of the group
    # that meets the conditions.    # Second pass: from bottom to top, check end-to-start connections
    i = len(merged_groups) - 1
    while i > 0:
        current_group = merged_groups[i]
        j = i - 1
        while j >= 0:
            previous_group = merged_groups[j]
            last_line_in_current = current_group[-1]
            first_line_in_previous = previous_group[0]

            # Check if the last line in the current group connects with the first line in the previous group
            if are_lines_connected(last_line_in_current, first_line_in_previous, tolerance):
                # Prepend the current group to the previous group
                merged_groups[j] = current_group + merged_groups[j]
                merged_groups_eigenvalues[j] += merged_groups_eigenvalues[i]
                # Remove the current group as it has been merged
                merged_groups.pop(i)
                merged_groups_eigenvalues.pop(i)
                break
            else:
                j -= 1
        i -= 1

    return merged_groups,merged_groups_eigenvalues

def group_connected_vortex_lines(vortex_lines,eigenvalues, tolerance=1e-6):

    # Group vortex lines based on their connectivity. Two lines are connected
    # if the start or end points of one line are within the tolerance distance of the
    # start or end points of another line.

    connected_groups = []
    connected_groups_eigenquals = []

    def are_lines_connected(line1, line2, tolerance):
        # Check if two lines are connected based on their start and end points.
        start1, end1 = line1[0], line1[-1]
        start2, end2 = line2[0], line2[-1]

        return (np.linalg.norm(np.array(start1) - np.array(end2)) <= tolerance or
                np.linalg.norm(np.array(end1) - np.array(start2)) <= tolerance)

    for index,line in enumerate(vortex_lines):
        added_to_group = False
        # Check if the current line connects with any existing group
        for group_index, group in enumerate(connected_groups):
            if any(are_lines_connected(line, group_line, tolerance) for group_line in group):
                group.append(line)
                connected_groups_eigenquals[group_index].append(eigenvalues[index])
                added_to_group = True
                break

        # If the line does not connect to any existing group, create a new group
        if not added_to_group:
            connected_groups.append([line])
            connected_groups_eigenquals.append([eigenvalues[index]])

    # Merge the obtained groups: Go backwards (starting from the last group) and check whether the starting point of
    # the first line of each group is the same as the end point of the last line of the other group.
    # If so, merge the two groups into one
    # Merge the connected vortex groups
    merged_vortex_groups, merged_vortex_groups_eigenvalues = merge_connected_groups(connected_groups,connected_groups_eigenquals)

    return merged_vortex_groups,merged_vortex_groups_eigenvalues




def generate_planes(image_data=None, selected_bounds=None, custom_planes=None):
    """
    Generate planes based on bounds
    """
    planes = []

    # Generate planes based on bounds
    if image_data and (selected_bounds or selected_bounds is None):
        bounds = image_data.GetBounds()  # Get the bounding box of the image data

        all_planes = {
            "Z-min": [(bounds[0], bounds[2], bounds[4]), (bounds[1], bounds[2], bounds[4]),
                      (bounds[1], bounds[3], bounds[4]), (bounds[0], bounds[3], bounds[4])],
            "Z-max": [(bounds[0], bounds[2], bounds[5]), (bounds[1], bounds[2], bounds[5]),
                      (bounds[1], bounds[3], bounds[5]), (bounds[0], bounds[3], bounds[5])],
            "Y-min": [(bounds[0], bounds[2], bounds[4]), (bounds[1], bounds[2], bounds[4]),
                      (bounds[1], bounds[2], bounds[5]), (bounds[0], bounds[2], bounds[5])],
            "Y-max": [(bounds[0], bounds[3], bounds[4]), (bounds[1], bounds[3], bounds[4]),
                      (bounds[1], bounds[3], bounds[5]), (bounds[0], bounds[3], bounds[5])],
            "X-min": [(bounds[0], bounds[2], bounds[4]), (bounds[0], bounds[3], bounds[4]),
                      (bounds[0], bounds[3], bounds[5]), (bounds[0], bounds[2], bounds[5])],
            "X-max": [(bounds[1], bounds[2], bounds[4]), (bounds[1], bounds[3], bounds[4]),
                      (bounds[1], bounds[3], bounds[5]), (bounds[1], bounds[2], bounds[5])]
        }

        # If no specific bounds are selected, generate all planes
        if selected_bounds is None:
            selected_bounds = ["Z-min", "Z-max", "Y-min", "Y-max", "X-min", "X-max"]

        # Append planes based on selected bounds
        planes.extend([all_planes[bound] for bound in selected_bounds if bound in all_planes])

    # Add custom planes if provided
    if custom_planes:
        for plane in custom_planes:
            if len(plane) == 4:  # Ensure each custom plane has 4 points
                planes.append(plane)
            else:
                raise ValueError(f"Each custom plane must have exactly 4 points. Got: {plane}")

    return planes


def divide_planes_into_triangles(planes, max_triangle_area):
    # Divide planes into triangles and ensure the area of each triangle is below max_triangle_area.


    def calculate_triangle_area(points):
        p0, p1, p2 = points
        vec1 = np.array(p1) - np.array(p0)
        vec2 = np.array(p2) - np.array(p0)
        cross_product = np.cross(vec1, vec2)
        area = np.linalg.norm(cross_product) / 2.0
        return area

    def subdivide_triangle(points):
        p0, p1, p2 = points
        mid01 = (np.array(p0) + np.array(p1)) / 2.0
        mid12 = (np.array(p1) + np.array(p2)) / 2.0
        mid20 = (np.array(p2) + np.array(p0)) / 2.0
        return [
            (p0, mid01.tolist(), mid20.tolist()),
            (mid01.tolist(), p1, mid12.tolist()),
            (mid01.tolist(), mid12.tolist(), mid20.tolist()),
            (mid20.tolist(), mid12.tolist(), p2)
        ]

    def subdivide_large_triangles(triangles, max_area):
        subdivided = []
        for triangle in triangles:
            queue = [triangle]
            while queue:
                tri = queue.pop()
                if calculate_triangle_area(tri) > max_area:
                    queue.extend(subdivide_triangle(tri))
                else:
                    subdivided.append(tri)
        return subdivided

    triangles = {}
    triangle_index = 0

    for plane_points in planes:
        # Define initial two triangles for the plane
        initial_triangles = [
            (plane_points[0], plane_points[1], plane_points[2]),
            (plane_points[0], plane_points[2], plane_points[3])
        ]

        # Subdivide triangles if necessary
        subdivided_triangles = subdivide_large_triangles(initial_triangles, max_triangle_area)

        # Store each triangle with its center
        for tri in subdivided_triangles:
            center = np.mean(tri, axis=0)
            triangles[f'triangle{triangle_index}'] = {
                'points': [tuple(p) for p in tri],
                'center': tuple(center)
            }
            triangle_index += 1

    return triangles

def create_plane_actors(planes):
    # Generate vtkActor objects for each plane defined by corner points.

    plane_actors = []
    for plane_points in planes:
        # Create a plane source
        plane_source = vtk.vtkPlaneSource()
        plane_source.SetOrigin(*plane_points[0])
        plane_source.SetPoint1(*plane_points[1])
        plane_source.SetPoint2(*plane_points[3])
        plane_source.Update()

        # Mapper for the plane
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane_source.GetOutputPort())

        # Actor for the plane
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.8, 0.8, 0.8)  # Light gray for the planes
        actor.GetProperty().SetOpacity(0.5)  # Semi-transparent for better visualization
        plane_actors.append(actor)

    return plane_actors

def smooth_transition_signal(signal_a, signal_b, freq_a, freq_b, sampling_rate, duration_transition):
    # get the transition from neutral bubbles' frequerncy to deformed bubbles
    # calculate durations according to the signal and sampling rate
    duration_a = len(signal_a) / sampling_rate
    duration_b = len(signal_b) / sampling_rate

    # timeline
    t_a = np.linspace(0, duration_a, len(signal_a), endpoint=False)
    t_transition = np.linspace(0, duration_transition, int(sampling_rate * duration_transition), endpoint=False)
    t_b = np.linspace(0, duration_b, len(signal_b), endpoint=False)

    # Get the 2/3 length element of a
    index_a_2_3 = len(signal_a)*2//3

    # Get the 1/10 length element of b
    index_b_1_10 = len(signal_b)//3

    # cut a and b
    signal_a_trimmed = signal_a[:index_a_2_3]
    signal_b_trimmed = signal_b[index_b_1_10:]

    #Get the last and first element of trimmed a and trimmed b
    last_value_trimmed_a = signal_a_trimmed[-1]
    first_value_trimmed_b = signal_b_trimmed[0]

    # construct transition signal
    transition_signal = np.zeros_like(t_transition)
    transition_length = len(t_transition)

    # the first 1/20 and last 1/20 length
    first_third_length = transition_length // 20
    last_third_start = transition_length * 19 // 20

    freq_transition_length = last_third_start-first_third_length+1
    freq_t_transition = np.linspace(0, freq_transition_length/sampling_rate, int(freq_transition_length), endpoint=False)

    # define the frequency of the transition signal, and from freq_a to freq_b smoothly
    freq_transition = np.linspace(freq_a, freq_b, freq_transition_length)
    # transition_signal[:] = np.sin(2 * np.pi * freq_transition * t_transition)

    def calculate_omega_n(f_n):
        omega_n = np.pi * f_n
        return omega_n

    def calculate_p_n(t, r0, omega_n):
        # betan = calculate_beta_n(n=15, nu=nu, rho=RHO_WATER, r0=r0)
        betan = 0.8
        t = 0.6 * t
        p_n = np.exp(-betan * t) * np.cos(2 * omega_n * t)

        return p_n
    transition_omega_n = calculate_omega_n(freq_transition)
    # transition_signal[first_third_length:last_third_start+1] = np.sin(2 * np.pi * freq_transition * freq_t_transition)
    transition_signal[first_third_length:last_third_start + 1] = calculate_p_n(freq_t_transition,0.005,transition_omega_n)
    # transition_signal[first_third_length:last_third_start + 1] = np.sin(2 * np.pi * freq_transition * freq_t_transition)

    # The first 1/3 is interpolated from the last non-zero value of signal a to the value at 1/3 of the transition signal
    transition_signal[:first_third_length] = np.linspace(
        last_value_trimmed_a, transition_signal[first_third_length], first_third_length
    )

    # The last 1/3 is interpolated from the value at 2/3 of the transition signal to the first non-zero value of signal b
    transition_signal[last_third_start:] = np.linspace(
        transition_signal[last_third_start], first_value_trimmed_b, transition_length - last_third_start
    )


    final_signal = np.concatenate((signal_a_trimmed, transition_signal, signal_b_trimmed))

    return final_signal

def distance(point1,point2):
    return np.sqrt(sum([(p1-p2)**2 for p1,p2 in zip(point1,point2)]))


