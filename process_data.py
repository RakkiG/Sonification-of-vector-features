import numpy as np

# Critical points from your vector field
required_points = [
    (10, 10, 10),
    (30, 10, 10),
    (50, 10, 10),
    (70, 10, 10),
    (10, 30, 10),
    (30, 30, 10),
    (50, 30, 10),
    (70, 30, 10),
    (10, 50, 10),
    (30, 50, 10),
    (50, 50, 10),
    (70, 50, 10),
    (10, 70, 10),
    (20, 70, 10)
]

# Example critical_points array (replace with your actual array)
critical_points = [
    (11, 11, 10),  # Close to (10, 10, 10)
    (30, 11, 10),
    (51, 10, 10),
    (71, 10, 10),
    (12, 31, 10),
    (32, 30, 10),
    (49, 29, 10),
    (68, 31, 10),
    (9, 51, 10),
    (31, 50, 11),
    (52, 50, 10),
    (70, 51, 10),
    (11, 71, 9),
    (19, 69, 10)
]

# Function to find the closest point
def find_closest_points(required_points, critical_points):
    closest_points = []
    critical_points_np = np.array(critical_points)  # Convert to NumPy array for efficient computation

    for point in required_points:
        # Calculate distances between the current point and all points in critical_points
        distances = np.linalg.norm(critical_points_np - np.array(point), axis=1)
        closest_idx = np.argmin(distances)  # Index of the closest point
        closest_points.append(tuple(critical_points_np[closest_idx]))  # Add the closest point

    return closest_points

# Find all closest points
closest_points = find_closest_points(required_points, critical_points)

# Display results
for i, (required, found) in enumerate(zip(required_points, closest_points)):
    print(f"Required point {required} -> Found point {found}")
