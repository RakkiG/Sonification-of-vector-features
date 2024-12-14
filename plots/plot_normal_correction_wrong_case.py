
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# Ensure the diagrams folder exists
output_dir = "../diagrams"
os.makedirs(output_dir, exist_ok=True)

# Save the plot to the diagrams folder
output_path = os.path.join(output_dir, "incorrect_normal_direction.png")

# Create a 3D visualization for incorrect normal direction
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Triangle vertices in 3D
V0 = np.array([0, 0, 0])
V1 = np.array([1, 0, 0])
V2 = np.array([0.5, 0.866, 0])

# Calculate the centroid of the triangle
C = (V0 + V1 + V2) / 3

# Incident point in 3D
P = np.array([0.3, 0.5, 0.5])

# Original normal vector (perpendicular to the triangle plane)
normal_vector = np.cross(V1 - V0, V2 - V0)
normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize

# Incorrect normal direction (opposite of correct direction)
incorrect_normal = -normal_vector

# Vector from centroid to P
centroid_to_P = P - C

# Plot the triangle
triangle = np.array([V0, V1, V2, V0])  # Loop back to close the triangle
ax.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], color='blue', label="Triangle")

# Plot the centroid
ax.scatter(*C, color='green', label="Centroid (C)", s=50)

# Plot the incident point
ax.scatter(*P, color='red', label="Incident Point (P)", s=50)

# Draw the incorrect normal vector
ax.quiver(*C, *incorrect_normal, color='orange', length=0.4, label="Incorrect Normal (n)", normalize=True)

# Draw the vector from centroid to P
ax.quiver(*C, *(P - C), color='cyan', length=0.57, label="Vector to Incident Point (P-C)", normalize=True)

# Remove axes, ticks, and grid
ax.axis('off')
# ax.set_title("Incorrect Normal Direction (Different Side)")

# Customize legend: Remove border
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.2, 0.7),fontsize=12.5)  # Move legend downwards
# Save the figure
plt.savefig(output_path)
print(f"Figure saved to {output_path}")

with Image.open(output_path) as img:
    cropped = img.crop((180, 240, img.width - 370, img.height - 90))  # 自定义裁剪范围
    cropped.save(output_path)

plt.show()
