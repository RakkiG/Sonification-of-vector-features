import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

#
microphone_position = np.array([2, 3, 1])
d = 2
camera_position = np.array([microphone_position[0] - d,
                            microphone_position[1] - d,
                            microphone_position[2] + d])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.bar3d(microphone_position[0] - 0.5, microphone_position[1] - 0.5, microphone_position[2] - 0.5,
          1, 1, 1, color='b', alpha=0.6)

ax.bar3d(camera_position[0] - 0.5, camera_position[1] - 0.5, camera_position[2] - 0.5,
          1, 1, 1, color='gray', alpha=0.6)

ax.plot([microphone_position[0], camera_position[0]],
        [microphone_position[1], camera_position[1]],
        [microphone_position[2], camera_position[2]], 'k--')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.set_zlim([0, 10])


ax.text(microphone_position[0]+1, microphone_position[1], microphone_position[2], 'Microphone', color='b')
ax.text(camera_position[0], camera_position[1], camera_position[2]+1, 'Camera', color='gray')

os.makedirs('diagram', exist_ok=True)

plt.savefig('diagrams/microphone_camera.png')
output_path = 'diagrams/microphone_camera.png'

with Image.open(output_path) as img:
    cropped = img.crop((155, 80, img.width - 100, img.height - 30))
    cropped.save(output_path)

plt.show()
