

import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

plt.figure(figsize=(4, 4), dpi=300)
ax = plt.axes(projection='3d')

r = 5
z = np.linspace(-15, 15, 1000)
theta = np.linspace(0, 6 * np.pi, 1000)

x = r * np.cos(theta)
y = r * np.sin(theta)

ax.plot([0, 0], [0, 0], [-20, 20], color='b', lw=3)


ax.plot(x, y, z, color='k', lw=2)

ax.set_xlim([x.min(), x.max()])
ax.set_ylim([y.min(), y.max()])
ax.set_zlim([z.min(), z.max()])


ax.grid(False)
ax.set_axis_off()

ax.plot([x[-1] - 0.6, x[-1]], [y[-1] - 0.6, y[-1]], [z[-1], z[-1]], color='black', lw=2)
ax.plot([x[-1], x[-1]], [y[-1], y[-1]], [z[-1] - 4, z[-1]], color='black', lw=2)

output_dir = "../diagrams"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

file_path = os.path.join(output_dir, "3d_arrow_plot.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0)

with Image.open(file_path) as img:
    cropped = img.crop((180, 180, img.width - 150, img.height - 150) )
    cropped.save(file_path)

plt.close()

print(f"Image saved to {file_path}")
