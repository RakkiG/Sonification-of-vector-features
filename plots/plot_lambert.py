import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Arc

radius = 4
center = (0, 0)

theta = np.linspace(0, 2 * np.pi, 100)
x = center[0] + radius * np.cos(theta)
y = center[1] + radius * np.sin(theta)

y_line = -4
x_range = np.linspace(-radius-3, radius+3, 100)

y_upper = y_line
y_lower = y_line - 1

plt.figure(figsize=(6, 6))
# plt.plot(x, y, label="Circle", color="black")

plt.fill_between(x_range, y_lower, y_upper, color='red', alpha=0.3, label="Tangent Rectangle")

plt.gca().set_aspect('equal', adjustable='box')


plt.axis('off')
plt.grid(False)

x_values = [-0.5, -0.5]
y_values = [-3.7, -4.3]

x_values2 = [0.5, 0.5]
y_values2 = [-3.7, -4.3]

x_values5 = [-0.5, 0.5]
y_values5 = [-4, -4]

x_values3 = [0, 0]
y_values3 = [-4, 8]

x_values4 = [0, -3.4]
y_values4 = [-4, 2]

x_values6 = [-0.2, 2]
y_values6 = [-4, -6]

plt.plot(x_values, y_values, label="Line Segment", color="green")
plt.plot(x_values2, y_values2, label="Line Segment", color="green")
plt.plot(x_values3, y_values3, label="Line Segment", color="black")
plt.plot(x_values4, y_values4, label="Line Segment", color="black")
plt.plot(x_values5, y_values5, label="Line Segment", color="green")
plt.plot(x_values6, y_values6, label="Line Segment", color="green")

plt.text(2.7, -6.7, 'dS', fontsize=12, ha='center', color='green')

start_point = (-5.1, 5)
end_point = (-3.35, 1.95)
plt.annotate('', xy=end_point, xytext=start_point,
             arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=1.5))

start_point2 = (0, -4)
end_point2 = (4, 0)
plt.annotate('', xy=end_point2, xytext=start_point2,
             arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=1.5))

r = 1
c= (0,-4)

arc = Arc(c, width=2*r, height=2*r, angle=0, theta1=40, theta2=90, color='red', lw=2)
plt.gca().add_patch(arc)

r = 1.2
c= (0,-4)

arc = Arc(c, width=2*r, height=2*r, angle=0, theta1=90, theta2=120, color='blue', lw=2)
plt.gca().add_patch(arc)

plt.text(-0.6, -1.8, r'$\varphi_0$', fontsize=12, ha='center', color='blue')
plt.text(0.5, -2.3, r'$\varphi$', fontsize=12, ha='center', color='red')


plt.savefig('diagrams/lambert.png', bbox_inches='tight')
output_path = 'diagrams/lambert.png'
with Image.open(output_path) as img:
    cropped = img.crop((110, 110, img.width - 100, img.height))
    cropped.save(output_path)

plt.show()
