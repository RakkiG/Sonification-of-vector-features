import numpy as np
import matplotlib.pyplot as plt
import math

from PIL import Image


f1 = 2  # 500 Hz
f2 = 1  # 250 Hz
f3 = 4  # 1000 Hz

# The unified time interval is [0, 1], and each signal draws a 1-second signal
t1 = np.linspace(0, 1, 1000)  # The time interval is [0, 1], 1000 sampling points
t2 = np.linspace(0, 1, 1000)
t3 = np.linspace(0, 1, 1000)

signal_1 = np.sin(2 * np.pi * f1 * t1)
signal_2 = np.sin(2 * np.pi * f2 * t2)  # 250Hz
signal_3 = np.sin(2 * np.pi * f3 * t3)  # 1000 Hz
pi = math.pi

# Define the x values that need to be marked (corresponding to 0, π/4, π/2, π, 3π/4, 2π, etc.)
x_marks = np.array([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, pi, 5 * pi / 4, 3 * pi / 2, 7 * pi / 4, 2 * pi])
t_marks1 = x_marks / (2 * pi * f1)
t_marks2 = x_marks / (2 * pi * f2)
t_marks3 = x_marks / (2 * pi * f3)

# y
y_marks_1 = np.sin(2 * np.pi * f1 * np.array(t_marks1))
y_marks_2 = np.sin(2 * np.pi * f2 * np.array(t_marks2))
y_marks_3 = np.sin(2 * np.pi * f3 * np.array(t_marks3))

# Translate along the y axis
shift_1 = 0  # 500 Hz
shift_2 = -4  # 250 Hz -4
shift_3 = -8  # 1000 Hz  -8


plt.figure(figsize=(10, 8))

# plot 500 Hz
plt.plot(t1, signal_1 + shift_1, label='Original Signal', color='b')
plt.scatter(t_marks1, y_marks_1 + shift_1, color='k')  # mark points

# plot 250 Hz
plt.plot(t2, signal_2 + shift_2, label='2 times timeline scaling Signal', color='r')
plt.scatter(t_marks2, y_marks_2 + shift_2, color='k')

# Connect the points of the first signal with the points of the second signal with a dotted line
for i in range(len(t_marks1)):
    # get the points positions from signal 1 and 2
    x1 = t_marks1[i]
    y1 = y_marks_1[i] + shift_1

    x2 = t_marks2[i]
    y2 = y_marks_2[i] + shift_2

    #connect two points
    plt.plot([x1, x2], [y1, y2], 'k--', lw=1)

# plot 1000 Hz
plt.plot(t3, signal_3 + shift_3, label=f'$1/2$ times timeline scaling Signal', color='g')
plt.scatter(t_marks3, y_marks_3 + shift_3, color='k')  # mark points

# # Connect the points of the first signal and the third signal with a dotted line
for i in range(len(t_marks2)):

    x2 = t_marks1[i]
    y2 = y_marks_1[i] + shift_1

    x3 = t_marks3[i]
    y3 = y_marks_3[i] + shift_3


    plt.plot([x2, x3], [y2, y3], 'k--', lw=1)

plt.grid(False)

plt.axhline(y=y_marks_1[0] + shift_1, xmin=0, xmax=1, color='gray', linestyle='-', lw=1)  # Draw the horizontal axis
plt.axvline(x=t_marks1[0], ymin=0, ymax=1, color='gray', linestyle='-', lw=1)

# 250 Hz
plt.axhline(y=y_marks_2[0] + shift_2, xmin=0, xmax=1, color='gray', linestyle='-', lw=1)
plt.axvline(x=t_marks2[0], ymin=0, ymax=1, color='gray', linestyle='-', lw=1)

# 1000 Hz
plt.axhline(y=y_marks_3[0] + shift_3, xmin=0, xmax=1, color='gray', linestyle='-', lw=1)
plt.axvline(x=t_marks3[0], ymin=0, ymax=1, color='gray', linestyle='--', lw=1)


plt.legend(loc='best', fontsize=18)


plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)

plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])

plt.tight_layout()
plt.savefig('diagrams/constant_resampling.png')

with Image.open('../diagrams/constant_resampling.png') as img:
    cropped = img.crop((180, 180, img.width - 150, img.height - 150) )
    cropped.save('diagrams/constant_resampling.png')

plt.show()