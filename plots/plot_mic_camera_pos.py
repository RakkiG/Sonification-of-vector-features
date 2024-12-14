import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

# 麦克风和相机的位置
microphone_position = np.array([2, 3, 1])  # 示例麦克风位置 (x, y, z)
d = 2  # 偏移量
camera_position = np.array([microphone_position[0] - d,
                            microphone_position[1] - d,
                            microphone_position[2] + d])

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制麦克风和相机
# 麦克风是蓝色立方体
ax.bar3d(microphone_position[0] - 0.5, microphone_position[1] - 0.5, microphone_position[2] - 0.5,
          1, 1, 1, color='b', alpha=0.6)

# 相机是灰色立方体
ax.bar3d(camera_position[0] - 0.5, camera_position[1] - 0.5, camera_position[2] - 0.5,
          1, 1, 1, color='gray', alpha=0.6)

# 绘制虚线连接麦克风和相机
ax.plot([microphone_position[0], camera_position[0]],
        [microphone_position[1], camera_position[1]],
        [microphone_position[2], camera_position[2]], 'k--')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 设置坐标轴范围
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.set_zlim([0, 10])

# 标注麦克风和相机的位置
ax.text(microphone_position[0]+1, microphone_position[1], microphone_position[2], 'Microphone', color='b')
ax.text(camera_position[0], camera_position[1], camera_position[2]+1, 'Camera', color='gray')

# 确保保存目录存在
os.makedirs('diagram', exist_ok=True)

# 保存图形到文件夹
plt.savefig('diagrams/microphone_camera.png')
output_path = 'diagrams/microphone_camera.png'

with Image.open(output_path) as img:
    cropped = img.crop((155, 80, img.width - 100, img.height - 30))  # 自定义裁剪范围
    cropped.save(output_path)

# 显示图形
plt.show()
