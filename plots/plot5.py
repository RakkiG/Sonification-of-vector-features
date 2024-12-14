
import matplotlib.pyplot as plt
import pandas as pd
import os

# Reading a CSV file
file_path = '/Users/gongyaqi/Desktop/MasterThesis/Generate_Sound_For_3D6/vector_field_process5/Echo/Trajectories/initial_sourcesource0_trajectory.csv'  # 替换为你的 CSV 文件路径
data = pd.read_csv(file_path, header=None)  # 如果没有列名，使用 header=None

# Get time data (first column)
time = data.iloc[:, 0]

# Extract the first value of the triple in the second column a
positions = data.iloc[:, 1].apply(lambda x: float(x.strip('()').split(',')[0]))

output_folder = "diagrams"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#plot
plt.figure(figsize=(10, 5))
plt.plot(time, positions, marker='o', markersize=4, linestyle='-', label='First Element of Position over Time')  # 调整 markersize 的值
plt.xlabel('Time (s)')
plt.ylabel('X Coordinate (a)')
plt.title('Change of X Coordinate (a) Over Time')
plt.legend()

#save
output_path = os.path.join(output_folder, "source0_x_position_over_time.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Plot saved to {output_path}")






