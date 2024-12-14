
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager  # 导入 font_manager


def plot_csv_data(file_path, start_row=0, end_row=None, output_folder='Echo/diagrams'):

    data = pd.read_csv(file_path)


    if end_row is None:
        end_row = len(data)


    data_subset = data.iloc[start_row:end_row]

    time = data_subset['Time']
    ratio = data_subset['Ratio']
    # ratio = data_subset['Distance']

    plt.figure(figsize=(10, 6))
    plt.plot(time, ratio, marker='o', linestyle='-', color='b', label='Ratio vs Time')

    plt.title('Reflection Ratio-Time', fontsize=16, fontweight='bold')
    plt.xlabel('Time [s]', fontsize=14, fontweight='bold')
    plt.ylabel('Ratio', fontsize=14, fontweight='bold')
    # plt.ylabel('Ratio', fontsize=14, fontweight='bold')

    plt.tick_params(axis='both', labelsize=12, width=2)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontweight('bold')

    plt.grid(True)

    font_properties = font_manager.FontProperties(weight='bold')  #
    plt.legend(fontsize=12, prop=font_properties)


    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(output_folder, 'ratio_vs_time_plot.png')


    plt.savefig(output_file, dpi=300)


    plt.close()

    print(f"Plot saved to {output_file}")

plot_csv_data(
    "/vector_field_process6/Echo/Trajectories/reflection_ratios.csv",
    start_row=76, end_row=148,
    output_folder='Echo/diagrams')
