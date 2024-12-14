

import matplotlib.pyplot as plt
import os


n_scale = [0, 2, 4, 4.5, 5, 5.5, 6.75, 8, 9.25, 10.5, 11.75]
n_ori = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
n_ana = [0, 2, 4, 6, 8, 10]
n_syn = [0.0, 1.0, 2.0, 5.4, 7.0, 8.6]
n_freq_change = [0, 2, 5, 10]
n_t_marks = [0, 4, 5.5, 11.75]
n_t_scale = [0, 2, 4, 4.5, 5, 5.5, 6.75, 8, 9.25, 10.5, 11.75]


output_folder = "diagrams"
os.makedirs(output_folder, exist_ok=True)


output_path = os.path.join(output_folder, "nonlinear_resampling.png")


plt.figure(figsize=(12, 6))

# t_original
plt.hlines(1, 0, max(n_ori) + 2, colors='black', linestyles='dashed')
plt.scatter(n_ori, [1] * len(n_ori), color='blue', label='$t_{ori}$')
for i, n in enumerate(n_ori):
    plt.text(n, 1.1, f"{n}", color='blue', ha='center')

# t_freq_change
plt.hlines(0.5, 0, max(n_freq_change) + 1, colors='black', linestyles='dashed')
plt.scatter(n_freq_change, [0.5] * len(n_freq_change), color='green', label='$t_{freq\\_change}$')
for i, n in enumerate(n_freq_change):
    plt.text(n, 0.6, f"{n:.2f}", color='green', ha='center', fontsize=9)

# t_marks
plt.hlines(0, 0, max(n_t_marks) + 1, colors='black', linestyles='dashed')
plt.scatter(n_t_marks, [0] * len(n_t_marks), color='red', label='$t_{marks}$')
for i, n in enumerate(n_t_marks):
    plt.text(n, 0.1, f"{n}", color='red', ha='center')

# t_original and t_freq_change combined
plt.hlines(-0.5, 0, max(n_ori) + 2, colors='black', linestyles='dashed')
plt.scatter(n_freq_change, [-0.5] * len(n_freq_change), color='green', label='$t_{freq\\_change}$')
plt.scatter(n_ori, [-0.5] * len(n_ori), color='blue')
for i, n in enumerate(n_freq_change):
    plt.text(n, -0.4, f"{n:.2f}", color='green', ha='center', fontsize=9)
for i, n in enumerate(n_ori):
    plt.text(n, -0.6, f"{n}", color='blue', ha='center')

# t_scale
plt.hlines(-1, 0, max(n_t_scale) + 1, colors='black', linestyles='dashed')
plt.scatter(n_t_scale, [-1] * len(n_t_scale), color='purple', label='$t_{scale}$')
plt.scatter(n_t_marks, [-1] * len(n_t_marks), color='red')
for i, n in enumerate(n_t_marks):
    plt.text(n, -1.1, f"{n}", color='purple', ha='center')
for i, n in enumerate(n_t_scale):
    plt.text(n, -0.9, f"{n:.1f}", color='red', ha='center')

# Draw arrows for resampling
for n_ori_i, n_scale_i in zip(n_ori, n_scale):
    plt.annotate('', xy=(n_scale_i, -1), xytext=(n_ori_i, -0.5),
                 arrowprops=dict(arrowstyle='->',
                                 linestyle='--',
                                 color='gray',
                                 lw=1,
                                 mutation_scale=10))

# Legend and title
plt.legend(loc='best', fontsize=12)
# plt.title("Nonlinear Resampling", pad=20)
plt.yticks([])  # Remove y-axis ticks
plt.xticks([])  # Remove x-axis ticks

# Hide spines
plt.gca().spines["left"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)

# Grid
plt.grid(True, axis='x', linestyle='--', alpha=0.6)

plt.draw()
plt.savefig(output_path, bbox_inches='tight')
plt.close()

