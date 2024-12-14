# import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import os


n_scale = [0, 2, 4, 4.5, 5, 5.5, 6.75, 8, 9.25, 10.5, 11.75]
n_ori = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
n_ana = [0, 2, 4, 6, 8, 10]
n_syn = [0, 1, 2, 5.4, 7, 8.6]


output_folder = "diagrams"
os.makedirs(output_folder, exist_ok=True)


plt.figure(figsize=(12, 6))


plt.hlines(1, 0, max(n_ori) + 2, colors='black', linestyles='dashed')
plt.scatter(n_ori, [1] * len(n_ori), color='blue')
for i, n in enumerate(n_ori):
    plt.text(n, 1.1, f"{n}", color='blue', ha='center')

plt.hlines(0.5, 0, max(n_scale) + 1, colors='black', linestyles='dashed')
plt.scatter(n_scale, [0.5] * len(n_scale), color='green')
for i, n in enumerate(n_scale):
    plt.text(n, 0.6, f"{n}", color='green', ha='center', fontsize=9)


plt.hlines(0, 0, max(n_scale) + 1, colors='black', linestyles='dashed')
plt.scatter(n_scale, [0] * len(n_scale), color='green', label='$t_{scale}$')
plt.scatter(n_ana, [0] * len(n_ana), color='red', label='$n_{ana}$')
for i, n in enumerate(n_scale):
    plt.text(n, 0.1, f"{n}", color='green', ha='center', fontsize=9)
for i, n in enumerate(n_ana):
    plt.text(n, -0.1, f"{n}", color='red', ha='center')


plt.hlines(-0.5, 0, max(n_ori) + 1, colors='black', linestyles='dashed')
plt.scatter(n_ori, [-0.5] * len(n_ori), color='blue', label='$t_{ori}$')
plt.scatter(n_syn, [-0.5] * len(n_syn), color='purple', label='$n_{syn}$')
for i, n in enumerate(n_ori):
    plt.text(n, -0.4, f"{n}", color='blue', ha='center')
for i, n in enumerate(n_syn):
    plt.text(n, -0.6, f"{n}", color='purple', ha='center')


for n_a, n_s in zip(n_ana, n_syn):
    plt.annotate('', xy=(n_s, -0.5), xytext=(n_a, 0),
                 arrowprops=dict(arrowstyle='->',
                                 linestyle='--',
                                 color='gray',
                                 lw=1,
                                 mutation_scale=10))

plt.legend(loc='lower right', fontsize=10)
# plt.title("Nonlinear Time-Stretching", pad=20)
# plt.xlabel("Time")
plt.yticks([])
plt.xticks([])


plt.gca().spines["left"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.gca().spines["top"].set_visible(False)

plt.gca().spines["bottom"].set_visible(False)


plt.grid(True, axis='x', linestyle='--', alpha=0.6)
output_path = os.path.join(output_folder, "time_stretching_diagram_no_borders.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

