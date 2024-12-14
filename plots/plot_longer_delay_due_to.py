import matplotlib.pyplot as plt


n_ana = [0, 2, 4, 6, 8, 10]  #
delay = [0, 0.5, 1, 1.5, 2, 2.5]  #
n_syn = [n + d for n, d in zip(n_ana, delay)]  #
block_duration = 1.5  #
overlap_duration = 0.5  #


plt.figure(figsize=(12, 6))

for n in n_ana:
    plt.hlines(1, n, n + block_duration, colors='blue', linestyles='solid', label='Analysis Block' if n == 0 else "")
    plt.text(n + block_duration / 2, 1.1, f"A{n}", color='blue', ha='center', fontsize=9)

for n in n_syn:
    plt.hlines(0.5, n, n + block_duration, colors='green', linestyles='solid', label='Synthesis Block' if n == n_syn[0] else "")
    plt.text(n + block_duration / 2, 0.4, f"S{n:.1f}", color='green', ha='center', fontsize=9)

for n_a, n_s in zip(n_ana, n_syn):
    plt.arrow(n_a + block_duration / 2, 1, n_s - n_a, -0.4, head_width=0.1, head_length=0.2, color='gray', alpha=0.7)
    plt.text((n_a + n_s) / 2, 0.7, f"Delay={n_s-n_a:.1f}", fontsize=8, color='gray', ha='center')

for n in n_syn:
    plt.fill_betweenx([0.5], n + block_duration - overlap_duration, n + block_duration, color='purple', alpha=0.3, label='Overlap Region' if n == n_syn[0] else "")

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=10)
plt.title("Adaptive Pitch-Shifting: Analysis to Synthesis Blocks", fontsize=12, pad=20)
plt.xlabel("Time")
plt.yticks([1, 0.5], ["Analysis Blocks", "Synthesis Blocks"])
plt.ylim(0.2, 1.3)
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.gca().spines["left"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)


plt.tight_layout()
plt.show()
