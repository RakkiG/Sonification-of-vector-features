

import numpy as np
import matplotlib.pyplot as plt


L = 30
Ss = 10
signal_length = 150
time = np.arange(signal_length)
signal = np.sin(2 * np.pi * 0.05 * time)


analysis_start = 50
analysis_end = analysis_start + L
nat_start = analysis_end + Ss
nat_end = nat_start + L


def cross_corr(x, y):
    return np.correlate(x, y, mode='valid')

X_ana = signal[analysis_start:analysis_end]
X_nat = signal[nat_start:nat_end]

cc = cross_corr(X_ana, X_nat)

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(time, signal, label="Original Signal", color='gray', alpha=0.7)

ax.add_patch(plt.Rectangle((analysis_start, -1), L, 2, color='r', alpha=0.6, label="Analysis Block (X_ana)"))

ax.add_patch(plt.Rectangle((nat_start, -1), L, 2, color='g', alpha=0.6, label="Natural Continuation Block (X_nat)"))

ax.annotate(f"Ss = {Ss}",
            xy=(analysis_end + Ss / 2, 0),
            xytext=(analysis_end + Ss / 2 + 5, 0.5),
            arrowprops=dict(facecolor='black', arrowstyle="->"),
            fontsize=12, color='blue')

shifted_time = np.arange(analysis_start, nat_end)
ax.plot(shifted_time, np.concatenate(([0] * (nat_start - analysis_start), cc, [0] * (signal_length - len(cc) - nat_end))),
        label="Cross-Correlation", color='purple', linewidth=2)

ax.set_title("Cross-Correlation for Signal Splicing")
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")
ax.legend(loc='upper right')

ax.set_xlim(0, signal_length)
ax.set_ylim(-2, 2)

plt.tight_layout()
plt.show()
