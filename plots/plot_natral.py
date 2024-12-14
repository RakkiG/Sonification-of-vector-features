
import os
import numpy as np
import matplotlib.pyplot as plt

L = 30  #
Ss=10  #

signal_length = 200
time = np.arange(signal_length)
signal = np.sin(2 * np.pi * 0.05 * time)

analysis_start = 50
analysis_end = analysis_start + L
nat_start = analysis_start + Ss
nat_end = nat_start + L

os.makedirs("../diagrams", exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(time, signal, label="Original Signal", color='gray', alpha=0.7)

ax.add_patch(plt.Rectangle((analysis_start, -1), L, 2, color='r', alpha=0.6, label="Previous Analysis Block"))

ax.add_patch(plt.Rectangle((nat_start, -1), L, 2, color='g', alpha=0.6, label="Natural Continuation block"))

ax.annotate(f"Ss",
            xy=(50,1.2),
            xytext=(30,1.7),
            arrowprops=dict(facecolor='black', arrowstyle="->"),
            fontsize=12, color='blue')

rec = plt.Rectangle((50, -1.2), 10, 2.4, linewidth=4, edgecolor='yellow', facecolor='none')
ax.add_patch(rec)

ax.set_title("Natural Continuation Block")
ax.set_xlabel("Time")
ax.set_ylabel("Signal Amplitude")
ax.legend(loc='upper right')

ax.set_xlim(0, signal_length)
ax.set_ylim(-2, 2)

plt.tight_layout()

plt.savefig("diagrams/natural_continuation_block.png")


plt.show()

