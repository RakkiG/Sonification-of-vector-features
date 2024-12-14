import numpy as np
import matplotlib.pyplot as plt

signal_length = 500
t = np.linspace(0, 2 * np.pi, signal_length)
original_signal = np.sin(t)

L = 100  #
overlap = 50  #
synthesis_signal = np.zeros(signal_length)

delayed_signal = np.zeros(signal_length)
for i in range(0, signal_length - L, overlap):
    analysis_block = original_signal[i:i + L]  #
    delayed_signal[i + overlap:i + L + overlap] = analysis_block  #
    synthesis_signal[i:i + L] += analysis_block  #

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, original_signal, label='Original Signal', color='blue')
plt.title('Original Signal')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, delayed_signal, label='Delayed Analysis Block', color='orange')
plt.title('Delayed Analysis Block')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, synthesis_signal, label='Synthesis Signal', color='green')
plt.title('Synthesis Signal (with Overlap-Add)')
plt.legend()

plt.tight_layout()
plt.show()
