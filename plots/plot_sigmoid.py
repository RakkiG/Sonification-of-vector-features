import os
import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid and flipped sigmoid functions
def sigmoid(x, x_shift=0, scale=1):
    return 1 / (1 + np.exp(-scale * (x - x_shift)))

def flipped_sigmoid(x, x_shift=0, scale=1):
    return 1 - sigmoid(x, x_shift, scale)

# Generate a range of x values for plotting
x = np.linspace(-10, 10, 500)

# Parameters for the sigmoid functions
x_shift = 0
scale = 1

y_sigmoid = sigmoid(x, x_shift=x_shift, scale=scale)
y_flipped_sigmoid = flipped_sigmoid(x, x_shift=x_shift, scale=scale)

output_folder = "diagrams"
os.makedirs(output_folder, exist_ok=True)


plt.figure(figsize=(10, 6))
plt.plot(x, y_sigmoid, label="Sigmoid Function", linewidth=2, color="blue")
plt.plot(x, y_flipped_sigmoid, label="Flipped Sigmoid Function", linewidth=2, linestyle='--', color="red")

plt.title("Sigmoid and Flipped Sigmoid Functions", fontsize=14, fontweight='bold')
plt.xlabel("x", fontsize=12, fontweight='bold')
plt.ylabel("f(x)", fontsize=12, fontweight='bold')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
plt.axhline(1, color='gray', linestyle='--', linewidth=0.7)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)


output_path = os.path.join(output_folder, "sigmoid_plot.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Plot saved to {output_path}")
