import matplotlib.pyplot as plt
import os

from PIL import Image


fig, ax = plt.subplots(figsize=(8, 5))


rect1 = plt.Rectangle((0, 1), 4, 0.5, linewidth=1.5, edgecolor='black', facecolor='none')
ax.add_patch(rect1)
ax.text(2, 1.25, r"$x_1(n)$", ha='center', va='center', fontsize=12)


rect2 = plt.Rectangle((3, 0.5), 4, 0.5, linewidth=1.5, edgecolor='black', facecolor='none')
ax.add_patch(rect2)
ax.text(5, 0.75, r"$x_2(n)$", ha='center', va='center', fontsize=12)


rect3 = plt.Rectangle((6, 0), 4, 0.5, linewidth=1.5, edgecolor='black', facecolor='none')
ax.add_patch(rect3)
ax.text(8, 0.25, r"$x_3(n)$", ha='center', va='center', fontsize=12)


ax.annotate('', xy=(2.5, 0.5), xytext=(0, 0.5),
            arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='<->', lw=1.5))


ax.annotate('', xy=(5.5, 0), xytext=(3, 0),
            arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='<->', lw=1.5))


ax.annotate('', xy=(3, 0.5), xytext=(2.25, 0.5),
            arrowprops=dict(facecolor='green', edgecolor='green', arrowstyle='<->', lw=1.5))


ax.annotate('', xy=(6, 0), xytext=(5.25, 0),
            arrowprops=dict(facecolor='green', edgecolor='green', arrowstyle='<->', lw=1.5))

ax.text(1.5, 0.7, r"$S_s = \alpha S_a$", ha='center', va='center', fontsize=12)
ax.text(4.5, 0.2, r"$S_s = \alpha S_a$", ha='center', va='center', fontsize=12)
ax.text(5, 1.5, r"$fade-out$", ha='center', va='center', fontsize=12)
ax.text(2.8, 0.25, r"fade-in", ha='center', va='center', fontsize=12)
ax.text(2.7, 0.7, r"$K_{m1}$", ha='center', va='center', fontsize=10, color='green')
ax.text(5.7, 0.2, r"$K_{m2}$", ha='center', va='center', fontsize=10, color='green')

rect1 = plt.Polygon([(3, 0.5), (4, 0.5), (4, 1.5), (3, 1.5)], color='gray', alpha=0.5, edgecolor='none')
ax.add_patch(rect1)

rect2 = plt.Polygon([(6, 0), (7, 0), (7, 1), (6, 1)], color='gray', alpha=0.5, edgecolor='none')
ax.add_patch(rect2)


ax.set_xlim(-1, 11)
ax.set_ylim(-1, 2)


ax.set_aspect('equal', adjustable='box')

ax.axis('off')


save_path = os.path.join("../diagrams", "time_stretching2.png")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, bbox_inches='tight')
print(f"save to {save_path}")

with Image.open(save_path) as img:
    cropped = img.crop((55, 20, img.width-60, img.height-50) )
    cropped.save(save_path)


plt.show()
