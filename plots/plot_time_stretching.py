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

rect4 = plt.Rectangle((0, 3.5), 4, 0.5, linewidth=1.5, edgecolor='black', facecolor='none')
ax.add_patch(rect4)
ax.text(2, 3.75, r"$x_1(n)$", ha='center', va='center', fontsize=12)

rect5 = plt.Rectangle((2, 3), 4, 0.5, linewidth=1.5, edgecolor='black', facecolor='none')
ax.add_patch(rect5)
ax.text(4, 3.25, r"$x_2(n)$", ha='center', va='center', fontsize=12)

rect6 = plt.Rectangle((4, 2.5), 4, 0.5, linewidth=1.5, edgecolor='black', facecolor='none')
ax.add_patch(rect6)
ax.text(6, 2.75, r"$x_3(n)$", ha='center', va='center', fontsize=12)

rect7 = plt.Rectangle((0, 4.2), 8, 0.5, linewidth=1.5, edgecolor='black', facecolor='none')
ax.add_patch(rect7)
ax.text(4, 4.45, r"$x(n)$", ha='center', va='center', fontsize=12)


ax.annotate('', xy=(2, 3), xytext=(0, 3),
            arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='<->', lw=1.5))

ax.annotate('', xy=(4, 2.5), xytext=(2, 2.5),
            arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='<->', lw=1.5))

ax.annotate('', xy=(3, 0.5), xytext=(0, 0.5),
            arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='<->', lw=1.5))


ax.annotate('', xy=(6, 0), xytext=(3, 0),
            arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='<->', lw=1.5))

ax.text(1, 3.25, r"$S_a$", ha='center', va='center', fontsize=12)
ax.text(3.25, 2.75, r"$S_a$", ha='center', va='center', fontsize=12)
ax.text(1.5, 0.7, r"$S_s = \alpha S_a$", ha='center', va='center', fontsize=12)
ax.text(4.5, 0.2, r"$S_s = \alpha S_a$", ha='center', va='center', fontsize=12)
ax.text(5, 1.5, r"$fade-out$", ha='center', va='center', fontsize=12)
ax.text(2.8, 0.25, r"fade-in", ha='center', va='center', fontsize=12)

rect1 = plt.Polygon([(3, 0.5), (4, 0.5), (4, 1.5), (3, 1.5)], color='gray', alpha=0.5, edgecolor='none')
ax.add_patch(rect1)

rect2 = plt.Polygon([(6, 0), (7, 0), (7, 1), (6, 1)], color='gray', alpha=0.5, edgecolor='none')
ax.add_patch(rect2)


ax.set_xlim(-1, 11)
ax.set_ylim(-1, 8)


ax.set_aspect('equal', adjustable='box')

ax.axis('off')


save_path = os.path.join("../diagrams", "rectangles_and_arrows.png")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, bbox_inches='tight',pad_inches=0)

with Image.open(save_path) as img:
    cropped = img.crop((36, 132, img.width-40, img.height-40) )
    cropped.save(save_path)

plt.show()
