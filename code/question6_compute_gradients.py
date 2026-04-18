import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

sigma = 2.0
size = 5
half = size // 2
x = np.arange(-half, half + 1, dtype=np.float64)
y = np.arange(-half, half + 1, dtype=np.float64)
xx, yy = np.meshgrid(x, y, indexing='xy')
G = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
Gx = -xx / (sigma**2) * G
Gy = -yy / (sigma**2) * G

from pathlib import Path
root = Path(__file__).resolve().parent

img = Image.open(root.parent / 'saved_results' / 'runway.png').convert('L')
arr = np.asarray(img, dtype=np.float32)

grad_x = cv2.filter2D(arr, -1, Gx)
grad_y = cv2.filter2D(arr, -1, Gy)

minx, maxx = grad_x.min(), grad_x.max()
miny, maxy = grad_y.min(), grad_y.max()

print('grad_x range', minx, maxx)
print('grad_y range', miny, maxy)

norm_x = 255 * (grad_x - minx) / (maxx - minx)
norm_y = 255 * (grad_y - miny) / (maxy - miny)
    
Image.fromarray(np.clip(norm_x, 0, 255).astype(np.uint8)).save(root.parent / 'saved_results' / 'runway_dog_grad_x.png')
Image.fromarray(np.clip(norm_y, 0, 255).astype(np.uint8)).save(root.parent / 'saved_results' / 'runway_dog_grad_y.png')
print('Saved runway_dog_grad_x.png and runway_dog_grad_y.png')

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
# Using rstride=1, cstride=1 for smaller kernels so we don't skip points
ax.plot_surface(xx, yy, Gx, cmap='jet', rstride=1, cstride=1, linewidth=0.1, antialiased=True)
ax.view_init(elev=25, azim=-45)
plt.suptitle(f'{size}x{size} Derivative-of-Gaussian Kernel (x-direction)', fontsize=13)
plt.tight_layout()
plt.savefig(root.parent / 'saved_results' / 'dog_kernel_3d.png', bbox_inches='tight')
print('Saved dog_kernel_3d.png')
