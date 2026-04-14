import numpy as np
from PIL import Image
import cv2

sigma = 2.0
size = 5
half = size // 2
x = np.arange(-half, half + 1, dtype=np.float64)
y = np.arange(-half, half + 1, dtype=np.float64)
xx, yy = np.meshgrid(x, y, indexing='xy')
G = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
Gx = -xx / (sigma**2) * G
Gy = -yy / (sigma**2) * G

img = Image.open('runway.png').convert('L')
arr = np.asarray(img, dtype=np.float32)

grad_x = cv2.filter2D(arr, -1, Gx)
grad_y = cv2.filter2D(arr, -1, Gy)

minx, maxx = grad_x.min(), grad_x.max()
miny, maxy = grad_y.min(), grad_y.max()

print('grad_x range', minx, maxx)
print('grad_y range', miny, maxy)

norm_x = 255 * (grad_x - minx) / (maxx - minx)
norm_y = 255 * (grad_y - miny) / (maxy - miny)
Image.fromarray(np.clip(norm_x, 0, 255).astype(np.uint8)).save('runway_dog_grad_x.png')
Image.fromarray(np.clip(norm_y, 0, 255).astype(np.uint8)).save('runway_dog_grad_y.png')
print('Saved runway_dog_grad_x.png and runway_dog_grad_y.png')
