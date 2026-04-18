import numpy as np
sigma = 2.0
size = 5
xy = np.arange(-size//2 + 1, size//2 + 1)
x, y = np.meshgrid(xy, xy)
G = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
Gx = -x / (sigma**2) * G
Gy = -y / (sigma**2) * G
# Normalize such that the sum of positive values is 1, or absolute sum is 1, or scale by constant?
# In assignment contexts, normalization of derivative filters often means the sum of positive elements is 1, 
# so that applying it to a uniform image yields 0, and the response to a unit step edge is 1.
Gx_norm = Gx / np.sum(np.abs(Gx))
Gy_norm = Gy / np.sum(np.abs(Gy))
# Let's print raw and some normalizations
print("Raw Gx:")
print(np.round(Gx, 5))
print("Normalized Gx (abs sum=1):")
print(np.round(Gx_norm, 5))
print("Normalized Gx (pos sum=1):")
print(np.round(Gx / np.sum(Gx[Gx>0]), 5))
