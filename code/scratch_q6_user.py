import numpy as np

h_size = 2
x = np.arange(-h_size, h_size + 1, 1)
y = np.arange(-h_size, h_size + 1, 1)

# Correctly assigning X for horizontal and Y for vertical
X, Y = np.meshgrid(x, y)
sigma = 2
g = 1 / (2 * np.pi * sigma**2) * np.exp(-(X**2 + Y**2) / (2 * sigma**2))
g = g / np.sum(g)

# X computes horizontal differences
gx = -(X / sigma**2) * g

# Y computes vertical differences
gy = -(Y / sigma**2) * g

gx = gx / np.sum(np.abs(gx))
gy = gy / np.sum(np.abs(gy))

print('Derivatives of a Gaussian for σ = 2 in the x-direction:')
print(np.round(gx, 5))
print('\nDerivatives of a Gaussian for σ = 2 in the y-direction:')
print(np.round(gy, 5))
