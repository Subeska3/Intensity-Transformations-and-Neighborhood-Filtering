import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Load the grayscale runway image
img = Image.open('runway.png').convert('L')
arr = np.asarray(img, dtype=np.float32)

# Compute DoG gradients (from previous work)
sigma = 2.0
size = 5
half = size // 2
x = np.arange(-half, half+1, dtype=np.float64)
y = np.arange(-half, half+1, dtype=np.float64)
xx, yy = np.meshgrid(x, y, indexing='xy')
G = np.exp(-(xx**2 + yy**2)/(2*sigma**2)) / (2*np.pi*sigma**2)
Gx = -xx / (sigma**2) * G
Gy = -yy / (sigma**2) * G

grad_x_dog = cv2.filter2D(arr, -1, Gx)
grad_y_dog = cv2.filter2D(arr, -1, Gy)
grad_mag_dog = np.sqrt(grad_x_dog**2 + grad_y_dog**2)

# Compute Sobel gradients
grad_x_sobel = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=5)
grad_y_sobel = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=5)
grad_mag_sobel = np.sqrt(grad_x_sobel**2 + grad_y_sobel**2)

# Normalize for visualization
def normalize(img):
    min_val, max_val = img.min(), img.max()
    return ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

grad_x_dog_norm = normalize(grad_x_dog)
grad_y_dog_norm = normalize(grad_y_dog)
grad_mag_dog_norm = normalize(grad_mag_dog)
grad_x_sobel_norm = normalize(grad_x_sobel)
grad_y_sobel_norm = normalize(grad_y_sobel)
grad_mag_sobel_norm = normalize(grad_mag_sobel)

# Create comparison plot
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0,0].imshow(arr, cmap='gray')
axes[0,0].set_title('Original Grayscale')
axes[0,0].axis('off')

axes[0,1].imshow(grad_x_dog_norm, cmap='gray')
axes[0,1].set_title('DoG Grad X')
axes[0,1].axis('off')

axes[0,2].imshow(grad_y_dog_norm, cmap='gray')
axes[0,2].set_title('DoG Grad Y')
axes[0,2].axis('off')

axes[0,3].imshow(grad_mag_dog_norm, cmap='gray')
axes[0,3].set_title('DoG Magnitude')
axes[0,3].axis('off')

axes[1,0].imshow(arr, cmap='gray')
axes[1,0].set_title('Original Grayscale')
axes[1,0].axis('off')

axes[1,1].imshow(grad_x_sobel_norm, cmap='gray')
axes[1,1].set_title('Sobel Grad X')
axes[1,1].axis('off')

axes[1,2].imshow(grad_y_sobel_norm, cmap='gray')
axes[1,2].set_title('Sobel Grad Y')
axes[1,2].axis('off')

axes[1,3].imshow(grad_mag_sobel_norm, cmap='gray')
axes[1,3].set_title('Sobel Magnitude')
axes[1,3].axis('off')

plt.tight_layout()
plt.savefig('question6_sobel_comparison.png', dpi=150, bbox_inches='tight')
print('Saved question6_sobel_comparison.png')

# Quantitative comparison
mean_diff_x = np.mean(np.abs(grad_x_dog_norm.astype(np.float32) - grad_x_sobel_norm.astype(np.float32)))
mean_diff_y = np.mean(np.abs(grad_y_dog_norm.astype(np.float32) - grad_y_sobel_norm.astype(np.float32)))
mean_diff_mag = np.mean(np.abs(grad_mag_dog_norm.astype(np.float32) - grad_mag_sobel_norm.astype(np.float32)))

print(f"Mean absolute difference in Grad X: {mean_diff_x:.2f}")
print(f"Mean absolute difference in Grad Y: {mean_diff_y:.2f}")
print(f"Mean absolute difference in Grad Magnitude: {mean_diff_mag:.2f}")

print("\nComments on differences:")
print("- DoG provides smoother gradients due to Gaussian smoothing, reducing noise sensitivity.")
print("- Sobel uses finite differences and is more sensitive to noise.")
print("- DoG gradients are typically smaller in magnitude and more blurred.")
print("- Sobel is faster to compute and commonly used for edge detection.")