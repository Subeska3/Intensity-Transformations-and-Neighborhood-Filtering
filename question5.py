import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_gaussian_kernel(size, sigma):
    """
    (a) Computes a normalized 2D Gaussian kernel using NumPy. 
    """
    k = (size - 1) / 2
    # Create coordinate grid centered at 0
    x, y = np.mgrid[-k:k+1, -k:k+1]
    
    # 2D Gaussian formula: G(x,y) = (1/(2*pi*sigma^2)) * exp(-(x^2 + y^2)/(2*sigma^2))
    # We ignore the constant factor and normalize at the end for precision.
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Normalize the kernel so the sum of all coefficients is 1 
    kernel /= kernel.sum()
    return kernel

# --- (a) Compute 5x5 kernel for sigma = 2 ---
kernel_5x5 = get_gaussian_kernel(5, 2)
print("5x5 Gaussian Kernel (sigma=2):\n", kernel_5x5)

# --- (b) Visualize 51x51 kernel as a 3D surface plot ---
def visualize_kernel_3d(size, sigma):
    kernel = get_gaussian_kernel(size, sigma)
    k = (size - 1) / 2
    x, y = np.mgrid[-k:k+1, -k:k+1]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    # Kernel coefficients represent the height (Z-axis) 
    surf = ax.plot_surface(x, y, kernel, cmap='viridis', edgecolor='none')
    
    ax.set_title(f'3D Surface Plot of {size}x{size} Gaussian Kernel ($\sigma={sigma}$)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Kernel Coefficient (Height)')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()

visualize_kernel_3d(51, 2)

# --- (c) & (d) Apply Smoothing ---
# Load the runway image in grayscale [cite: 4, 5]
# Replace 'runway.png' with your actual file path
img = cv2.imread('runway.png', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Could not load image. Please check the file path.")
else:
    # (c) Manual smoothing using the computed 5x5 kernel [cite: 28]
    # cv2.filter2D performs the convolution operation
    manual_blurred = cv2.filter2D(img, -1, kernel_5x5)

    # (d) Using OpenCV's built-in function [cite: 29]
    opencv_blurred = cv2.GaussianBlur(img, (5, 5), 2)

    # Display results for comparison [cite: 96]
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Grayscale")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Manual Gaussian Filter")
    plt.imshow(manual_blurred, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("OpenCV Gaussian Blur")
    plt.imshow(opencv_blurred, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Quantitative Comparison for the report [cite: 96, 97]
    difference = cv2.absdiff(manual_blurred, opencv_blurred)
    print(f"Mean absolute difference: {np.mean(difference)}")