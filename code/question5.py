import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16, 'font.weight': 'bold', 'axes.titleweight': 'bold', 'axes.labelweight': 'bold'})
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

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

# --- (b) Visualize 51x51 kernel as a 3D surface plot ---
def visualize_kernel_3d(size, sigma, output_path):
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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    root = Path(__file__).resolve().parent
    output_dir = root.parent / "saved_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- (a) Compute 5x5 kernel for sigma = 2 ---
    kernel_5x5 = get_gaussian_kernel(5, 2)
    print("5x5 Gaussian Kernel (sigma=2):\n", kernel_5x5)

    # --- (b) Visualize 51x51 kernel as a 3D surface plot ---
    visualize_kernel_3d(51, 2, output_dir / "question5_3d_kernel.png")

    # --- (c) & (d) Apply Smoothing ---
    input_path = root / "runway.png"
    if not input_path.exists():
        input_path = root.parent / "saved_results" / "runway.png"
    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find runway image at {input_path}")
        
    img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Could not load image. Please check the file path.")
    else:
        # (c) Manual smoothing using the computed 5x5 kernel
        manual_blurred = cv2.filter2D(img, -1, kernel_5x5)
        cv2.imwrite(str(output_dir / "runway_manual_gaussian_5x5.png"), manual_blurred)

        # (d) Using OpenCV's built-in function
        opencv_blurred = cv2.GaussianBlur(img, (5, 5), 2)
        cv2.imwrite(str(output_dir / "runway_opencv_gaussian_5x5.png"), opencv_blurred)

        # Display results for comparison
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
        grid_path = output_dir / "question5_comparison_grid.png"
        plt.savefig(grid_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Quantitative Comparison for the report
        difference = cv2.absdiff(manual_blurred, opencv_blurred)
        print(f"Mean absolute difference: {np.mean(difference)}")
        
        print(f"\nSaved to {output_dir}:")
        print(" - question5_3d_kernel.png")
        print(" - runway_manual_gaussian_5x5.png")
        print(" - runway_opencv_gaussian_5x5.png")
        print(f" - {grid_path.name}")

if __name__ == "__main__":
    main()