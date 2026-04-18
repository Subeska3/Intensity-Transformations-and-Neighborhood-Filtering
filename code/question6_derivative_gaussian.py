import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

def load_image_gray(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32)

def get_gaussian_kernel(size, sigma):
    """
    Computes a normalized 2D Gaussian kernel.
    """
    k = (size - 1) / 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel, x, y

def plot_3d_dog_kernel(save_path=None):
    h_size = 25
    x = np.arange(-h_size, h_size + 1, 1)
    y = np.arange(-h_size, h_size + 1, 1)
    Y, X = np.meshgrid(x, y)
    sigma = 10 
    g_51 = 1 / (2 * np.pi * sigma**2) * np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    g_51 = g_51 / np.sum(g_51)
    gx_51 = -(X / sigma**2) * g_51
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, gx_51, cmap='jet', rstride=2, cstride=2, linewidth=0.1, antialiased=True)
    ax.view_init(elev=25, azim=-45)
    plt.suptitle('51×51 Derivative-of-Gaussian Kernel (x-direction)', fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)

def get_derivative_gaussian_kernels(size, sigma):
    h_size = size // 2
    x = np.arange(-h_size, h_size + 1, 1)
    y = np.arange(-h_size, h_size + 1, 1)
    X, Y = np.meshgrid(x, y)
    g = 1 / (2 * np.pi * sigma**2) * np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    g = g / np.sum(g)
    gx = -(X / sigma**2) * g
    gy = -(Y / sigma**2) * g
    gx = gx / np.sum(np.abs(gx))
    gy = gy / np.sum(np.abs(gy))
    return gx, gy

def main():
    # Load the grayscale image
    root = Path(__file__).resolve().parent
    
    # Draw and save the 3D plot of the 51x51 kernel as requested
    plot_3d_dog_kernel(root / "dog_kernel_3d.png")
    
    image_path = root.parent / "a1images" / "brain_proton_density_slice.png"
    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Failed to load image: {image_path}")
        return

    # Compute derivative kernels for sigma=2, 5x5
    kernel_x, kernel_y = get_derivative_gaussian_kernels(5, 2)

    print("Derivative of Gaussian Kernel in X direction:")
    print(kernel_x)
    print("\nDerivative of Gaussian Kernel in Y direction:")
    print(kernel_y)

    # Apply kernels to get gradients
    grad_x_dog = cv2.filter2D(img_gray.astype(np.float32), -1, kernel_x)
    grad_y_dog = cv2.filter2D(img_gray.astype(np.float32), -1, kernel_y)

    # Compute magnitude
    grad_mag_dog = np.sqrt(grad_x_dog**2 + grad_y_dog**2)

    # Using OpenCV Sobel
    grad_x_sobel = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=5)
    grad_y_sobel = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=5)
    grad_mag_sobel = np.sqrt(grad_x_sobel**2 + grad_y_sobel**2)

    # Visualize and compare
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 4, 1)
    plt.title("Original Image")
    plt.imshow(img_gray, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.title("DoG Grad X")
    plt.imshow(grad_x_dog, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.title("DoG Grad Y")
    plt.imshow(grad_y_dog, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 4, 4)
    plt.title("DoG Grad Magnitude")
    plt.imshow(grad_mag_dog, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 4, 5)
    plt.title("Sobel Grad X")
    plt.imshow(grad_x_sobel, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 4, 6)
    plt.title("Sobel Grad Y")
    plt.imshow(grad_y_sobel, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 4, 7)
    plt.title("Sobel Grad Magnitude")
    plt.imshow(grad_mag_sobel, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 4, 8)
    plt.title("Difference in Magnitude")
    diff = cv2.absdiff(grad_mag_dog.astype(np.uint8), grad_mag_sobel.astype(np.uint8))
    plt.imshow(diff, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(root / "derivative_gaussian_comparison.png")
    # plt.show()  # Commented out for headless execution

    # Quantitative comparison
    mean_diff_x = np.mean(cv2.absdiff(grad_x_dog.astype(np.uint8), grad_x_sobel.astype(np.uint8)))
    mean_diff_y = np.mean(cv2.absdiff(grad_y_dog.astype(np.uint8), grad_y_sobel.astype(np.uint8)))
    mean_diff_mag = np.mean(cv2.absdiff(grad_mag_dog.astype(np.uint8), grad_mag_sobel.astype(np.uint8)))

    print(f"Mean absolute difference in Grad X: {mean_diff_x}")
    print(f"Mean absolute difference in Grad Y: {mean_diff_y}")
    print(f"Mean absolute difference in Grad Magnitude: {mean_diff_mag}")

    # Comment on differences
    print("\nComments on differences:")
    print("The Derivative of Gaussian (DoG) provides smoother gradients due to the Gaussian smoothing, reducing noise.")
    print("Sobel operator is an approximation using finite differences and is more sensitive to noise.")
    print("DoG gradients are typically smaller in magnitude and more blurred compared to Sobel.")
    print("Sobel is faster to compute and commonly used for edge detection.")

if __name__ == "__main__":
    main()