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

def get_derivative_gaussian_kernels(size, sigma):
    """
    Computes the derivative of Gaussian kernels in x and y directions.
    """
    gaussian_kernel, x, y = get_gaussian_kernel(size, sigma)
    kernel_x = - (x / sigma**2) * gaussian_kernel
    kernel_y = - (y / sigma**2) * gaussian_kernel
    return kernel_x, kernel_y

def main():
    # Load the grayscale image
    root = Path(__file__).resolve().parent
    image_path = root / "woman_grayscale.png"
    if not image_path.exists():
        print("Grayscale image not found, converting woman.avif to grayscale.")
        img_color = cv2.imread(str(root / "woman.avif"))
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(image_path), img_gray)
    else:
        img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

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