from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy import ndimage

def load_image_gray(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.uint8)

def save_image(image: np.ndarray, path: Path) -> None:
    Image.fromarray(image.astype(np.uint8), mode="L").save(path)

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()  # normalize
    return kernel

def visualize_kernel_3d(kernel: np.ndarray, filename: str) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(kernel.shape[0])
    y = np.arange(kernel.shape[1])
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, kernel, cmap='viridis')
    plt.savefig(filename)
    plt.close()
    print(f"3D plot saved as {filename}")

def main() -> None:
    root = Path(__file__).resolve().parent
    input_path = root / "runway.png"
    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find image at {input_path}")

    # (a) Compute normalized 5x5 Gaussian kernel for σ = 2
    kernel_5x5 = gaussian_kernel(5, 2.0)
    print("5x5 Gaussian kernel for σ=2:")
    print(kernel_5x5)
    np.savetxt(root / "gaussian_kernel_5x5.txt", kernel_5x5)

    # (b) Visualize 51x51 Gaussian kernel as 3D surface plot
    kernel_51x51 = gaussian_kernel(51, 2.0)
    visualize_kernel_3d(kernel_51x51, root / "gaussian_kernel_51x51_3d.png")

    # Load grayscale image
    image_gray = load_image_gray(input_path)
    image_float = image_gray.astype(np.float32) / 255.0

    # (c) Apply Gaussian smoothing using manually computed kernel (11x11 for visible effect)
    smoothed_manual = ndimage.convolve(image_float, gaussian_kernel(11, 3.0), mode='nearest')
    smoothed_manual_uint8 = (smoothed_manual * 255).astype(np.uint8)
    save_image(smoothed_manual_uint8, root / "runway_gaussian_manual_11x11.png")

    # (d) Apply using OpenCV's cv.GaussianBlur() (11x11 with sigma=3.0)
    smoothed_opencv = cv2.GaussianBlur(image_gray, (11, 11), 3.0)
    save_image(smoothed_opencv, root / "runway_gaussian_opencv_11x11.png")

    print("Saved:")
    print(" - gaussian_kernel_5x5.txt")
    print(" - gaussian_kernel_51x51_3d.png")
    print(" - runway_gaussian_manual_11x11.png")
    print(" - runway_gaussian_opencv_11x11.png")

if __name__ == "__main__":
    main()