from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

def load_image_gray(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0

def save_image(image: np.ndarray, path: Path) -> None:
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image_uint8, mode="L").save(path)

def mean_filter(image: np.ndarray, size: int) -> np.ndarray:
    # size is the kernel size, e.g., 3 for 3x3
    kernel = np.ones((size, size)) / (size * size)
    return ndimage.convolve(image, kernel, mode='nearest')

def median_filter(image: np.ndarray, size: int) -> np.ndarray:
    return ndimage.median_filter(image, size=size, mode='nearest')

def gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    return ndimage.gaussian_filter(image, sigma=sigma, mode='nearest')

def main() -> None:
    root = Path(__file__).resolve().parent
    input_path = root / "runway.png"
    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find runway image at {input_path}")

    image = load_image_gray(input_path)

    # Apply mean filter with 3x3 kernel
    mean_filtered = mean_filter(image, 3)
    save_image(mean_filtered, root / "runway_mean_filter_3x3.png")

    # Apply median filter with 3x3 kernel
    median_filtered = median_filter(image, 3)
    save_image(median_filtered, root / "runway_median_filter_3x3.png")

    # Apply Gaussian filter with sigma=1.0
    gaussian_filtered = gaussian_filter(image, 1.0)
    save_image(gaussian_filtered, root / "runway_gaussian_filter_sigma1.png")

    print("Saved:")
    print(" - runway_mean_filter_3x3.png")
    print(" - runway_median_filter_3x3.png")
    print(" - runway_gaussian_filter_sigma1.png")

if __name__ == "__main__":
    main()