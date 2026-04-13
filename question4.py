from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

def load_image(path: Path) -> np.ndarray:
    image = Image.open(path)
    return np.asarray(image, dtype=np.float32) / 255.0

def save_image(image: np.ndarray, path: Path) -> None:
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image_uint8).save(path)

def mean_filter(image: np.ndarray, size: int) -> np.ndarray:
    if image.ndim == 2:
        kernel = np.ones((size, size)) / (size * size)
        return ndimage.convolve(image, kernel, mode='nearest')
    else:  # multichannel
        filtered = np.zeros_like(image)
        for c in range(image.shape[2]):
            kernel = np.ones((size, size)) / (size * size)
            filtered[:, :, c] = ndimage.convolve(image[:, :, c], kernel, mode='nearest')
        return filtered

def median_filter(image: np.ndarray, size: int) -> np.ndarray:
    if image.ndim == 2:
        return ndimage.median_filter(image, size=size, mode='nearest')
    else:
        filtered = np.zeros_like(image)
        for c in range(image.shape[2]):
            filtered[:, :, c] = ndimage.median_filter(image[:, :, c], size=size, mode='nearest')
        return filtered

def gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    if image.ndim == 2:
        return ndimage.gaussian_filter(image, sigma=sigma, mode='nearest')
    else:
        filtered = np.zeros_like(image)
        for c in range(image.shape[2]):
            filtered[:, :, c] = ndimage.gaussian_filter(image[:, :, c], sigma=sigma, mode='nearest')
        return filtered

def main() -> None:
    root = Path(__file__).resolve().parent
    input_path = root / "a1images" / "emma.jpg"
    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find runway image at {input_path}")

    image = load_image(input_path)

    # Apply mean filter with 3x3 kernel
    mean_filtered = mean_filter(image, 3)
    save_image(mean_filtered, root / "emma_mean_filter_3x3.png")

    # Apply median filter with 3x3 kernel
    median_filtered = median_filter(image, 3)
    save_image(median_filtered, root / "emma_median_filter_3x3.png")

    # Apply Gaussian filter with sigma=1.0
    gaussian_filtered = gaussian_filter(image, 1.0)
    save_image(gaussian_filtered, root / "emma_gaussian_filter_sigma1.png")

    print("Saved:")
    print(" - emma_mean_filter_3x3.png")
    print(" - emma_median_filter_3x3.png")
    print(" - emma_gaussian_filter_sigma1.png")

if __name__ == "__main__":
    main()