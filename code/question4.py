from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

def load_image_gray(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.uint8)

def save_image(image: np.ndarray, path: Path) -> None:
    if image.ndim == 2:
        Image.fromarray(image.astype(np.uint8), mode="L").save(path)
    else:
        Image.fromarray(image.astype(np.uint8)).save(path)

def otsu_threshold(image: np.ndarray) -> int:
    hist = np.histogram(image.flatten(), bins=256, range=(0, 256))[0]
    total_pixels = image.size
    current_max, threshold = 0, 0
    sumT = np.sum(np.arange(256) * hist)
    sumB = 0
    wB = 0
    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total_pixels - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sumT - sumB) / wF
        between = wB * wF * (mB - mF) ** 2
        if between > current_max:
            current_max = between
            threshold = i
    return threshold

def foreground_equalization(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    foreground_pixels = image[mask]
    hist, bins = np.histogram(foreground_pixels, bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]
    equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized * 255)
    return equalized.astype(np.uint8).reshape(image.shape)

def mean_filter(image: np.ndarray, size: int) -> np.ndarray:
    kernel = np.ones((size, size)) / (size * size)
    return ndimage.convolve(image, kernel, mode='nearest')

def median_filter(image: np.ndarray, size: int) -> np.ndarray:
    return ndimage.median_filter(image, size=size, mode='nearest')

def gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    return ndimage.gaussian_filter(image, sigma=sigma, mode='nearest')

def main() -> None:
    root = Path(__file__).resolve().parent
    input_path = root / "Woman.jpg"
    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find image at {input_path}")

    # Load and convert to grayscale
    image_gray = load_image_gray(input_path)
    save_image(image_gray, root / "woman_grayscale.png")

    # Apply Otsu thresholding
    threshold = otsu_threshold(image_gray)
    mask = (image_gray >= threshold).astype(np.uint8) * 255
    save_image(mask, root / "woman_otsu_mask.png")

    # Foreground equalization
    equalized = foreground_equalization(image_gray, mask > 0)
    save_image(equalized, root / "woman_foreground_equalized.png")

    # Convert to float for filtering
    equalized_float = equalized.astype(np.float32) / 255.0

    # Apply mean filter with 3x3 kernel
    mean_filtered = mean_filter(equalized_float, 3)
    save_image((mean_filtered * 255).astype(np.uint8), root / "woman_mean_filter_3x3.png")

    # Apply median filter with 3x3 kernel
    median_filtered = median_filter(equalized_float, 3)
    save_image((median_filtered * 255).astype(np.uint8), root / "woman_median_filter_3x3.png")

    # Apply Gaussian filter with sigma=1.0
    gaussian_filtered = gaussian_filter(equalized_float, 1.0)
    save_image((gaussian_filtered * 255).astype(np.uint8), root / "woman_gaussian_filter_sigma1.png")

    print("Saved:")
    print(" - woman_grayscale.png")
    print(" - woman_otsu_mask.png")
    print(" - woman_foreground_equalized.png")
    print(" - woman_mean_filter_3x3.png")
    print(" - woman_median_filter_3x3.png")
    print(" - woman_gaussian_filter_sigma1.png")

if __name__ == "__main__":
    main()