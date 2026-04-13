from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image_gray(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0

def save_image(image: np.ndarray, path: Path) -> None:
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image_uint8, mode="L").save(path)

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    # image is 0-1
    # Convert to 0-255 for processing
    img_255 = (image * 255).astype(np.uint8)
    
    # Compute histogram
    hist, bins = np.histogram(img_255.flatten(), bins=256, range=[0, 256])
    
    # Compute CDF
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]  # normalize to 0-1
    
    # Equalize
    equalized = np.interp(img_255.flatten(), bins[:-1], cdf_normalized * 255)
    equalized = equalized.reshape(img_255.shape).astype(np.uint8)
    
    # Back to 0-1
    return equalized.astype(np.float32) / 255.0

def plot_histograms(original, equalized):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(original.ravel(), bins=256, range=(0, 1), color='gray', alpha=0.7)
    plt.title('Original Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(equalized.ravel(), bins=256, range=(0, 1), color='gray', alpha=0.7)
    plt.title('Equalized Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('histograms.png')  # Save instead of show
    print("Histograms saved as histograms.png")

def main() -> None:
    root = Path(__file__).resolve().parent
    input_path = root / "runway.png"
    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find runway image at {input_path}")

    image = load_image_gray(input_path)
    
    equalized = histogram_equalization(image)
    
    save_image(equalized, root / "runway_equalized.png")
    
    plot_histograms(image, equalized)
    
    print("Saved: runway_equalized.png")

if __name__ == "__main__":
    main()