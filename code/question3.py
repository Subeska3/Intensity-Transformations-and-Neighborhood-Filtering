from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16, 'font.weight': 'bold', 'axes.titleweight': 'bold', 'axes.labelweight': 'bold'})

def load_image_gray(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0

def save_image(image: np.ndarray, path: Path) -> None:
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image_uint8, mode="L").save(path)

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    img_255 = (image * 255).astype(np.uint8)
    hist, bins = np.histogram(img_255.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]  # normalize to 0-1
    equalized = np.interp(img_255.flatten(), bins[:-1], cdf_normalized * 255)
    equalized = equalized.reshape(img_255.shape).astype(np.uint8)
    return equalized.astype(np.float32) / 255.0

def plot_histograms(original, equalized, output_path):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(original.ravel(), bins=256, range=(0, 1), color='blue', alpha=0.7)
    plt.title('Original Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(equalized.ravel(), bins=256, range=(0, 1), color='orange', alpha=0.7)
    plt.title('Equalized Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Histograms saved as {output_path.name}")

def main() -> None:
    root = Path(__file__).resolve().parent
    input_path = root / "runway.png"
    if not input_path.exists():
        input_path = root.parent / "saved_results" / "runway.png"
    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find runway image at {input_path}")

    output_dir = root.parent / "saved_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_image_gray(input_path)
    
    equalized = histogram_equalization(image)
    
    save_image(equalized, output_dir / "runway_equalized.png")
    
    hist_path = output_dir / "question3_histograms.png"
    plot_histograms(image, equalized, hist_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original Gray')
    axes[0].axis('off')
    
    axes[1].imshow(equalized, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Histogram Equalized')
    axes[1].axis('off')
    
    plt.tight_layout()
    grid_path = output_dir / "question3_comparison_grid.png"
    plt.savefig(grid_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved to {output_dir}:")
    print(" - runway_equalized.png")
    print(f" - {hist_path.name}")
    print(f" - {grid_path.name}")

if __name__ == "__main__":
    main()