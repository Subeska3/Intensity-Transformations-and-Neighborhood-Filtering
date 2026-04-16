from pathlib import Path

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16, 'font.weight': 'bold', 'axes.titleweight': 'bold', 'axes.labelweight': 'bold'})
import numpy as np
from PIL import Image


def load_image_gray(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def save_image(image: np.ndarray, path: Path) -> None:
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image_uint8, mode="L").save(path)


def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("Gamma must be a positive value.")
    return np.power(image, gamma)


def contrast_stretch(image: np.ndarray, r1: float, r2: float) -> np.ndarray:
    if not (0 <= r1 < r2 <= 1):
        raise ValueError("Require 0 <= r1 < r2 <= 1.")
    output = np.zeros_like(image)
    mask_low = image < r1
    mask_mid = (image >= r1) & (image <= r2)
    mask_high = image > r2
    output[mask_low] = 0.0
    output[mask_mid] = (image[mask_mid] - r1) / (r2 - r1)
    output[mask_high] = 1.0
    return output


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
    save_image(image, output_dir / "runway_gray.png")

    gamma_05 = gamma_correction(image, 0.5)
    save_image(gamma_05, output_dir / "runway_gamma_0.5.png")

    gamma_2 = gamma_correction(image, 2.0)
    save_image(gamma_2, output_dir / "runway_gamma_2.0.png")

    stretch = contrast_stretch(image, r1=0.2, r2=0.8)
    save_image(stretch, output_dir / "runway_contrast_stretch_r1_0.2_r2_0.8.png")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Original Gray')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gamma_05, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Gamma = 0.5')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(gamma_2, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Gamma = 2.0')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(stretch, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Contrast Stretch (r1=0.2, r2=0.8)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    grid_path = output_dir / "question1_comparison_grid.png"
    plt.savefig(grid_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved to {output_dir}:")
    print(" - runway_gray.png")
    print(" - runway_gamma_0.5.png")
    print(" - runway_gamma_2.0.png")
    print(" - runway_contrast_stretch_r1_0.2_r2_0.8.png")
    print(f" - {grid_path.name}")


if __name__ == "__main__":
    main()
