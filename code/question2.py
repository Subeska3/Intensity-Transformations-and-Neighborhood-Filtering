from pathlib import Path
import numpy as np
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16, 'font.weight': 'bold', 'axes.titleweight': 'bold', 'axes.labelweight': 'bold'})

def load_image(path: Path) -> np.ndarray:
    image = Image.open(path)
    return np.asarray(image, dtype=np.float32) / 255.0

def save_image(image: np.ndarray, path: Path) -> None:
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image_uint8).save(path)

def gamma_correction_lab_L(image_lab: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("Gamma must be a positive value.")
    L, a, b = image_lab[:, :, 0], image_lab[:, :, 1], image_lab[:, :, 2]
    L_corrected = 100 * np.power(L / 100, gamma)
    return np.stack([L_corrected, a, b], axis=-1)

def main() -> None:
    root = Path(__file__).resolve().parent
    input_path = root.parent / "a1images" / "highlights_and_shadows.jpg"
    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find image at {input_path}")

    output_dir = root.parent / "saved_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    image_rgb = load_image(input_path)
    
    # Convert to LAB
    pil_image = Image.fromarray((image_rgb * 255).astype(np.uint8))
    lab_image = pil_image.convert('LAB')
    lab_array = np.asarray(lab_image, dtype=np.float32)
    
    # Gamma correction on L channel, gamma = 0.5 to brighten shadows
    gamma = 0.5
    lab_corrected = gamma_correction_lab_L(lab_array, gamma)
    
    # Convert back to RGB
    corrected_pil = Image.fromarray(lab_corrected.astype(np.uint8), mode='LAB').convert('RGB')
    corrected_rgb = np.asarray(corrected_pil, dtype=np.float32) / 255.0
    
    # Save corrected image
    save_image(corrected_rgb, output_dir / "highlights_and_shadows_gamma_corrected.jpg")
    
    # Convert to grayscale for histogram comparison
    img_uint8 = (image_rgb * 255.0).astype(np.uint8)
    img_corrected_uint8 = (corrected_rgb * 255.0).astype(np.uint8)
    
    img_gray = cv.cvtColor(img_uint8, cv.COLOR_RGB2GRAY)
    img_corrected_gray = cv.cvtColor(img_corrected_uint8, cv.COLOR_RGB2GRAY)
    
    hist_original = np.bincount(img_gray.ravel(), minlength=256)
    hist_corrected = np.bincount(img_corrected_gray.ravel(), minlength=256)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].bar(range(256), hist_original, color='blue', alpha=0.7)
    ax[0].set_title('Intensity vs Frequency — Original')
    ax[0].set_xlabel('Pixel Intensity')
    ax[0].set_ylabel('Frequency')
    ax[0].set_xlim([0, 255])
    
    ax[1].bar(range(256), hist_corrected, color='red', alpha=0.7)
    ax[1].set_title(f'Intensity vs Frequency (γ = {gamma})')
    ax[1].set_xlabel('Pixel Intensity')
    ax[1].set_ylabel('Frequency')
    ax[1].set_xlim([0, 255])
    plt.tight_layout()
    
    hist_path = output_dir / "question2_histograms.png"
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    

    # Plot comparison grid
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original RGB')
    axes[0].axis('off')
    
    axes[1].imshow(corrected_rgb)
    axes[1].set_title('Gamma Corrected RGB')
    axes[1].axis('off')
    
    plt.tight_layout()
    grid_path = output_dir / "question2_comparison_grid.png"
    plt.savefig(grid_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gamma value used: {gamma}")
    print(f"Saved to {output_dir}:")
    print(" - highlights_and_shadows_gamma_corrected.jpg")
    print(f" - {hist_path.name}")
    print(f" - {grid_path.name}")

if __name__ == "__main__":
    main()