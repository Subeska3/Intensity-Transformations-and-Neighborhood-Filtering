from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(path: Path) -> np.ndarray:
    image = Image.open(path)
    return np.asarray(image, dtype=np.float32) / 255.0

def save_image(image: np.ndarray, path: Path) -> None:
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image_uint8).save(path)

def gamma_correction_lab_L(image_lab: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("Gamma must be a positive value.")
    # image_lab is in LAB, L is first channel, scaled 0-100
    L, a, b = image_lab[:, :, 0], image_lab[:, :, 1], image_lab[:, :, 2]
    L_corrected = 100 * np.power(L / 100, gamma)
    return np.stack([L_corrected, a, b], axis=-1)

def plot_histograms(original_L, corrected_L, title1, title2):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(original_L.ravel(), bins=256, range=(0, 100), color='gray', alpha=0.7)
    plt.title(title1)
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.hist(corrected_L.ravel(), bins=256, range=(0, 100), color='gray', alpha=0.7)
    plt.title(title2)
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('q2_histograms.png')  # Save instead of show
    print("Histograms saved as q2_histograms.png")

def main() -> None:
    root = Path(__file__).resolve().parent
    input_path = root / "a1images" / "highlights_and_shadows.jpg"
    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find image at {input_path}")

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
    save_image(corrected_rgb, root / "highlights_and_shadows_gamma_corrected.jpg")
    
    # Plot histograms of L channels
    original_L = lab_array[:, :, 0]
    corrected_L = lab_corrected[:, :, 0]
    plot_histograms(original_L, corrected_L, 'Original L Channel Histogram', 'Gamma Corrected L Channel Histogram')
    
    print(f"Gamma value used: {gamma}")
    print("Saved: highlights_and_shadows_gamma_corrected.jpg")

if __name__ == "__main__":
    main()