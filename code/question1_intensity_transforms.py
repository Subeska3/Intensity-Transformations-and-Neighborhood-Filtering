from pathlib import Path

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
        raise FileNotFoundError(f"Cannot find runway image at {input_path}")

    image = load_image_gray(input_path)
    save_image(image, root / "runway_gray.png")

    gamma_05 = gamma_correction(image, 0.5)
    save_image(gamma_05, root / "runway_gamma_0.5.png")

    gamma_2 = gamma_correction(image, 2.0)
    save_image(gamma_2, root / "runway_gamma_2.0.png")

    stretch = contrast_stretch(image, r1=0.2, r2=0.8)
    save_image(stretch, root / "runway_contrast_stretch_r1_0.2_r2_0.8.png")

    print("Saved:")
    print(" - runway_gray.png")
    print(" - runway_gamma_0.5.png")
    print(" - runway_gamma_2.0.png")
    print(" - runway_contrast_stretch_r1_0.2_r2_0.8.png")


if __name__ == "__main__":
    main()
