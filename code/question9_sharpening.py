import cv2
import numpy as np
from pathlib import Path

def unsharp_mask(image, sigma=1.0, strength=1.5):
    """
    Apply unsharp masking to sharpen the image.

    Parameters:
    - image: Input image (grayscale or color)
    - sigma: Standard deviation for Gaussian blur
    - strength: Sharpening strength factor

    Returns:
    - Sharpened image
    """
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)

    # Compute the mask (high-pass filter)
    mask = cv2.subtract(image.astype(np.float32), blurred.astype(np.float32))

    # Add the mask back to the original image
    sharpened = cv2.addWeighted(image.astype(np.float32), 1.0, mask, strength, 0)

    # Clip to valid range and convert back to uint8
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened

def main():
    # Paths
    input_image_path = Path('a1images/einstein.png')
    output_dir = Path('saved_results')
    output_dir.mkdir(exist_ok=True)
    output_image_path = output_dir / 'question9_sharpened_einstein.png'

    # Load the image
    image = cv2.imread(str(input_image_path))
    if image is None:
        print(f"Error: Could not load image from {input_image_path}")
        return

    # Convert to grayscale for sharpening
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply unsharp masking
    sharpened = unsharp_mask(gray_image, sigma=1.0, strength=1.5)

    # Save the sharpened image
    cv2.imwrite(str(output_image_path), sharpened)
    print(f"Sharpened image saved to {output_image_path}")

if __name__ == "__main__":
    main()