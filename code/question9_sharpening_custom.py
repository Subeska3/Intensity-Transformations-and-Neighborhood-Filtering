import cv2
import numpy as np
from pathlib import Path

def unsharp_mask(image, sigma=1.0, strength=1.5):
    """
    Apply unsharp masking to sharpen the image.
    """
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)

    # Compute the mask (high-pass filter)
    # mask = original - blurred
    mask = cv2.subtract(image.astype(np.float32), blurred.astype(np.float32))

    # Add the mask back to the original image
    # sharpened = original + strength * mask
    sharpened = cv2.addWeighted(image.astype(np.float32), 1.0, mask, strength, 0)

    # Clip to valid range and convert back to uint8
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened

def main():
    # Paths
    input_image_path = Path('a1images/einstein.png')
    output_dir = Path('saved_results')
    output_dir.mkdir(exist_ok=True)
    
    # Load the image
    image = cv2.imread(str(input_image_path))
    if image is None:
        print(f"Error: Could not load image from {input_image_path}")
        return

    # Sharpening strengths to try
    strengths = [0.5, 1.5, 3.0]
    
    results = []
    # Add original
    results.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    for s in strengths:
        sharpened = unsharp_mask(gray_image, sigma=1.0, strength=s)
        results.append(sharpened)
        
        # Save individual result
        out_path = output_dir / f'question9_sharpened_{input_image_path.stem}_s{s}.png'
        cv2.imwrite(str(out_path), sharpened)
        print(f"Sharpened image (strength={s}) saved to {out_path}")

    # Create a comparison grid with labels
    h, w = gray_image.shape
    labeled_results = []
    
    labels = ["Original"] + [f"S={s}" for s in strengths]
    
    for res, label in zip(results, labels):
        # Create a copy to avoid modifying the original result
        labeled_res = res.copy()
        # Add text label
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(labeled_res, label, (10, 30), font, 1, (255,), 2, cv2.LINE_AA)
        labeled_results.append(labeled_res)

    comparison = np.hstack(labeled_results)
        
    grid_path = output_dir / 'question9_sharpening_comparison.png'
    cv2.imwrite(str(grid_path), comparison)
    print(f"Comparison grid with labels saved to {grid_path}")

if __name__ == "__main__":
    main()
