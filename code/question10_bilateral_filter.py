import cv2
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt

def manual_bilateral_filter(image, diameter, sigma_s, sigma_r):
    if diameter % 2 == 0:
        diameter += 1
    radius = diameter // 2
    h, w = image.shape
    output = np.zeros_like(image, dtype=np.float32)
    padded_image = cv2.copyMakeBorder(image, radius, radius, radius, radius, cv2.BORDER_REFLECT).astype(np.float32)
    y_grid, x_grid = np.mgrid[-radius:radius+1, -radius:radius+1]
    spatial_weights = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma_s**2))
    for i in range(h):
        for j in range(w):
            local_region = padded_image[i:i+diameter, j:j+diameter]
            center_val = padded_image[i+radius, j+radius]
            range_weights = np.exp(-((local_region - center_val)**2) / (2 * sigma_r**2))
            weights = spatial_weights * range_weights
            output[i, j] = np.sum(local_region * weights) / np.sum(weights)
    return np.clip(output, 0, 255).astype(np.uint8)


def main():
    # Setup paths
    input_image_path = Path('../a1images/jeniffer.jpg')

    if not input_image_path.exists():
        input_image_path = Path('a1images/jeniffer.jpg')
    if not input_image_path.exists():
        input_image_path = Path('../a1images/einstein.png')
    if not input_image_path.exists():
        input_image_path = Path('a1images/einstein.png')

    output_dir = Path('saved_results')
    if not output_dir.exists() and Path('../saved_results').exists():
        output_dir = Path('../saved_results')
    elif not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading image from: {input_image_path}")
    image = cv2.imread(str(input_image_path))
    if image is None:
        print(f"Error: Could not load image from {input_image_path}")
        return

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Parameters
    diameter = 9
    sigma_s = 75.0
    sigma_r = 50.0

    results = [("Original", gray_image)]

    # (b) Gaussian smoothing using OpenCV
    print("Applying OpenCV Gaussian Blur...")
    gaussian_cv = cv2.GaussianBlur(gray_image, (diameter, diameter), sigmaX=sigma_s, sigmaY=sigma_s)
    results.append(("Gaussian (OpenCV)", gaussian_cv))

    # (c) Bilateral filtering using OpenCV
    print("Applying OpenCV Bilateral Filter...")
    bilateral_cv = cv2.bilateralFilter(gray_image, d=diameter, sigmaColor=sigma_r, sigmaSpace=sigma_s)
    results.append(("Bilateral (OpenCV)", bilateral_cv))

    # (d) Manual bilateral filtering
    print(f"Applying Manual Bilateral Filter (d={diameter}, sigma_s={sigma_s}, sigma_r={sigma_r})...")
    print("Please wait, manual implementation may take a few seconds...")
    start_time = time.time()
    bilateral_manual = manual_bilateral_filter(gray_image, diameter, sigma_s, sigma_r)
    end_time = time.time()
    print(f"Manual Bilateral Filter complete in {end_time - start_time:.2f} seconds.")
    results.append(("Bilateral (Manual)", bilateral_manual))

    # Save individual results
    for label, res in results:
        safe_label = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out_path = output_dir / f'question10_{safe_label}.png'
        cv2.imwrite(str(out_path), res)
        print(f"Saved: {out_path}")

    # ✅ FIX: Create comparison grid with large, visible titles on TOP of each image
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, (label, res) in enumerate(results):
        axes[i].imshow(res, cmap='gray')
        axes[i].set_title(label, fontsize=18, fontweight='bold', pad=12)  # ✅ Large bold title
        axes[i].axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.25)  # ✅ Extra vertical space for titles
    plt.tight_layout()

    grid_path = output_dir / 'question10_comparison_grid.png'
    plt.savefig(grid_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved comparison grid: {grid_path}")

if __name__ == "__main__":
    main()