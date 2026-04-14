import numpy as np
from PIL import Image
from pathlib import Path
import cv2

def zoom_nearest_neighbor(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Zoom an image using nearest-neighbor interpolation.
    
    Args:
        image: Input image as numpy array (grayscale or RGB)
        scale_factor: Scaling factor s in (0, 10]
    
    Returns:
        Zoomed image
    """
    if scale_factor <= 0 or scale_factor > 10:
        raise ValueError("Scale factor must be in (0, 10]")
    
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    if image.ndim == 2:  # Grayscale
        zoomed = np.zeros((new_h, new_w), dtype=image.dtype)
        for i in range(new_h):
            for j in range(new_w):
                src_i = int(i / scale_factor)
                src_j = int(j / scale_factor)
                src_i = min(src_i, h - 1)
                src_j = min(src_j, w - 1)
                zoomed[i, j] = image[src_i, src_j]
    else:  # RGB
        zoomed = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
        for i in range(new_h):
            for j in range(new_w):
                src_i = int(i / scale_factor)
                src_j = int(j / scale_factor)
                src_i = min(src_i, h - 1)
                src_j = min(src_j, w - 1)
                zoomed[i, j] = image[src_i, src_j]
    
    return zoomed


def zoom_bilinear(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Zoom an image using bilinear interpolation.
    
    Args:
        image: Input image as numpy array (grayscale or RGB)
        scale_factor: Scaling factor s in (0, 10]
    
    Returns:
        Zoomed image
    """
    if scale_factor <= 0 or scale_factor > 10:
        raise ValueError("Scale factor must be in (0, 10]")
    
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    # Use OpenCV's resize for efficiency
    if image.ndim == 2:  # Grayscale
        zoomed = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:  # RGB
        zoomed = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    return zoomed


def compute_normalized_ssd(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute normalized sum of squared differences between two images.
    
    Args:
        img1: First image
        img2: Second image (must be same size as img1)
    
    Returns:
        Normalized SSD value
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")
    
    # Convert to float for computation
    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)
    
    # Compute SSD
    ssd = np.sum((img1_float - img2_float) ** 2)
    
    # Normalize by image size and range
    n_pixels = np.prod(img1.shape[:-1] if img1.ndim == 3 else img1.shape)
    max_val = 255.0  # Assuming 8-bit images
    normalized_ssd = ssd / (n_pixels * max_val ** 2)
    
    return normalized_ssd


def test_zoom_algorithms():
    """
    Test zoom algorithms on provided image pairs.
    """
    root = Path(__file__).resolve().parent
    test_dir = root / 'a1images' / 'a1q8images'
    
    # Image pairs: (small, large)
    image_pairs = [
        ('im01small.png', 'im01.png'),
        ('im02small.png', 'im02.png'),
        ('im03small.png', 'im03.png'),
        ('taylor_small.jpg', 'taylor.jpg'),
    ]
    
    results = []
    
    for small_name, large_name in image_pairs:
        small_path = test_dir / small_name
        large_path = test_dir / large_name
        
        if not small_path.exists() or not large_path.exists():
            print(f"Skipping {small_name} - {large_name}: files not found")
            continue
        
        # Load images
        small_img = np.array(Image.open(small_path))
        large_img = np.array(Image.open(large_path))
        
        # Convert to grayscale if RGB
        if small_img.ndim == 3:
            small_img = np.mean(small_img, axis=2).astype(np.uint8)
        if large_img.ndim == 3:
            large_img = np.mean(large_img, axis=2).astype(np.uint8)
        
        # Calculate required scale factor
        scale_factor = large_img.shape[0] / small_img.shape[0]
        
        # Apply zoom methods with exact target size
        target_size = large_img.shape
        zoomed_nn = cv2.resize(small_img, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
        zoomed_bilinear = cv2.resize(small_img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # Compute SSD
        ssd_nn = compute_normalized_ssd(zoomed_nn, large_img)
        ssd_bilinear = compute_normalized_ssd(zoomed_bilinear, large_img)
        
        results.append({
            'pair': f"{small_name} -> {large_name}",
            'scale_factor': scale_factor,
            'ssd_nn': ssd_nn,
            'ssd_bilinear': ssd_bilinear,
            'small_size': small_img.shape,
            'large_size': large_img.shape
        })
        
        print(f"\n{small_name} -> {large_name}")
        print(f"  Scale factor: {scale_factor:.2f}x")
        print(f"  Small size: {small_img.shape} -> Large size: {large_img.shape}")
        print(f"  Normalized SSD (Nearest-Neighbor): {ssd_nn:.6f}")
        print(f"  Normalized SSD (Bilinear): {ssd_bilinear:.6f}")
        print(f"  Bilinear is {ssd_nn/ssd_bilinear:.2f}x better")
        
        # Save zoomed images for visualization
        Image.fromarray(zoomed_nn).save(root / f"zoomed_nn_{small_name}")
        Image.fromarray(zoomed_bilinear).save(root / f"zoomed_bilinear_{small_name}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for result in results:
        print(f"\n{result['pair']}")
        print(f"  Nearest-Neighbor SSD: {result['ssd_nn']:.6f}")
        print(f"  Bilinear SSD: {result['ssd_bilinear']:.6f}")
        improvement = (result['ssd_nn'] - result['ssd_bilinear']) / result['ssd_nn'] * 100
        print(f"  Improvement: {improvement:.2f}%")
    
    # Average improvement
    avg_nn = np.mean([r['ssd_nn'] for r in results])
    avg_bilinear = np.mean([r['ssd_bilinear'] for r in results])
    print(f"\nAverage Normalized SSD (Nearest-Neighbor): {avg_nn:.6f}")
    print(f"Average Normalized SSD (Bilinear): {avg_bilinear:.6f}")
    print(f"Average improvement: {(avg_nn - avg_bilinear) / avg_nn * 100:.2f}%")


if __name__ == "__main__":
    test_zoom_algorithms()
