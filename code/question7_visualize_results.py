import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import cv2

root = Path(__file__).resolve().parent
test_dir = root / 'a1images' / 'a1q8images'

# Image pairs for visualization
image_pairs = [
    ('im01small.png', 'im01.png'),
    ('im02small.png', 'im02.png'),
]

fig, axes = plt.subplots(len(image_pairs), 4, figsize=(16, 10))

for idx, (small_name, large_name) in enumerate(image_pairs):
    small_path = test_dir / small_name
    large_path = test_dir / large_name
    
    if not small_path.exists() or not large_path.exists():
        continue
    
    # Load images
    small_img = np.array(Image.open(small_path))
    large_img = np.array(Image.open(large_path))
    
    # Convert to grayscale if RGB
    if small_img.ndim == 3:
        small_img = np.mean(small_img, axis=2).astype(np.uint8)
    if large_img.ndim == 3:
        large_img = np.mean(large_img, axis=2).astype(np.uint8)
    
    # Zoom using both methods
    target_size = large_img.shape
    zoomed_nn = cv2.resize(small_img, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    zoomed_bilinear = cv2.resize(small_img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    
    # Display
    axes[idx, 0].imshow(small_img, cmap='gray')
    axes[idx, 0].set_title(f'{small_name}\n(Original Small)')
    axes[idx, 0].axis('off')
    
    axes[idx, 1].imshow(zoomed_nn, cmap='gray')
    axes[idx, 1].set_title('Zoomed (Nearest-Neighbor)')
    axes[idx, 1].axis('off')
    
    axes[idx, 2].imshow(zoomed_bilinear, cmap='gray')
    axes[idx, 2].set_title('Zoomed (Bilinear)')
    axes[idx, 2].axis('off')
    
    axes[idx, 3].imshow(large_img, cmap='gray')
    axes[idx, 3].set_title(f'{large_name}\n(Original Large)')
    axes[idx, 3].axis('off')

plt.tight_layout()
plt.savefig('question7_zoom_comparison.png', dpi=150, bbox_inches='tight')
print('Saved question7_zoom_comparison.png')

# Create a results summary figure
fig, axes = plt.subplots(1, 1, figsize=(12, 6))

results = [
    ('im01', 0.002068, 0.001753),
    ('im02', 0.000404, 0.000281),
    ('im03', 0.001011, 0.000748),
    ('taylor', 0.003457, 0.003251),
]

pairs = [r[0] for r in results]
nn_ssd = [r[1] for r in results]
bilinear_ssd = [r[2] for r in results]

x = np.arange(len(pairs))
width = 0.35

bars1 = axes.bar(x - width/2, nn_ssd, width, label='Nearest-Neighbor', color='steelblue')
bars2 = axes.bar(x + width/2, bilinear_ssd, width, label='Bilinear', color='coral')

axes.set_xlabel('Image Pair', fontsize=12)
axes.set_ylabel('Normalized SSD', fontsize=12)
axes.set_title('Question 7: Zoom Interpolation Comparison (SSD)', fontsize=14)
axes.set_xticks(x)
axes.set_xticklabels(pairs)
axes.legend()
axes.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.6f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('question7_ssd_comparison.png', dpi=150, bbox_inches='tight')
print('Saved question7_ssd_comparison.png')
