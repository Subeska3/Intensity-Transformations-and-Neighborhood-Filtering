import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# List of images to display for Question 6
images = [
    ('runway_gray.png', 'Grayscale Original'),
    ('runway_dog_grad_x.png', 'DoG Horizontal Gradient'),
    ('runway_dog_grad_y.png', 'DoG Vertical Gradient'),
    ('derivative_gaussian_51x51_x_surface.png', '51x51 DoG Kernel Surface Plot'),
    ('runway_dog_gradients_combined.png', 'Combined Gradients'),
    ('question6_sobel_comparison.png', 'Sobel vs DoG Comparison')  # Will be created in (e)
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (img_path, title) in enumerate(images):
    try:
        img = mpimg.imread(img_path)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title, fontsize=12)
        axes[i].axis('off')
    except FileNotFoundError:
        axes[i].text(0.5, 0.5, f'Image not found:\n{img_path}', 
                    ha='center', va='center', transform=axes[i].transAxes)
        axes[i].set_title(title, fontsize=12)
        axes[i].axis('off')

plt.tight_layout()
plt.savefig('question1_all_results.png', dpi=150, bbox_inches='tight')
print('Saved question1_all_results.png')