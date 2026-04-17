import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def homomorphic_filter(image_path, gamma_h=2.2, gamma_l=0.3, c=1, D0=30):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    img_log = np.log1p(np.array(img, dtype="float"))
    fft = np.fft.fft2(img_log)
    fft_shift = np.fft.fftshift(fft)
    M, N = img.shape
    H = np.zeros((M, N), dtype=np.float32)
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M/2)**2 + (v - N/2)**2)
            H[u, v] = (gamma_h - gamma_l) * (1 - np.exp(-c * (D**2) / (D0**2))) + gamma_l
    G_shift = fft_shift * H
    G = np.fft.ifftshift(G_shift)
    img_filtered_log = np.fft.ifft2(G)
    img_filtered_log = np.real(img_filtered_log)
    img_exp = np.exp(img_filtered_log) - 1
    img_exp = np.clip(img_exp, 0, None) # avoid negative values before normalization
    img_normalized = cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX)
    img_final = np.uint8(img_normalized)
    
    return img, img_final

if __name__ == "__main__":
    # Choose an image that likely has varying illumination
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_img_path = os.path.join(project_root, "a1images", "highlights_and_shadows.jpg")
    
    try:
        # Tweak parameters based on typical values
        original_img, filtered_img = homomorphic_filter(default_img_path, gamma_h=2.0, gamma_l=0.5, c=1, D0=30)
        
        # Save results
        results_dir = os.path.join(project_root, "saved_results")
        os.makedirs(results_dir, exist_ok=True)
        
        cv2.imwrite(os.path.join(results_dir, "q12_original.jpg"), original_img)
        cv2.imwrite(os.path.join(results_dir, "q12_homomorphic_filtered.jpg"), filtered_img)
        
        # Plotting
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(filtered_img, cmap='gray')
        plt.title('Homomorphic Filtered')
        plt.axis('off')
        
        plot_path = os.path.join(results_dir, "q12_homomorphic_comparison.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Results saved to {results_dir}")
        plt.close()
        
    except FileNotFoundError as e:
        print(e)
