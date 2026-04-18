import cv2
import numpy as np
from pathlib import Path
from PIL import Image

try:
    import fitz
except ImportError:
    fitz = None


def extract_figure4_from_pdf(pdf_path: Path, out_path: Path) -> bool:
    if fitz is None:
        return False
    doc = fitz.open(str(pdf_path))
    # Search all pages for the label "Figure 4"
    for page_index, page in enumerate(doc):
        rects = page.search_for('Figure 4')
        if not rects:
            rects = page.search_for('Figure4')
        if rects:
            rect = rects[0]
            margin = fitz.Rect(-50, -300, 50, 50)
            clip = rect + margin
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=clip)
            pix.save(str(out_path))
            return True
    # fallback: render page 4 entirely
    if len(doc) >= 4:
        page = doc[3]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        pix.save(str(out_path))
        return True
    return False


def load_noisy_image(root: Path, filename: str | None = None) -> np.ndarray:
    candidates = []
    if filename:
        candidates.append(root / filename)
    candidates.extend([
        root / 'question8_input.png',
        root.parent / 'saved_results' / 'question8_input.png',
        root / 'page3_fig4_crop_auto2.png',
        root / 'page3_fig4_crop.png',
        root / 'page4_render.png',
    ])
    for candidate in candidates:
        if candidate.exists():
            image = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                return image
    pdf_path = root / 'it5437_2026_assignment_01.pdf'
    if pdf_path.exists() and fitz is not None:
        if extract_figure4_from_pdf(pdf_path, root / 'question8_input.png'):
            return cv2.imread(str(root / 'question8_input.png'), cv2.IMREAD_GRAYSCALE)
    filename_msg = f' or {filename}' if filename else ''
    raise FileNotFoundError(f'Cannot find Figure 4 input image{filename_msg}. Place the salt-and-pepper image as question8_input.png in the assignment folder.')


def apply_gaussian_smoothing(image: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma)


def apply_median_filter(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.medianBlur(image, ksize)


def save_comparison(original: np.ndarray, gaussian: np.ndarray, median: np.ndarray, path: Path) -> None:
    h, w = original.shape
    canvas = np.zeros((h, w * 3), dtype=np.uint8)
    canvas[:, 0:w] = original
    canvas[:, w:2*w] = gaussian
    canvas[:, 2*w:3*w] = median
    canvas = cv2.putText(canvas.copy(), 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
    canvas = cv2.putText(canvas, 'Gaussian', (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
    canvas = cv2.putText(canvas, 'Median', (2*w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
    cv2.imwrite(str(path), canvas)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description='Denoise salt-and-pepper image using Gaussian and median filters.')
    parser.add_argument('--input', '-i', type=str, default=None, help='Input noisy image filename in the assignment folder')
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    noisy = load_noisy_image(root, args.input)
    gaussian = apply_gaussian_smoothing(noisy, ksize=5, sigma=1.0)
    median = apply_median_filter(noisy, ksize=5)

    output_dir = root.parent / 'saved_results'
    output_dir.mkdir(exist_ok=True)

    cv2.imwrite(str(output_dir / 'question8_original.png'), noisy)
    cv2.imwrite(str(output_dir / 'question8_gaussian.png'), gaussian)
    cv2.imwrite(str(output_dir / 'question8_median.png'), median)
    save_comparison(noisy, gaussian, median, output_dir / 'question8_comparison.png')

    print(f'Saved results to {output_dir}')

    import matplotlib.pyplot as plt
    f = noisy
    g_median = median

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(f, cmap='gray', vmin=0, vmax=255)
    ax[0].set_title('Corrupted Image (Salt & Pepper)')
    ax[0].axis('off')
    ax[1].imshow(g_median, cmap='gray', vmin=0, vmax=255)
    ax[1].set_title('Median Filtering (5×5)')
    ax[1].axis('off')
    plt.tight_layout()
    
    # Save the figure as well for record
    fig.savefig(output_dir / 'question8_matplotlib_comparison.png')
    print(f'Saved: {output_dir / "question8_matplotlib_comparison.png"}')
    
    # plt.show() # Commented out for automated environment, use fig.savefig instead

    # Print some statistics
    print('Original stats:', noisy.min(), noisy.max(), noisy.mean(), noisy.std())
    print('Gaussian stats:', gaussian.min(), gaussian.max(), gaussian.mean(), gaussian.std())
    print('Median stats:', median.min(), median.max(), median.mean(), median.std())


if __name__ == '__main__':
    main()
