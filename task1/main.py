import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def custom_bgr_to_gray(image_bgr):  # (RGB) --> сіре
    b, g, r = image_bgr[:, :, 0], image_bgr[:, :, 1], image_bgr[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.uint8)

def custom_histogram(image_gray):   # histogram
    hist = np.bincount(image_gray.ravel(), minlength=256)
    return hist

def custom_entropy(hist):
    # entropy
    total_pixels = hist.sum()
    if total_pixels == 0:
        return 0.0
    
    probabilities = hist / total_pixels

    p_non_zero = probabilities[probabilities > 0]

    entropy_value = -np.sum(p_non_zero * np.log2(p_non_zero))
    
    return entropy_value

def custom_laplacian_variance(image_gray):      # Accuracy (Lapl)
    img = image_gray.astype(float)

    padded = np.pad(img, ((1, 1), (1, 1)), mode='reflect')

    laplacian = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4 * img
    )

    return np.var(laplacian)

def analyze_frequency_domain(image_gray):   # spectrum analysis

    f = np.fft.fft2(image_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-5)
    return np.mean(magnitude_spectrum), magnitude_spectrum

def analyze_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: failed to load image {image_path}")
        return None
    gray = custom_bgr_to_gray(img) if len(img.shape) == 3 else img

    hist = custom_histogram(gray)
    ent_score = custom_entropy(hist)

    laplacian_var = custom_laplacian_variance(gray)

    fft_score, fft_viz = analyze_frequency_domain(gray)
    
    return {
        "filename": os.path.basename(image_path),
        "entropy": ent_score,
        "fft_mean": fft_score,
        "sharpness": laplacian_var,
        "image_data": img,
        "fft_viz": fft_viz
    }

def visualize_results(results):
    if not results: return
    num_imgs = len(results)
    fig, axs = plt.subplots(num_imgs, 3, figsize=(15, 5 * num_imgs))
    if num_imgs == 1: axs = np.expand_dims(axs, axis=0)

    for i, res in enumerate(results):
        img_rgb = cv2.cvtColor(res['image_data'], cv2.COLOR_BGR2RGB)
        axs[i, 0].imshow(img_rgb)
        axs[i, 0].set_title(f"{res['filename']}\nEnt: {res['entropy']:.2f}")
        axs[i, 0].axis('off')

        colors = ('b', 'g', 'r')
        for j, col in enumerate(colors):
            channel = res['image_data'][:, :, j]
            hist = custom_histogram(channel)
            axs[i, 1].plot(hist, color=col)
            
        axs[i, 1].set_title("Colour histogram")
        axs[i, 1].set_xlim([0, 256])
        
        axs[i, 2].imshow(res['fft_viz'], cmap='gray')
        axs[i, 2].set_title(f"FFT Spectrum (Mean: {res['fft_mean']:.1f})")
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    folder = "test_images" 
    if os.path.exists(folder):
        image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        results = []
        
        print(f"{'Файл':<20} | {'Entropy':<10} | {'Accuracy (Lapl)':<15} | {'FFT Score':<10}")
        print("-" * 65)
        
        for img_path in image_files:
            res = analyze_image(img_path)
            if res:
                results.append(res)
                print(f"{res['filename']:<20} | {res['entropy']:<10.4f} | {res['sharpness']:<15.2f} | {res['fft_mean']:<10.2f}")
        
        visualize_results(results)