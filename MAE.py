import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Optional

# One histogram:
# good = all label1 images from veis, salami, rulaad, kalkun
# bad  = all images from no_label/label1
products = ["veis", "salami", "rulaad", "kalkun"]
label = "product_area"
bad_dir = os.path.join("no_label", label)

rescale_factor = 1.0
use_hist_equalization = True
valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def load_and_preprocess_image(path: str, rescale_factor: float = 1.0) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Could not read image: {path}")
        return None

    if rescale_factor != 1.0:
        new_w = max(1, int(img.shape[1] * rescale_factor))
        new_h = max(1, int(img.shape[0] * rescale_factor))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if use_hist_equalization:
        gray = cv2.equalizeHist(gray)

    return gray


def resize_to_match(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
    return img1, img2


def mean_absolute_error_arrays(img1: np.ndarray, img2: np.ndarray) -> float:
    img1, img2 = resize_to_match(img1, img2)
    return float(np.mean(np.abs(img1.astype(np.float64) - img2.astype(np.float64))))


def get_image_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        print(f"Folder does not exist: {folder}")
        return []

    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(valid_extensions)
    ]
    return sorted(files)


def compute_mae_to_templates(
    image_path: str,
    template_paths: List[str],
    rescale_factor: float = 1.0
) -> Tuple[Optional[float], Optional[float]]:
    start_time = time.perf_counter()

    image = load_and_preprocess_image(image_path, rescale_factor)
    if image is None:
        return None, None

    maes = []
    for template_path in template_paths:
        template = load_and_preprocess_image(template_path, rescale_factor)
        if template is None:
            continue
        maes.append(mean_absolute_error_arrays(template, image))

    if not maes:
        return None, None

    end_time = time.perf_counter()
    return min(maes), end_time - start_time


def suggest_threshold(mae_good_values: List[float], mae_bad_values: List[float]) -> Optional[float]:
    if not mae_good_values or not mae_bad_values:
        return None

    good_max = max(mae_good_values)
    bad_min = min(mae_bad_values)

    if good_max < bad_min:
        return (good_max + bad_min) / 2.0

    return (float(np.median(mae_good_values)) + float(np.median(mae_bad_values))) / 2.0


def print_stats(name: str, values: List[float]) -> None:
    if not values:
        print(f"{name}: no values")
        return

    print(f"\n{name} stats")
    print(f"Count:  {len(values)}")
    print(f"Min:    {min(values):.4f}")
    print(f"Max:    {max(values):.4f}")
    print(f"Mean:   {np.mean(values):.4f}")
    print(f"Median: {np.median(values):.4f}")
    print(f"Std:    {np.std(values):.4f}")


def save_mae_histogram(
    mae_good_values: List[float],
    mae_bad_values: List[float],
    label: str,
    rescale_factor: float,
    threshold: Optional[float],
    out_dir: str = "results",
    bins: int = 30
) -> None:
    if not (mae_good_values or mae_bad_values):
        print("No MAE values to plot")
        return

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(9, 5))

    if mae_good_values:
        plt.hist(mae_good_values, bins=bins, alpha=0.65, label="all products label1")
    if mae_bad_values:
        plt.hist(mae_bad_values, bins=bins, alpha=0.65, label="no_label label1")

    if threshold is not None:
        plt.axvline(threshold, linestyle="--", linewidth=2, label=f"threshold = {threshold:.2f}")

    plt.xlabel("MAE")
    plt.ylabel("Count")
    plt.title("All products label1 vs no_label label1 (MAE)")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"mae_histogram_all_products_{label}_{rescale_factor}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved histogram: {out_path}")


def main():
    all_good_images: List[str] = []

    for product in products:
        good_dir = os.path.join(product, label)
        product_images = get_image_files(good_dir)
        all_good_images.extend(product_images)

    bad_images = get_image_files(bad_dir)

    if not all_good_images:
        print("No good images found.")
        return

    if len(all_good_images) < 2:
        print("Need at least 2 good images.")
        return

    # Use ALL good images as templates
    template_paths = all_good_images.copy()

    print("Using all good images as templates:")
    print(f"Template count: {len(template_paths)}")

    mae_good = []
    mae_bad = []
    calculation_times = []

    # For good images, compare each image to all OTHER good images
    for img_path in all_good_images:
        other_templates = [p for p in template_paths if p != img_path]
        if not other_templates:
            continue

        mae_value, calc_time = compute_mae_to_templates(img_path, other_templates, rescale_factor)
        if mae_value is not None:
            mae_good.append(mae_value)
            calculation_times.append(calc_time)

    # For bad images, compare to all good templates
    for img_path in bad_images:
        mae_value, calc_time = compute_mae_to_templates(img_path, template_paths, rescale_factor)
        if mae_value is not None:
            mae_bad.append(mae_value)
            calculation_times.append(calc_time)

    print(f"\nMAE good: {len(mae_good)} values")
    print(f"MAE bad:  {len(mae_bad)} values")

    print_stats("GOOD (all label1 products)", mae_good)
    print_stats("BAD (no_label label1)", mae_bad)

    threshold = suggest_threshold(mae_good, mae_bad)
    if threshold is not None:
        print(f"\nSuggested threshold: {threshold:.4f}")
    else:
        print("\nCould not suggest threshold automatically")

    save_mae_histogram(
        mae_good,
        mae_bad,
        label,
        rescale_factor,
        threshold=threshold
    )

    if calculation_times:
        avg_time_ms = (sum(calculation_times) / len(calculation_times)) * 1000.0
        print(f"\nAverage comparison time: {avg_time_ms:.2f} ms")


if __name__ == "__main__":
    main()