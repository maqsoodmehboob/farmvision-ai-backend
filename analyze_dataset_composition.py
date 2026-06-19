# ==============================================================
# analyze_dataset_composition.py — Farm Vision Dataset Composition Audit
#
# PURPOSE:
#   Investigates the "shortcut learning" hypothesis: that the model
#   might be learning IMAGE COMPOSITION (zoom level, single-leaf vs
#   multi-leaf, background clutter) rather than actual leaf
#   morphology — because different crop folders were photographed
#   in systematically different styles.
#
#   This is the suspected cause of the paradox you observed:
#   an unclear MULTI-leaf image -> correctly classified as Tomato,
#   but an isolated SINGLE leaf from the same plant -> "Invalid".
#
#   This script does NOT fix anything by itself — it measures and
#   reports per-class statistics so you can SEE whether this bias
#   exists, and where. If Tomato images are systematically different
#   in any of these stats compared to the other 3 crops, that
#   confirms the hypothesis and tells you what to standardize.
#
# WHAT IT MEASURES (per crop folder):
#   1. Average image aspect ratio (wide multi-leaf shots vs tight
#      single-leaf close-ups have very different aspect ratios)
#   2. Average leaf/green coverage ratio (reuses the same HSV logic
#      as main.py's leaf_coverage_ratio — multi-leaf wide shots vs
#      single-leaf close-ups have very different coverage %)
#   3. Average edge density (busy/cluttered multi-object backgrounds
#      vs clean single-leaf backgrounds)
#   4. Average brightness (different datasets often have different
#      lighting conventions)
#
# USAGE:
#   python analyze_dataset_composition.py
#
#   Expects: datasets/CropClassifier/{Tomato,Potato,Cucumber,Cauliflower}/
#   (or edit DATASET_ROOT below to point at your disease-folder
#   datasets if you want to audit those instead)
# ==============================================================

import os
from PIL import Image, ImageFilter
import numpy as np

DATASET_ROOT     = os.path.join("datasets", "CropClassifier")
MAX_IMAGES_PER_CLASS = 200   # sample cap for speed


def analyze_image(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    aspect_ratio = w / h

    # Leaf/green coverage (same logic as main.py)
    hsv = img.convert("HSV")
    arr = np.array(hsv)
    hh, ss, vv = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    plant_mask = (hh >= 20) & (hh <= 110) & (ss >= 30) & (vv >= 20)
    green_ratio = float(np.mean(plant_mask))

    # Edge density (busy background vs clean single-leaf shot)
    gray  = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_density = float(np.array(edges, dtype=np.float32).mean())

    # Brightness
    brightness = float(np.array(gray, dtype=np.float32).mean())

    return aspect_ratio, green_ratio, edge_density, brightness


def analyze_folder(folder_path):
    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(files) > MAX_IMAGES_PER_CLASS:
        np.random.seed(42)
        files = list(np.random.choice(files, MAX_IMAGES_PER_CLASS, replace=False))

    stats = {"aspect_ratio": [], "green_ratio": [], "edge_density": [], "brightness": []}
    for fname in files:
        try:
            ar, gr, ed, br = analyze_image(os.path.join(folder_path, fname))
            stats["aspect_ratio"].append(ar)
            stats["green_ratio"].append(gr)
            stats["edge_density"].append(ed)
            stats["brightness"].append(br)
        except Exception as e:
            print(f"    ⚠️  Skipped {fname}: {e}")

    return {k: (np.mean(v), np.std(v)) for k, v in stats.items()}, len(files)


def main():
    print("\n" + "=" * 70)
    print("  Farm Vision — Dataset Composition Bias Audit")
    print("=" * 70)

    if not os.path.exists(DATASET_ROOT):
        print(f"\n❌ Folder not found: {DATASET_ROOT}")
        print("   Edit DATASET_ROOT in this script to point at your dataset.")
        return

    results = {}
    for class_name in sorted(os.listdir(DATASET_ROOT)):
        class_path = os.path.join(DATASET_ROOT, class_name)
        if not os.path.isdir(class_path):
            continue
        print(f"\n📁 Analyzing: {class_name}...")
        stats, n = analyze_folder(class_path)
        results[class_name] = stats
        print(f"   ({n} images sampled)")

    if not results:
        print("\n❌ No class folders found to analyze.")
        return

    # Print comparison table
    print("\n" + "=" * 70)
    print("  COMPARISON TABLE  (mean ± std)")
    print("=" * 70)

    metrics = ["aspect_ratio", "green_ratio", "edge_density", "brightness"]
    metric_labels = {
        "aspect_ratio": "Aspect Ratio (W/H)",
        "green_ratio":  "Green/Leaf Coverage",
        "edge_density": "Edge Density (clutter)",
        "brightness":   "Brightness (0-255)",
    }

    for metric in metrics:
        print(f"\n  {metric_labels[metric]}:")
        values = {}
        for class_name, stats in results.items():
            mean, std = stats[metric]
            values[class_name] = mean
            print(f"    {class_name:15s}: {mean:7.3f} ± {std:.3f}")

        # Flag if any class deviates heavily from the others
        all_vals  = list(values.values())
        overall_avg = np.mean(all_vals)
        overall_std = np.std(all_vals)
        for class_name, v in values.items():
            if overall_std > 0 and abs(v - overall_avg) > 1.5 * overall_std:
                print(f"    ⚠️  {class_name} deviates notably from the other classes on this metric")

    print("\n" + "=" * 70)
    print("  HOW TO READ THIS")
    print("=" * 70)
    print("""
  If one crop (e.g. Tomato) shows a meaningfully different
  Green/Leaf Coverage or Edge Density compared to the other 3 —
  that confirms the model may be learning "this image looks like
  a wide multi-leaf shot" instead of "this leaf has Tomato-specific
  texture." That would explain why an isolated single leaf from a
  Tomato plant can get rejected: it no longer matches the composition
  pattern the model actually learned.

  NEXT STEP if a bias is found:
  Re-crop/standardize that class's training images to match the
  same single-leaf, centered, similar-zoom style as the other
  classes, then retrain. This script only diagnoses the issue —
  fixing it requires re-curating the actual image files, which
  needs a human reviewing the dataset (not something to automate
  blindly, since some crops may genuinely be photographed at
  different natural distances).
""")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()