# ==============================================================
# calibrate_blur_threshold.py — Farm Vision Threshold Calibration Tool
#
# PURPOSE:
#   Run this against a folder of your own real photos (mix of sharp
#   and blurry leaf images) to see what scores they actually get.
#   This tells you where to set BLUR_EDGE_VARIANCE_THRESHOLD in
#   main.py — instead of guessing.
#
# HOW TO USE:
#   1. Put your test photos in two folders:
#        test_photos/sharp/    ← 10-15 clear, in-focus leaf photos
#        test_photos/blurry/   ← 10-15 deliberately blurry/shaky photos
#   2. Run: python calibrate_blur_threshold.py
#   3. Look at the printed scores and the suggested threshold at the end.
#
# No TensorFlow needed — this only tests the blur-detection logic,
# which is pure PIL/numpy math.
# ==============================================================

import os
from PIL import Image, ImageFilter
import numpy as np

TEST_DIR = "test_photos"   # expects test_photos/sharp/ and test_photos/blurry/


def blur_score(image_path):
    """Same logic as is_blurry() in main.py — returns the raw score."""
    img   = Image.open(image_path).convert("L")          # grayscale
    edges = img.filter(ImageFilter.FIND_EDGES)
    arr   = np.array(edges, dtype=np.float32)
    return float(arr.var())


def scan_folder(folder_path):
    scores = []
    if not os.path.exists(folder_path):
        print(f"  ⚠️  Folder not found: {folder_path}")
        return scores

    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        fpath = os.path.join(folder_path, fname)
        try:
            score = blur_score(fpath)
            scores.append(score)
            print(f"    {fname:35s} → score: {score:8.1f}")
        except Exception as e:
            print(f"    {fname:35s} → ERROR: {e}")
    return scores


def main():
    print("\n" + "=" * 60)
    print("  Farm Vision — Blur Threshold Calibration")
    print("=" * 60)

    sharp_dir  = os.path.join(TEST_DIR, "sharp")
    blurry_dir = os.path.join(TEST_DIR, "blurry")

    print(f"\n📸 SHARP images ({sharp_dir}):")
    sharp_scores = scan_folder(sharp_dir)

    print(f"\n📷 BLURRY images ({blurry_dir}):")
    blurry_scores = scan_folder(blurry_dir)

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    if not sharp_scores or not blurry_scores:
        print("\n⚠️  Need at least a few images in BOTH folders to calibrate.")
        print(f"   Create: {sharp_dir}/  and  {blurry_dir}/")
        print("   Put 10-15 real photos in each, then run this script again.")
        return

    sharp_min  = min(sharp_scores)
    sharp_avg  = sum(sharp_scores) / len(sharp_scores)
    blurry_max = max(blurry_scores)
    blurry_avg = sum(blurry_scores) / len(blurry_scores)

    print(f"\n  Sharp images   → min: {sharp_min:.1f}   avg: {sharp_avg:.1f}   (n={len(sharp_scores)})")
    print(f"  Blurry images  → max: {blurry_max:.1f}   avg: {blurry_avg:.1f}   (n={len(blurry_scores)})")

    if blurry_max < sharp_min:
        # Clean separation — pick the midpoint
        suggested = (blurry_max + sharp_min) / 2
        print(f"\n  ✅ Clean separation found!")
        print(f"  👉 Suggested BLUR_EDGE_VARIANCE_THRESHOLD = {suggested:.1f}")
        print(f"     (Set this value in main.py)")
    else:
        # Overlap — there's no single clean cutoff
        suggested = (sharp_avg + blurry_avg) / 2
        print(f"\n  ⚠️  Some overlap between sharp and blurry scores.")
        print(f"     This means blur alone won't perfectly separate every case —")
        print(f"     that's expected, the other 2 gates (leaf coverage + ImageNet")
        print(f"     sanity check) help catch what this one misses.")
        print(f"  👉 Suggested BLUR_EDGE_VARIANCE_THRESHOLD = {suggested:.1f}")
        print(f"     (Set this value in main.py — biased toward not rejecting")
        print(f"     genuinely sharp photos)")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()