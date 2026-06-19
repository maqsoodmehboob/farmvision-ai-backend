# ==============================================================
# build_centroids.py — Farm Vision Feature-Similarity Gate Builder
#
# PURPOSE:
#   Strengthens OOD (out-of-distribution) rejection for cases like
#   "Mango leaf -> Tomato" WITHOUT retraining the crop classifier.
#
#   How it works: instead of trusting the final softmax probability
#   (which is always forced to pick one of 4 classes, even for a
#   mango leaf it has never seen), we look at the FEATURE VECTOR the
#   network produces just before the final classification layer.
#   This vector is the model's internal "fingerprint" of the image.
#
#   We compute the AVERAGE fingerprint (centroid) for each of the
#   4 known crops using your existing training images, then save
#   those centroids to disk. At inference time, main.py compares a
#   new image's fingerprint to all 4 centroids using cosine
#   similarity — if it isn't close to ANY of them, it gets rejected
#   as unknown, regardless of what softmax says.
#
#   This uses your ALREADY-TRAINED crop_classifier.h5 as-is. No
#   retraining required. Run this once (or after every retrain).
#
# REQUIRES:
#   - models/crop_classifier.h5  (already trained)
#   - datasets/CropClassifier/{Tomato,Potato,Cucumber,Cauliflower}/
#     (the same training images already used to train the classifier)
#
# OUTPUT:
#   - models/crop_centroids.json
#
# USAGE:
#   python build_centroids.py
# ==============================================================

import os
import json
import numpy as np
import tensorflow as tf

MODEL_DIR        = "models"
DATASET_DIR      = "datasets"
CROP_CLASSIFIER_FOLDER = "crop_dataset"
CROP_CLASSIFIER_ROOT   = "."   # crop_dataset is at backend_python/crop_dataset, NOT inside datasets/
IMG_SIZE          = (224, 224)
MAX_IMAGES_PER_CLASS = 300

# Existing dataset structure (same as train_model.py CROP_CONFIG)
# Script will use these if datasets/CropClassifier/ doesn't exist
EXISTING_CROP_STRUCTURE = {
    "Tomato":      {"folder": "Tomato Diseases in Pakistan", "mode": "split"},   # datasets/Tomato Diseases in Pakistan/train/
    "Potato":      {"folder": "Potato",                      "mode": "split"},   # datasets/Potato/Training/
    "Cucumber":    {"folder": "Cucumber",                    "mode": "auto"},    # datasets/Cucumber/<disease>/
    "Cauliflower": {"folder": "Cauliflower",                 "mode": "auto"},    # datasets/Cauliflower/<disease>/
}


def load_feature_extractor():
    model_path = os.path.join(MODEL_DIR, "crop_classifier.h5")
    if not os.path.exists(model_path):
        print(f"❌ Crop classifier not found at {model_path}")
        return None

    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"✅ Loaded crop_classifier.h5")
    print(f"   Last 4 layers: {[l.name for l in model.layers[-4:]]}")

    # Sequential models need to be "called" on dummy data once before
    # model.input becomes accessible — this is a Keras quirk with .h5 loading
    try:
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = model(dummy, training=False)
        print(f"   Model built via dummy call ✅")
    except Exception as e:
        print(f"   ⚠️  Dummy call failed: {e}")

    # Find the best feature layer:
    # We want the last Dense layer BEFORE the final softmax output Dense.
    # Layer order (from output): dense_1 (softmax) → dropout → dense → ...
    # We want 'dense' (index -3) — the actual learned feature representation
    # SKIP dropout (-2) because it randomly zeros neurons at training time;
    # at inference training=False it's a passthrough anyway, but -3 (dense)
    # is cleaner as a feature vector.
    feature_layer = None
    for layer in reversed(model.layers):
        if hasattr(layer, 'units') and layer.name != model.layers[-1].name:
            # First Dense layer going backwards that isn't the output layer
            feature_layer = layer
            break

    if feature_layer is None:
        # Fallback — use second-to-last layer
        feature_layer = model.layers[-3]

    print(f"   Using layer '{feature_layer.name}' as feature vector")

    try:
        extractor = tf.keras.Model(inputs=model.input, outputs=feature_layer.output)
        print(f"   Feature extractor ready ✅  (output shape: {feature_layer.output_shape})")
        return extractor
    except Exception as e:
        print(f"   ⚠️  tf.keras.Model failed: {e}")
        print(f"   Trying Sequential slice approach...")
        # Fallback for stubborn Sequential models — rebuild using layers
        idx = model.layers.index(feature_layer)
        extractor = tf.keras.Sequential(model.layers[:idx+1])
        extractor(dummy, training=False)  # build it
        print(f"   Feature extractor ready via Sequential slice ✅")
        return extractor


def collect_images_for_crop(crop_name):
    """
    Collects image paths for a crop from existing dataset structure.
    Tries multiple common folder patterns automatically.
    """
    info   = EXISTING_CROP_STRUCTURE.get(crop_name, {})
    folder = info.get("folder", crop_name)
    mode   = info.get("mode", "auto")

    # Option 1: dedicated CropClassifier folder
    simple_path = os.path.join(CROP_CLASSIFIER_ROOT, CROP_CLASSIFIER_FOLDER, crop_name)
    if os.path.exists(simple_path):
        files = [os.path.join(simple_path, f) for f in os.listdir(simple_path)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if files:
            print(f"  Using: {simple_path} ({len(files)} images)")
            return files

    # Option 2+: Try multiple possible root paths for the crop
    candidates = []
    if mode == "split":
        candidates = [
            os.path.join(DATASET_DIR, folder, "train"),     # lowercase
            os.path.join(DATASET_DIR, folder, "Train"),     # capital T
            os.path.join(DATASET_DIR, folder, "Training"),  # full word capital
            os.path.join(DATASET_DIR, folder, "training"),  # full word lower
            os.path.join(DATASET_DIR, folder),              # flat — no subfolder
        ]
    else:
        candidates = [
            os.path.join(DATASET_DIR, folder),            # datasets/Cucumber/
            os.path.join(DATASET_DIR, folder, "train"),
        ]

    for root in candidates:
        if not os.path.exists(root):
            continue

        # Check if root itself contains images directly
        direct_imgs = [os.path.join(root, f) for f in os.listdir(root)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if direct_imgs:
            print(f"  Using: {root} ({len(direct_imgs)} images directly)")
            return direct_imgs

        # Check if root contains disease class subfolders
        all_files = []
        for entry in os.listdir(root):
            sub = os.path.join(root, entry)
            if not os.path.isdir(sub):
                continue
            imgs = [os.path.join(sub, f) for f in os.listdir(sub)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            all_files.extend(imgs)

        if all_files:
            print(f"  Using: {root} ({len(all_files)} images across all subfolders)")
            return all_files

    print(f"  ❌ Could not find dataset for {crop_name}")
    print(f"     Tried these paths:")
    for c in candidates:
        print(f"       {c}")
    print(f"     Please check your datasets/ folder structure.")
    return []


def load_and_preprocess(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return arr


def compute_centroid_for_class(extractor, image_paths):
    if len(image_paths) > MAX_IMAGES_PER_CLASS:
        np.random.seed(42)
        image_paths = list(np.random.choice(image_paths, MAX_IMAGES_PER_CLASS, replace=False))

    print(f"  Processing {len(image_paths)} images...")

    embeddings = []
    for path in image_paths:
        try:
            arr = load_and_preprocess(path)
            embeddings.append(arr)
        except Exception as e:
            print(f"    ⚠️  Skipped {os.path.basename(path)}: {e}")

    if not embeddings:
        return None, None, 0

    batch    = np.stack(embeddings)
    vecs     = extractor.predict(batch, verbose=0, batch_size=32)
    centroid = np.mean(vecs, axis=0)

    norms         = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
    self_sims     = norms @ centroid_norm
    min_self_sim  = float(np.percentile(self_sims, 5))

    # SAFETY CLAMP: never let a class's similarity floor go below this
    # absolute minimum. Without this, classes with more diverse training
    # images (like Tomato, which combines multiple photography styles)
    # can end up with an unusually loose floor (e.g. 0.72 vs 0.90+ for
    # other classes) — making it easy for unrelated images (even human
    # photos) to "pass" by matching loosely against that one class.
    ABSOLUTE_SIMILARITY_FLOOR = 0.80
    if min_self_sim < ABSOLUTE_SIMILARITY_FLOOR:
        print(f"    ⚠️  Computed floor ({min_self_sim:.4f}) is below safety minimum — clamping to {ABSOLUTE_SIMILARITY_FLOOR}")
        min_self_sim = ABSOLUTE_SIMILARITY_FLOOR

    return centroid.tolist(), min_self_sim, len(embeddings)


def main():
    print("\n" + "=" * 60)
    print("  Farm Vision — Building Feature-Similarity Centroids")
    print("=" * 60 + "\n")

    extractor = load_feature_extractor()
    if extractor is None:
        return

    centroids_data = {}

    for crop_name in ["Tomato", "Potato", "Cucumber", "Cauliflower"]:
        print(f"\n📁 Crop: {crop_name}")
        image_paths = collect_images_for_crop(crop_name)

        if not image_paths:
            print(f"  ⚠️  Skipping {crop_name} — no images found")
            continue

        centroid, min_sim, n = compute_centroid_for_class(extractor, image_paths)
        if centroid is None:
            print(f"  ⚠️  Skipping {crop_name} — could not compute centroid")
            continue

        centroids_data[crop_name] = {
            "centroid":       centroid,
            "min_similarity": round(min_sim, 4),
            "n_images_used":  n,
        }
        print(f"  ✅ Centroid built — self-similarity floor: {min_sim:.4f}")

    if not centroids_data:
        print("\n❌ No centroids built — check dataset paths above.")
        return

    out_path = os.path.join(MODEL_DIR, "crop_centroids.json")
    with open(out_path, "w") as f:
        json.dump(centroids_data, f, indent=2)

    print("\n" + "=" * 60)
    print(f"✅ Saved: {out_path}")
    print(f"   Classes: {list(centroids_data.keys())}")
    print("\n   Restart your FastAPI server — Gate 4 will activate automatically.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()