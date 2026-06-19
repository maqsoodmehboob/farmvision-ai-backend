# ==============================================================
# evaluate_models.py — Farm Vision Model Evaluation (Fixed)
# Run: py -3.11 evaluate_models.py
# ==============================================================

import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
try:
    tf.get_logger().setLevel('ERROR')
except:
    pass

MODEL_DIR   = "models"
DATASET_DIR = "datasets"
IMG_SIZE    = (224, 224)

TEST_CONFIG = {
    "Potato": {
        "test_folder": "Testing",
        "folder_name": "Potato",
    },
    "Tomato": {
        "test_folder": "valid",
        "folder_name": "Tomato Diseases in Pakistan",
    },
    "Cucumber": {
        "test_folder": None,
        "folder_name": "Cucumber",
    },
    "Cauliflower": {
        "test_folder": None,
        "folder_name": "Cauliflower",
    },
}

def normalize(name):
    """Normalize class name: lowercase + remove underscores/spaces"""
    return name.lower().replace("_", "").replace(" ", "").replace("-", "")

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

def evaluate_crop(crop_name):
    print(f"\n{'='*55}")
    print(f"  Evaluating: {crop_name}")
    print(f"{'='*55}")

    config      = TEST_CONFIG[crop_name]
    folder_name = config["folder_name"]
    test_folder = config["test_folder"]

    if not test_folder:
        print(f"  ⚠️  No separate test set — skipping {crop_name}")
        return None

    # Load model
    model_path = os.path.join(MODEL_DIR, f"{crop_name}_model.h5")
    class_path = os.path.join(MODEL_DIR, f"{crop_name}_classes.json")

    if not os.path.exists(model_path):
        print(f"  ❌ Model not found: {model_path}")
        return None

    model = tf.keras.models.load_model(model_path, compile=False)
    with open(class_path) as f:
        idx_to_class = json.load(f)

    print(f"  Classes in model: {list(idx_to_class.values())}")

    # Test path
    test_path = os.path.join(DATASET_DIR, folder_name, test_folder)
    if not os.path.exists(test_path):
        print(f"  ❌ Test folder not found: {test_path}")
        return None

    correct       = 0
    total         = 0
    class_results = {}

    for folder_class in sorted(os.listdir(test_path)):
        class_dir = os.path.join(test_path, folder_class)
        if not os.path.isdir(class_dir):
            continue

        # Find matching model class name
        matched_model_class = None
        for idx, model_class in idx_to_class.items():
            if normalize(folder_class) == normalize(model_class):
                matched_model_class = model_class
                break

        if not matched_model_class:
            # Try partial match
            for idx, model_class in idx_to_class.items():
                if normalize(folder_class) in normalize(model_class) or \
                   normalize(model_class) in normalize(folder_class):
                    matched_model_class = model_class
                    break

        if not matched_model_class:
            print(f"  ⚠️  No match for folder '{folder_class}' in model classes")
            continue

        print(f"  Folder '{folder_class}' → Model class '{matched_model_class}'")

        class_correct = 0
        class_total   = 0

        for img_file in os.listdir(class_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(class_dir, img_file)
            try:
                img_array  = preprocess_image(img_path)
                preds      = model.predict(img_array, verbose=0)
                pred_idx   = str(np.argmax(preds))
                pred_class = idx_to_class.get(pred_idx, "Unknown")

                if normalize(pred_class) == normalize(matched_model_class):
                    class_correct += 1
                    correct       += 1

                class_total += 1
                total       += 1

            except Exception as e:
                continue

        if class_total > 0:
            class_acc = (class_correct / class_total) * 100
            class_results[folder_class] = class_acc
            status = "✅" if class_acc >= 80 else "⚠️ " if class_acc >= 50 else "❌"
            print(f"  {status} {folder_class:25} → {class_correct}/{class_total} = {class_acc:.1f}%")

    if total > 0:
        overall = (correct / total) * 100
        print(f"\n  🎯 {crop_name} Test Accuracy: {correct}/{total} = {overall:.2f}%")
        return overall

    return None

def main():
    print("\n" + "="*55)
    print("  🌿 Farm Vision — Model Evaluation on Test Data")
    print("="*55)

    results = {}
    for crop in ["Potato", "Tomato", "Cucumber", "Cauliflower"]:
        acc = evaluate_crop(crop)
        if acc is not None:
            results[crop] = acc

    print("\n" + "="*55)
    print("  📊 FINAL TEST RESULTS")
    print("="*55)

    if results:
        for crop, acc in results.items():
            status = "✅" if acc >= 85 else "⚠️ " if acc >= 70 else "❌"
            print(f"  {status} {crop:15} → {acc:.2f}%")

        avg = sum(results.values()) / len(results)
        print(f"\n  🎯 Overall Test Accuracy: {avg:.2f}%")

        # Also show validation accuracy
        print(f"\n  📋 Validation Accuracy (from training):")
        val_acc = {
            "Potato":      92.79,
            "Tomato":      93.63,
            "Cucumber":    82.10,
            "Cauliflower": 100.0,
        }
        for crop, acc in val_acc.items():
            print(f"     {crop:15} → {acc:.2f}%")
        avg_val = sum(val_acc.values()) / len(val_acc)
        print(f"\n  🎯 Overall Validation Accuracy: {avg_val:.2f}%")

    print("="*55)

if __name__ == "__main__":
    main()