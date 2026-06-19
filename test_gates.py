# ==============================================================
# test_gates.py — Farm Vision Local Gate Tester
#
# Tests all 4 validation gates locally WITHOUT starting FastAPI.
# Just put test images in test_images/ folder and run:
#   python test_gates.py
#
# Folder structure:
#   test_images/
#     ├── should_pass/      ← Real clear leaf photos (should get prediction)
#     └── should_reject/    ← Human, mango, blurry, random objects (should be rejected)
# ==============================================================

import os, io, json
import numpy as np
from PIL import Image, ImageFilter

# ── CONFIG (must match main.py) ──────────────────────────────
MODEL_DIR                    = "models"
IMG_SIZE                     = (224, 224)
BLUR_EDGE_VARIANCE_THRESHOLD = 313.8
LEAF_GREEN_RATIO_THRESHOLD   = 0.12
CROP_MARGIN_THRESHOLD        = 0.15
CROP_CONF_THRESHOLD          = 0.85
CROP_CENTROIDS_PATH          = os.path.join(MODEL_DIR, "crop_centroids.json")
SIMILARITY_FALLBACK_MIN      = 0.80

PLANT_KEYWORDS = [
    "banana","corn","cucumber","artichoke","cauliflower","broccoli",
    "cabbage","squash","pepper","mushroom","fig","pineapple",
    "strawberry","lemon","orange","apple","pomegranate","plant",
    "flower","leaf","vegetable","fruit","daisy","sunflower","rose",
    "pot","vase","tree","fern","herb","vine","bamboo","cardoon",
    "zucchini","potato",
]
NON_PLANT_REJECT_KEYWORDS = [
    "person","suit","tie","sweatshirt","cardigan","jersey","shirt",
    "sunglasses","face","groom","bride","military","uniform",
    "cellular","phone","laptop","keyboard","monitor","screen",
    "remote","calculator","car","vehicle","bicycle","motor",
    "dog","cat","bird","fish","snake",
]
# ─────────────────────────────────────────────────────────────

# Colors for terminal output
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

print(f"\n{BOLD}Loading TensorFlow...{RESET}")
try:
    import tensorflow as tf
    TF_OK = True
    print(f"{GREEN}✅ TensorFlow loaded{RESET}")
except ImportError:
    TF_OK = False
    print(f"{YELLOW}⚠️  TensorFlow not found — Gates 3 and 4 will be skipped{RESET}")

# ── LOAD MODELS ──────────────────────────────────────────────
crop_classifier   = None
crop_classes      = {}
sanity_model      = None
feature_extractor = None
crop_centroids    = {}

if TF_OK:
    cc_path = os.path.join(MODEL_DIR, "crop_classifier.h5")
    cc_cls  = os.path.join(MODEL_DIR, "crop_classes.json")
    if os.path.exists(cc_path) and os.path.exists(cc_cls):
        crop_classifier = tf.keras.models.load_model(cc_path, compile=False)
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = crop_classifier(dummy, training=False)
        with open(cc_cls) as f:
            crop_classes = json.load(f)
        print(f"{GREEN}✅ Crop classifier loaded{RESET}")

        # Build feature extractor
        feature_layer = None
        for layer in reversed(crop_classifier.layers):
            if hasattr(layer, 'units') and layer.name != crop_classifier.layers[-1].name:
                feature_layer = layer
                break
        if feature_layer is None:
            feature_layer = crop_classifier.layers[-3]
        try:
            feature_extractor = tf.keras.Model(
                inputs=crop_classifier.input,
                outputs=feature_layer.output,
            )
        except Exception:
            idx = crop_classifier.layers.index(feature_layer)
            feature_extractor = tf.keras.Sequential(crop_classifier.layers[:idx+1])
            feature_extractor(dummy, training=False)
        print(f"{GREEN}✅ Feature extractor built (layer: {feature_layer.name}){RESET}")
    else:
        print(f"{YELLOW}⚠️  crop_classifier.h5 not found — crop check skipped{RESET}")

    # ImageNet sanity model
    try:
        sanity_model = tf.keras.applications.MobileNetV2(weights="imagenet")
        print(f"{GREEN}✅ ImageNet sanity model loaded{RESET}")
    except Exception as e:
        print(f"{YELLOW}⚠️  Sanity model failed: {e}{RESET}")

    # Centroids
    if os.path.exists(CROP_CENTROIDS_PATH):
        with open(CROP_CENTROIDS_PATH) as f:
            crop_centroids = json.load(f)
        print(f"{GREEN}✅ Centroids loaded: {list(crop_centroids.keys())}{RESET}")
    else:
        print(f"{YELLOW}⚠️  crop_centroids.json not found — Gate 4 skipped{RESET}")

# ── GATE FUNCTIONS ────────────────────────────────────────────
def gate1_blur(pil_img):
    gray  = pil_img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    score = float(np.array(edges, dtype=np.float32).var())
    passed = score >= BLUR_EDGE_VARIANCE_THRESHOLD
    return passed, score

def gate2_leaf_coverage(pil_img):
    hsv = pil_img.convert("HSV")
    arr = np.array(hsv)
    h, s, v = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    mask = (h>=28)&(h<=115)&(s>=30)&(v>=20)
    ratio = float(np.mean(mask))
    return ratio >= LEAF_GREEN_RATIO_THRESHOLD, ratio

def gate3_imagenet(pil_img):
    if sanity_model is None:
        return True, [], "SKIPPED"
    img  = pil_img.convert("RGB").resize((224,224))
    arr  = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    arr  = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    preds  = sanity_model.predict(arr, verbose=0)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]
    labels  = [lbl.lower().replace("_"," ") for (_,lbl,_) in decoded]
    for lbl in labels[:3]:
        if any(bad in lbl for bad in NON_PLANT_REJECT_KEYWORDS):
            return False, labels, "NON-PLANT"
    for lbl in labels:
        if any(good in lbl for good in PLANT_KEYWORDS):
            return True, labels, "PLANT"
    return True, labels, "UNCERTAIN-PASS"

def gate4_similarity(pil_img):
    if feature_extractor is None or not crop_centroids:
        return True, None, None, None, "SKIPPED"
    img  = pil_img.convert("RGB").resize(IMG_SIZE)
    arr  = np.expand_dims(np.array(img, dtype=np.float32)/255.0, axis=0)

    # First get softmax prediction to know which centroid to compare against
    if crop_classifier is not None:
        preds      = crop_classifier.predict(arr, verbose=0)
        sorted_idx = np.argsort(preds[0])[::-1]
        max_prob   = float(preds[0][sorted_idx[0]])
        margin     = max_prob - float(preds[0][sorted_idx[1]])
        crop_idx   = str(sorted_idx[0])
        predicted  = crop_classes.get(crop_idx, "Unknown")
    else:
        return True, None, None, None, "SKIPPED"

    embedding = feature_extractor.predict(arr, verbose=0)[0]
    all_sims  = {}
    for name, info in crop_centroids.items():
        c = np.array(info["centroid"], dtype=np.float32)
        sim = float(np.dot(embedding, c) / (np.linalg.norm(embedding)*np.linalg.norm(c)+1e-8))
        all_sims[name] = sim

    best_match  = max(all_sims, key=all_sims.get)
    best_sim    = all_sims[best_match]
    pred_sim    = all_sims.get(predicted, 0)
    floor       = crop_centroids.get(predicted, {}).get("min_similarity", SIMILARITY_FALLBACK_MIN)

    if max_prob < CROP_CONF_THRESHOLD:
        return False, pred_sim, best_match, all_sims, f"LOW_CONF({max_prob*100:.0f}%)"
    if margin < CROP_MARGIN_THRESHOLD:
        return False, pred_sim, best_match, all_sims, f"AMBIGUOUS(margin={margin*100:.0f}pp)"
    if pred_sim < floor:
        return False, pred_sim, best_match, all_sims, f"SIM_TOO_LOW({pred_sim:.3f}<{floor:.3f})"

    return True, pred_sim, best_match, all_sims, f"OK({predicted} {pred_sim:.3f})"

# ── TEST RUNNER ───────────────────────────────────────────────
def test_image(path, expected_pass):
    name = os.path.basename(path)
    try:
        pil_img = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"  {RED}ERROR loading {name}: {e}{RESET}")
        return False

    results = {}

    # Gate 1 — Blur
    g1_pass, blur_score = gate1_blur(pil_img)
    results["G1 Blur"] = (g1_pass, f"score={blur_score:.0f} thresh={BLUR_EDGE_VARIANCE_THRESHOLD}")

    # Gate 2 — Leaf coverage
    g2_pass, coverage = gate2_leaf_coverage(pil_img)
    results["G2 Leaf"] = (g2_pass, f"coverage={coverage*100:.0f}% thresh={LEAF_GREEN_RATIO_THRESHOLD*100:.0f}%")

    # Gate 3 — ImageNet sanity
    g3_pass, labels, reason = gate3_imagenet(pil_img)
    top2 = ", ".join(labels[:2]) if labels else "n/a"
    results["G3 Sanity"] = (g3_pass, f"{reason} top2=[{top2}]")

    # Gate 4 — Feature similarity
    g4_pass, sim, best, all_sims, detail = gate4_similarity(pil_img)
    results["G4 Similarity"] = (g4_pass, detail or "SKIPPED")

    # Overall
    all_pass    = g1_pass and g2_pass and g3_pass and g4_pass
    correct     = (all_pass == expected_pass)
    label_str   = f"{GREEN}PASS{RESET}" if all_pass else f"{RED}REJECT{RESET}"
    expect_str  = "PASS" if expected_pass else "REJECT"
    status_str  = f"{GREEN}✅ CORRECT{RESET}" if correct else f"{RED}❌ WRONG (expected {expect_str}){RESET}"

    print(f"\n  {BOLD}{name}{RESET}")
    print(f"  Overall: {label_str}  {status_str}")
    for gate_name, (gate_pass, info) in results.items():
        icon = f"{GREEN}✅{RESET}" if gate_pass else f"{RED}❌{RESET}"
        print(f"    {icon} {gate_name}: {info}")

    return correct

def run_folder(folder, expected_pass, label):
    if not os.path.exists(folder):
        print(f"\n{YELLOW}⚠️  Folder not found: {folder}{RESET}")
        print(f"   Create it and put images inside, then run again.")
        return 0, 0

    files = [f for f in os.listdir(folder)
             if f.lower().endswith((".jpg",".jpeg",".png",".webp"))]
    if not files:
        print(f"\n{YELLOW}⚠️  No images found in {folder}{RESET}")
        return 0, 0

    print(f"\n{'='*60}")
    print(f"{BOLD}{label}{RESET}  ({len(files)} images — expected: {'PASS' if expected_pass else 'REJECT'})")
    print(f"{'='*60}")

    correct_count = 0
    for fname in sorted(files):
        path    = os.path.join(folder, fname)
        correct = test_image(path, expected_pass)
        if correct:
            correct_count += 1

    return correct_count, len(files)

# ── MAIN ─────────────────────────────────────────────────────
def main():
    print(f"\n{BOLD}{'='*60}")
    print(f"  Farm Vision — Local Gate Test")
    print(f"{'='*60}{RESET}")

    test_root     = "test_images"
    pass_folder   = os.path.join(test_root, "should_pass")
    reject_folder = os.path.join(test_root, "should_reject")

    if not os.path.exists(test_root):
        os.makedirs(pass_folder, exist_ok=True)
        os.makedirs(reject_folder, exist_ok=True)
        print(f"\n{YELLOW}Created test_images/ folder.")
        print(f"Put your images inside:")
        print(f"  test_images/should_pass/    ← clear leaf photos")
        print(f"  test_images/should_reject/  ← human/mango/blurry/random")
        print(f"Then run: python test_gates.py{RESET}")
        return

    p_correct, p_total = run_folder(pass_folder,   True,  "SHOULD PASS (clear leaf photos)")
    r_correct, r_total = run_folder(reject_folder, False, "SHOULD REJECT (human/mango/blurry)")

    total   = p_total + r_total
    correct = p_correct + r_correct

    print(f"\n{'='*60}")
    print(f"{BOLD}  RESULTS SUMMARY{RESET}")
    print(f"{'='*60}")
    if total == 0:
        print(f"{YELLOW}  No images tested. Add images to test_images/ folders.{RESET}")
    else:
        pct = correct/total*100
        color = GREEN if pct >= 80 else (YELLOW if pct >= 60 else RED)
        print(f"  Should Pass:   {p_correct}/{p_total} correct")
        print(f"  Should Reject: {r_correct}/{r_total} correct")
        print(f"  {color}{BOLD}Overall: {correct}/{total} ({pct:.0f}%){RESET}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()