# ==============================================================
# main.py — Farm Vision FastAPI Backend v3.3
# Features:
#   - CNN Disease Detection (4 crops)
#   - Crop Classifier with Unknown/Other crop handling
#   - Confidence threshold check
#   - Weather API
#   - Voice Query
#   - Multilingual (English, Urdu, Roman Urdu)
# Run: py -3.11 -m uvicorn main:app --host 0.0.0.0 --port $PORT
# ==============================================================

import sys
# Force UTF-8 output on Windows — without this, printing Urdu/Arabic
# text to the Windows console raises UnicodeEncodeError and crashes
# the request (Windows terminals default to cp1252/cp437, which
# cannot encode Urdu script characters).
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass  # older Python versions — safe to ignore

import os, json, io, tempfile
import numpy as np
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFilter

# ---- TensorFlow ----
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    import tensorflow as tf
    try:
        tf.get_logger().setLevel('ERROR')
    except:
        pass
    TF_AVAILABLE = True
    print("✅ TensorFlow loaded")
except Exception as e:
    TF_AVAILABLE = False
    print(f"⚠️  TensorFlow: {e}")

# ---- Speech Recognition ----
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
    print("✅ Speech Recognition loaded")
except:
    SR_AVAILABLE = False
    print("⚠️  SpeechRecognition not available")

WEATHER_KEY = os.getenv("WEATHER_API_KEY", "")

app = FastAPI(title="Farm Vision API", version="3.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL_DIR = "models"
IMG_SIZE  = (224, 224)

# Confidence thresholds
# Higher crop threshold = stricter rejection of non-leaf images
CROP_CONF_THRESHOLD    = 0.80   # Lowered from 0.85 — test data showed 0.85 was rejecting legitimate
                                  # images (82% confidence) while reject-side failures were all <60%,
                                  # so 0.80 catches genuine non-crops without false-rejecting real ones.
                                  # Other 3 gates (leaf coverage, ImageNet, similarity) provide backup.
DISEASE_CONF_THRESHOLD = 0.65   # Disease model min confidence

# ============================================================
# PRE-CLASSIFICATION VALIDATION GATE — Thresholds
# These run BEFORE the crop classifier, no retraining needed.
# Goal: catch human photos, blurry shots, cropped/partial leaves,
# and random objects before they ever reach the 4-way classifier.
# ============================================================
BLUR_EDGE_VARIANCE_THRESHOLD = 313.8  # Calibrated from real test photos (sharp min: 364.7, blurry max: 262.9)
LEAF_GREEN_RATIO_THRESHOLD   = 0.12   # Below this = not enough green/plant content
HIGH_CONFIDENCE_LEAF_RATIO   = 0.35   # Above this, skip Gate 3 (ImageNet) — speed optimization.
                                        # Rationale: a clearly high green/plant-tissue ratio is
                                        # already strong evidence of a real leaf photo. Gate 4
                                        # (feature similarity) still runs regardless and provides
                                        # a deeper semantic check, so this skip is not "unguarded."
CROP_MARGIN_THRESHOLD        = 0.15   # top1 - top2 probability gap; below this = ambiguous
SANITY_MIN_SCORE             = 0.10   # Minimum ImageNet top-1 score for ANY plant-keyword match

# ============================================================
# GATE 4 — FEATURE SIMILARITY CHECK (centroid-based OOD rejection)
# This is the strongest defense against mango/unknown-crop -> Tomato
# misclassification, because it doesn't rely on softmax probability
# at all. It compares the image's internal feature fingerprint
# against the AVERAGE fingerprint of each known crop. If an image
# (e.g. mango) doesn't genuinely resemble ANY known crop in feature
# space, it gets rejected even if softmax was "confident."
#
# Requires running build_centroids.py once (uses your EXISTING
# trained crop_classifier.h5 — no retraining of the classifier
# itself needed). If models/crop_centroids.json doesn't exist yet,
# this gate silently fails open (skips) so nothing breaks.
# ============================================================
CROP_CENTROIDS_PATH      = os.path.join("models", "crop_centroids.json")
SIMILARITY_FALLBACK_MIN  = 0.80   # used if a class has no calibrated min_similarity

# Coarse keyword list used to check if ImageNet (general 1000-class model)
# thinks the image contains *anything* plant/vegetable/fruit-like.
# This is intentionally broad — it is a sanity gate, not a precise classifier.
PLANT_KEYWORDS = [
    "banana", "corn", "cucumber", "artichoke", "cauliflower", "broccoli",
    "cabbage", "squash", "pepper", "mushroom", "fig", "pineapple",
    "strawberry", "lemon", "orange", "apple", "pomegranate", "plant",
    "flower", "leaf", "vegetable", "fruit", "daisy", "sunflower", "rose",
    "pot", "vase", "tree", "fern", "herb", "vine", "bamboo", "cardoon",
    "zucchini", "potato",
]

# Curated reject list — if ANY of these appear in the top-3 ImageNet
# predictions, we reject immediately regardless of plant keyword match.
# These are common "this is clearly NOT a leaf" categories.
NON_PLANT_REJECT_KEYWORDS = [
    "person", "suit", "tie", "sweatshirt", "cardigan", "jersey", "shirt",
    "sunglasses", "face", "groom", "bride", "military", "uniform",
    "cellular", "phone", "laptop", "keyboard", "monitor", "screen",
    "remote", "calculator", "car", "vehicle", "bicycle", "motor",
    "dog", "cat", "bird", "fish", "snake",
]

# ============================================================
# VALIDATION GATE — Multilingual rejection messages
# ============================================================
INVALID_IMAGE_MESSAGES = {
    "blurry": {
        "en":    "Image is too blurry or unclear to analyze. Please retake the photo in good lighting, holding the camera steady.",
        "ur":    "تصویر بہت دھندلی یا غیر واضح ہے۔ براہ کرم اچھی روشنی میں اور کیمرہ ساکت رکھ کر دوبارہ تصویر لیں۔",
        "roman": "Tasveer bohat dhundli ya ghair wazeh hai. Meherbani karke achi roshni mein aur camera steady rakh kar dobara tasveer lein.",
    },
    "no_leaf": {
        "en":    "Could not detect a leaf in this image. Please make sure the leaf fills most of the frame.",
        "ur":    "اس تصویر میں پتا نہیں مل سکا۔ براہ کرم یقینی بنائیں کہ پتا تصویر کے بڑے حصے میں نظر آئے۔",
        "roman": "Is tasveer mein patta nahi mil saka. Meherbani karke yaqeeni banayen ke patta tasveer ke bare hisse mein nazar aaye.",
    },
    "not_a_plant": {
        "en":    "This does not appear to be a plant leaf image. Please upload a photo of a crop leaf only.",
        "ur":    "یہ پودے کے پتے کی تصویر معلوم نہیں ہوتی۔ براہ کرم صرف فصل کے پتے کی تصویر اپ لوڈ کریں۔",
        "roman": "Yeh paudy ke patte ki tasveer maloom nahi hoti. Meherbani karke sirf fasal ke patte ki tasveer upload karein.",
    },
    "ambiguous": {
        "en":    "The crop type could not be determined with confidence — the image may show more than one crop or an unclear angle. Please retake with a single leaf clearly in frame.",
        "ur":    "فصل کی قسم اعتماد سے معلوم نہیں ہو سکی — تصویر میں ایک سے زیادہ فصل یا غیر واضح زاویہ ہو سکتا ہے۔ براہ کرم ایک پتے کی واضح تصویر دوبارہ لیں۔",
        "roman": "Fasal ki qisam confidence se maloom nahi ho saki — tasveer mein ek se zyada fasal ya ghair wazeh zaviya ho sakta hai. Meherbani karke ek patte ki wazeh tasveer dobara lein.",
    },
}

# ============================================================
# DISEASE INFO — 3 Languages
# ============================================================
DISEASE_INFO = {
    "en": {
        "Early Blight": {
            "cause":       "Alternaria solani fungus. Warm (24-29°C) humid conditions.",
            "prevention":  "Crop rotation, remove infected leaves, copper-based fungicide.",
            "treatment":   "Mancozeb or Chlorothalonil spray every 7 days.",
            "weather_note":"High humidity accelerates this disease.",
            "prediction":  "40% more spread in 2 weeks if weather stays humid."
        },
        "Late Blight": {
            "cause":       "Phytophthora infestans. Cool (10-20°C), wet weather.",
            "prevention":  "Resistant varieties, avoid overhead irrigation.",
            "treatment":   "Metalaxyl-M + Mancozeb. Destroy infected plants.",
            "weather_note":"Rain and cool nights are ideal for late blight.",
            "prediction":  "High risk during monsoon. Preventive spray needed."
        },
        "Leaf Mold": {
            "cause":       "Passalora fulva fungus. High humidity environments.",
            "prevention":  "Increase air circulation, reduce humidity.",
            "treatment":   "Copper sulfate spray. Remove affected leaves promptly.",
            "weather_note":"21C+ with 85%+ humidity is high risk.",
            "prediction":  "Reduce greenhouse humidity to control spread."
        },
        "Powdery Mildew": {
            "cause":       "Erysiphe fungus. Dry and warm conditions.",
            "prevention":  "Resistant varieties, proper spacing.",
            "treatment":   "Sulfur-based fungicide or Neem oil spray.",
            "weather_note":"Dry weather with moderate temperature increases risk.",
            "prediction":  "High risk in summer. Regular monitoring needed."
        },
        "Mosaic Virus": {
            "cause":       "Tobacco Mosaic Virus. Spreads via insects and tools.",
            "prevention":  "Resistant varieties, aphid control, sanitize tools.",
            "treatment":   "No cure. Remove and destroy infected plants.",
            "weather_note":"Warm weather increases insect vector activity.",
            "prediction":  "Monitor insects, use preventive insecticide."
        },
        "Bacterial Spot": {
            "cause":       "Xanthomonas bacteria. Spreads through rain and tools.",
            "prevention":  "Certified seed, copper bactericide, sanitize tools.",
            "treatment":   "Copper-based spray. Burn infected plants.",
            "weather_note":"Warm wet weather increases bacterial spread.",
            "prediction":  "Monitor closely during rainy season."
        },
        "Downy Mildew": {
            "cause":       "Peronospora fungus. Cool moist conditions.",
            "prevention":  "Good drainage, proper spacing.",
            "treatment":   "Metalaxyl or Fosetyl-Al fungicide.",
            "weather_note":"Cool nights and humid days are perfect conditions.",
            "prediction":  "More attention needed in winter."
        },
        "Anthracnose": {
            "cause":       "Colletotrichum fungus. Warm wet conditions.",
            "prevention":  "Crop rotation, remove infected debris.",
            "treatment":   "Mancozeb or Copper fungicide.",
            "weather_note":"Rain dramatically increases risk.",
            "prediction":  "High risk post-monsoon. Preventive fungicide needed."
        },
        "Black Rot": {
            "cause":       "Xanthomonas campestris bacteria. Warm wet conditions.",
            "prevention":  "Crop rotation, certified seed.",
            "treatment":   "Copper bactericide. Remove severely infected plants.",
            "weather_note":"After rain with moderate temperature — high risk.",
            "prediction":  "Monitor closely after monsoon."
        },
        "Healthy": {
            "cause":       "No disease detected.",
            "prevention":  "Continue good agricultural practices.",
            "treatment":   "No treatment required.",
            "weather_note":"Monitor regularly during weather changes.",
            "prediction":  "Low disease risk. Maintain current practices."
        },
        "Unknown": {
            "cause":       "Could not identify the crop or disease.",
            "prevention":  "Please upload a clear image of a leaf.",
            "treatment":   "N/A",
            "weather_note":"N/A",
            "prediction":  "Please retake the photo in good lighting."
        },
    },

    "ur": {
        "Early Blight": {
            "cause":       "الٹرنیریا سولانی فنگس۔ گرم اور نم موسم میں پھیلتی ہے۔",
            "prevention":  "فصل کی تبدیلی، متاثرہ پتے ہٹائیں، کاپر فنگی سائیڈ۔",
            "treatment":   "مانکوزیب یا کلوروتھالونیل ہر 7 دن میں اسپرے کریں۔",
            "weather_note":"زیادہ نمی اور معتدل درجہ حرارت بیماری بڑھاتا ہے۔",
            "prediction":  "2 ہفتوں میں 40% زیادہ پھیلنے کا خطرہ۔"
        },
        "Late Blight": {
            "cause":       "فائیٹوفتھورا فنگس۔ ٹھنڈے اور بارش والے موسم میں۔",
            "prevention":  "مزاحم اقسام استعمال کریں، اوپری آبپاشی سے بچیں۔",
            "treatment":   "میٹالیکسل + مانکوزیب۔ متاثرہ پودے ختم کریں۔",
            "weather_note":"بارش اور ٹھنڈی راتیں بیماری کے لیے موزوں ہیں۔",
            "prediction":  "مون سون میں زیادہ خطرہ — فوری اسپرے کریں۔"
        },
        "Leaf Mold": {
            "cause":       "پاسالورا فلوا فنگس۔ گرین ہاؤس اور زیادہ نمی میں۔",
            "prevention":  "ہوا کی گردش بڑھائیں، نمی کم کریں۔",
            "treatment":   "کاپر سلفیٹ اسپرے، متاثرہ پتے فوری ہٹائیں۔",
            "weather_note":"21C سے زیادہ اور 85% نمی — زیادہ خطرہ۔",
            "prediction":  "گرین ہاؤس کی نمی کم کریں۔"
        },
        "Powdery Mildew": {
            "cause":       "ایریسیفی فنگس۔ خشک اور گرم موسم میں۔",
            "prevention":  "مزاحم اقسام، مناسب فاصلہ۔",
            "treatment":   "سلفر فنگی سائیڈ یا نیم آئل اسپرے۔",
            "weather_note":"خشک موسم خطرہ بڑھاتا ہے۔",
            "prediction":  "گرمیوں میں زیادہ خطرہ — نگرانی ضروری۔"
        },
        "Mosaic Virus": {
            "cause":       "ٹوبیکو موزیک وائرس۔ کیڑوں اور آلودہ اوزاروں سے پھیلتا ہے۔",
            "prevention":  "مزاحم اقسام، افیڈز کنٹرول، اوزار صاف رکھیں۔",
            "treatment":   "براہ راست علاج نہیں۔ متاثرہ پودے ہٹائیں۔",
            "weather_note":"گرم موسم میں کیڑوں کی سرگرمی بڑھتی ہے۔",
            "prediction":  "کیڑوں کی آبادی نگرانی کریں۔"
        },
        "Bacterial Spot": {
            "cause":       "زینتھوموناس بیکٹیریا۔ بارش اور آلودہ اوزاروں سے پھیلتا ہے۔",
            "prevention":  "تصدیق شدہ بیج، کاپر بیکٹیری سائیڈ، اوزار صاف کریں۔",
            "treatment":   "کاپر اسپرے۔ متاثرہ پودے جلائیں۔",
            "weather_note":"گرم اور نم موسم بیکٹیریا پھیلاتا ہے۔",
            "prediction":  "بارش کے موسم میں نگرانی کریں۔"
        },
        "Downy Mildew": {
            "cause":       "پیرونوسپورا فنگس۔ ٹھنڈا اور نم موسم۔",
            "prevention":  "اچھی نکاسی آب، مناسب فاصلہ۔",
            "treatment":   "میٹالیکسل یا فوسیٹل فنگی سائیڈ اسپرے۔",
            "weather_note":"ٹھنڈی راتیں اور نم دن بہترین حالات ہیں۔",
            "prediction":  "سردیوں میں زیادہ توجہ دیں۔"
        },
        "Anthracnose": {
            "cause":       "کولیٹوٹریکم فنگس۔ گرم اور نم حالات میں پھیلتی ہے۔",
            "prevention":  "فصل کی تبدیلی، متاثرہ ملبہ ہٹائیں۔",
            "treatment":   "مانکوزیب یا کاپر فنگی سائیڈ۔",
            "weather_note":"بارش اور نمی خطرہ بڑھاتی ہے۔",
            "prediction":  "مون سون کے بعد احتیاطی فنگی سائیڈ لگائیں۔"
        },
        "Black Rot": {
            "cause":       "زینتھوموناس بیکٹیریا۔ گرم اور نم موسم۔",
            "prevention":  "فصل کی تبدیلی، تصدیق شدہ بیج۔",
            "treatment":   "کاپر بیکٹیری سائیڈ۔ شدید متاثرہ پودے ہٹائیں۔",
            "weather_note":"بارش کے بعد معتدل درجہ حرارت میں خطرہ۔",
            "prediction":  "مون سون کے بعد قریب سے نگرانی کریں۔"
        },
        "Healthy": {
            "cause":       "کوئی بیماری نہیں ملی۔",
            "prevention":  "اچھی زرعی مشق جاری رکھیں۔",
            "treatment":   "کوئی علاج ضروری نہیں۔",
            "weather_note":"موسمی تبدیلیوں میں باقاعدگی سے چیک کریں۔",
            "prediction":  "ابھی بیماری کا خطرہ کم ہے۔"
        },
        "Unknown": {
            "cause":       "فصل یا بیماری کی شناخت نہیں ہو سکی۔",
            "prevention":  "براہ کرم پتے کی واضح تصویر اپ لوڈ کریں۔",
            "treatment":   "لاگو نہیں",
            "weather_note":"لاگو نہیں",
            "prediction":  "اچھی روشنی میں دوبارہ تصویر لیں۔"
        },
    },

    "roman": {
        "Early Blight": {
            "cause":       "Alternaria solani fungus. Garm aur nami wale mausam mein phelta hai.",
            "prevention":  "Fasal badlein, mutasira patte hatayein, copper fungicide use karein.",
            "treatment":   "Mancozeb ya Chlorothalonil har 7 din mein spray karein.",
            "weather_note":"Zyada nami aur moderate temperature bimari tez karta hai.",
            "prediction":  "2 hafton mein 40% zyada phelne ka khatra."
        },
        "Late Blight": {
            "cause":       "Phytophthora fungus. Thanda aur barshat wale mausam mein.",
            "prevention":  "Muzahim aqsam use karein, upar se aabpashi se bachein.",
            "treatment":   "Metalaxyl + Mancozeb. Mutasira podey khatam karein.",
            "weather_note":"Barish aur thandi raatein bimari ke liye moazoon hain.",
            "prediction":  "Monsoon mein zyada khatra — fori spray karein."
        },
        "Leaf Mold": {
            "cause":       "Passalora fulva fungus. Greenhouse aur zyada nami mein.",
            "prevention":  "Hawa ki gardish barhaein, nami kam karein.",
            "treatment":   "Copper sulfate spray, mutasira patte fori hatayein.",
            "weather_note":"21C se upar aur 85% nami — zyada khatra.",
            "prediction":  "Greenhouse ki nami kam karein."
        },
        "Powdery Mildew": {
            "cause":       "Erysiphe fungus. Khushk aur garm mausam mein.",
            "prevention":  "Muzahim aqsam, munasib fasla.",
            "treatment":   "Sulfur fungicide ya Neem oil spray karein.",
            "weather_note":"Khushk mausam khatra barhata hai.",
            "prediction":  "Garmiyon mein zyada khatra — nighrani zaruri."
        },
        "Mosaic Virus": {
            "cause":       "Tobacco Mosaic Virus. Keeron aur aaloodah auzaar se phelta hai.",
            "prevention":  "Muzahim aqsam, aphids control, auzaar saaf rakhein.",
            "treatment":   "Seedha ilaj nahi. Mutasira podey hatayein.",
            "weather_note":"Garm mausam mein keeron ki sargarmi barhti hai.",
            "prediction":  "Keeron ki aabadi nighrani karein."
        },
        "Bacterial Spot": {
            "cause":       "Xanthomonas bacteria. Barish aur aaloodah auzaar se phelta hai.",
            "prevention":  "Tasdiq shuda beej, copper bactericide, auzaar saaf karein.",
            "treatment":   "Copper spray. Mutasira podey jalayein.",
            "weather_note":"Garm aur nami mausam bacteria pheilata hai.",
            "prediction":  "Barish ke mausam mein nighrani karein."
        },
        "Downy Mildew": {
            "cause":       "Peronospora fungus. Thanda aur nami wala mausam.",
            "prevention":  "Achhi nikaasi aab, munasib fasla.",
            "treatment":   "Metalaxyl ya Fosetyl fungicide spray.",
            "weather_note":"Thandi raatein aur nami din behtareen halaat hain.",
            "prediction":  "Sardiyon mein zyada tawajjuh dein."
        },
        "Anthracnose": {
            "cause":       "Colletotrichum fungus. Garm aur nami halaat mein phelta hai.",
            "prevention":  "Fasal badlein, mutasira malba hatayein.",
            "treatment":   "Mancozeb ya Copper fungicide.",
            "weather_note":"Barish aur nami khatra barhata hai.",
            "prediction":  "Monsoon ke baad ihtiyati fungicide lagayein."
        },
        "Black Rot": {
            "cause":       "Xanthomonas bacteria. Garm aur nami mausam.",
            "prevention":  "Fasal badlein, tasdiq shuda beej.",
            "treatment":   "Copper bactericide. Shadeed mutasira podey hatayein.",
            "weather_note":"Barish ke baad moderate temperature mein khatra.",
            "prediction":  "Monsoon ke baad qareeb se nighrani karein."
        },
        "Healthy": {
            "cause":       "Koi bimari nahi mili.",
            "prevention":  "Achhi ziraati mashq jaari rakhein.",
            "treatment":   "Koi ilaj zaruri nahi.",
            "weather_note":"Mausami tabdiliyon mein baqaida check karte rahein.",
            "prediction":  "Abhi bimari ka khatra kam hai."
        },
        "Unknown": {
            "cause":       "Fasal ya bimari ki shanakht nahi ho saki.",
            "prevention":  "Meherbani karke patte ki wazeh tasveer upload karein.",
            "treatment":   "Laagoo nahi",
            "weather_note":"Laagoo nahi",
            "prediction":  "Achhi roshni mein dobara tasveer lein."
        },
    },
}

def get_disease_info(disease, language="en"):
    lang_data = DISEASE_INFO.get(language, DISEASE_INFO["en"])
    en_data   = DISEASE_INFO["en"]
    return lang_data.get(disease) or en_data.get(disease) or en_data["Unknown"]

# ============================================================
# MODEL LOADING
# ============================================================
loaded_models   = {}
loaded_classes  = {}
crop_classifier = None
crop_classes    = {}
sanity_model    = None   # Stock ImageNet MobileNetV2 — used only for the pre-classification sanity gate
feature_extractor = None # Sub-model of crop_classifier — outputs embedding instead of softmax
crop_combined_model = None  # Single model outputting [embedding, softmax] in ONE forward pass (speed optimization)
crop_centroids     = {}  # {class_name: {"centroid": [...], "min_similarity": float}}

def load_models():
    global crop_classifier, crop_classes, feature_extractor, crop_centroids, crop_combined_model

    if not TF_AVAILABLE:
        print("⚠️  TensorFlow not available — skipping model load")
        return

    for crop in ["Potato", "Tomato", "Cucumber", "Cauliflower"]:
        mp = os.path.join(MODEL_DIR, f"{crop}_model.h5")
        cp = os.path.join(MODEL_DIR, f"{crop}_classes.json")
        if os.path.exists(mp) and os.path.exists(cp):
            try:
                loaded_models[crop]  = tf.keras.models.load_model(mp, compile=False)
                with open(cp) as f:
                    loaded_classes[crop] = json.load(f)
                print(f"✅ {crop} model loaded")
            except Exception as e:
                print(f"❌ {crop} failed: {e}")
        else:
            print(f"⚠️  {crop} not found: {mp}")

    cc_path = os.path.join(MODEL_DIR, "crop_classifier.h5")
    cc_cls  = os.path.join(MODEL_DIR, "crop_classes.json")
    if os.path.exists(cc_path) and os.path.exists(cc_cls):
        try:
            crop_classifier = tf.keras.models.load_model(cc_path, compile=False)
            with open(cc_cls) as f:
                crop_classes = json.load(f)
            print(f"✅ Crop classifier loaded — classes: {list(crop_classes.values())}")

            # Build the feature-extractor sub-model.
            # Sequential models need a dummy forward pass before model.input
            # is accessible — call it once with a zeros tensor to build the graph.
            try:
                dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
                _ = crop_classifier(dummy, training=False)

                # Find best feature layer: last Dense before the output Dense
                feature_layer = None
                feature_layer_idx = None
                for idx_, layer in enumerate(crop_classifier.layers):
                    if hasattr(layer, 'units') and layer.name != crop_classifier.layers[-1].name:
                        feature_layer = layer
                        feature_layer_idx = idx_
                if feature_layer is None:
                    feature_layer_idx = len(crop_classifier.layers) - 3
                    feature_layer = crop_classifier.layers[feature_layer_idx]

                try:
                    feature_extractor = tf.keras.Model(
                        inputs=crop_classifier.input,
                        outputs=feature_layer.output,
                    )
                    # Functional model worked — combined dual-output is easy
                    crop_combined_model = tf.keras.Model(
                        inputs=crop_classifier.input,
                        outputs=[feature_layer.output, crop_classifier.output],
                    )
                except Exception:
                    # Sequential-loaded-from-h5 quirk: .input/.output properties
                    # aren't reliably accessible on the loaded model OR on a
                    # Sequential() wrapper built from its layers. Sidestep this
                    # entirely by manually building a fresh functional graph:
                    # feed a brand-new symbolic Input through each original
                    # layer in order. This works regardless of how the
                    # original model was saved/loaded.
                    try:
                        fresh_input = tf.keras.Input(shape=(224, 224, 3))
                        x = fresh_input
                        for layer in crop_classifier.layers[:feature_layer_idx+1]:
                            x = layer(x)
                        embedding_tensor = x
                        for layer in crop_classifier.layers[feature_layer_idx+1:]:
                            x = layer(x)
                        softmax_tensor = x

                        crop_combined_model = tf.keras.Model(
                            inputs=fresh_input,
                            outputs=[embedding_tensor, softmax_tensor],
                        )
                        feature_extractor = tf.keras.Model(
                            inputs=fresh_input,
                            outputs=embedding_tensor,
                        )
                        print(f"✅ Combined dual-output model built via fresh functional graph (saves 1 forward pass per request)")
                    except Exception as e2:
                        crop_combined_model = None
                        # Last-resort fallback: feature_extractor only, as a
                        # plain Sequential (works fine for .predict(), even
                        # though .output/.input introspection doesn't work on it)
                        feature_extractor = tf.keras.Sequential(crop_classifier.layers[:feature_layer_idx+1])
                        feature_extractor(dummy, training=False)
                        print(f"⚠️  Combined model unavailable, falling back to 2 separate predicts: {e2}")

                print(f"✅ Feature extractor built (layer: {feature_layer.name})")
            except Exception as e:
                feature_extractor = None
                crop_combined_model = None
                print(f"⚠️  Could not build feature extractor: {e}")

        except Exception as e:
            print(f"⚠️  Crop classifier failed: {e}")

    # Load pre-computed centroids if they exist (built by build_centroids.py)
    if os.path.exists(CROP_CENTROIDS_PATH):
        try:
            with open(CROP_CENTROIDS_PATH) as f:
                crop_centroids = json.load(f)
            print(f"✅ Feature-similarity centroids loaded: {list(crop_centroids.keys())}")
        except Exception as e:
            crop_centroids = {}
            print(f"⚠️  Could not load centroids: {e}")
    else:
        print(f"ℹ️  No centroids file found ({CROP_CENTROIDS_PATH}) — Gate 4 will be skipped.")
        print(f"   Run build_centroids.py once to activate it.")

def load_sanity_model():
    """
    Loads the stock ImageNet-pretrained MobileNetV2 (NOT our fine-tuned one).
    This is a completely separate model used only to sanity-check that the
    uploaded image plausibly contains a plant/vegetable/leaf before we ever
    run our own crop classifier. No retraining required — this is the exact
    same MobileNetV2 architecture downloaded with its original 1000-class
    ImageNet weights.
    """
    global sanity_model
    if not TF_AVAILABLE:
        return
    try:
        sanity_model = tf.keras.applications.MobileNetV2(weights="imagenet")
        print("✅ ImageNet sanity-check model loaded")
    except Exception as e:
        sanity_model = None
        print(f"⚠️  Sanity model failed to load (gate will be skipped): {e}")

# ============================================================
# IMAGE PROCESSING
# ============================================================
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

# ============================================================
# PRE-CLASSIFICATION VALIDATION GATE
# Runs BEFORE the crop classifier. Catches blurry/cropped images,
# non-leaf objects, and human photos — none of this requires
# retraining any of our own models.
# ============================================================

def is_blurry(pil_img):
    """
    Blur/quality check using PIL's edge filter as a Laplacian-variance
    proxy (no OpenCV dependency needed). Sharp images have high-variance
    edges; blurry images have flat, low-variance edges.
    Returns (is_blurry: bool, score: float)
    """
    try:
        gray  = pil_img.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        arr   = np.array(edges, dtype=np.float32)
        score = float(arr.var())
        return score < BLUR_EDGE_VARIANCE_THRESHOLD, score
    except Exception as e:
        print(f"  ⚠️  Blur check error: {e}")
        return False, -1.0  # fail open — don't block on a broken check

def leaf_coverage_ratio(pil_img):
    """
    HSV-based green/plant-tissue pixel ratio.

    NOTE: An earlier version of this function added a separate RGB-based
    skin-detection rule, but testing showed it incorrectly flagged
    diseased yellow-brown leaf tissue as "skin" too (both have R>G>B
    ordering), causing genuine diseased leaf photos to fail this gate.
    That rule was removed. Hue-based separation alone is more reliable:
    skin tones cluster around hue 15-25 (PIL byte scale), while even
    yellowing/diseased leaf tissue sits higher around hue 30-50+.
    The lower bound below is tuned to this gap.

    Human-photo rejection is primarily handled by Gate 3 (ImageNet
    sanity check) and Gate 4 (feature similarity) instead — this gate
    focuses on genuinely empty/non-plant scenes (walls, sky, etc).

    Returns (ratio: float)
    """
    try:
        hsv = pil_img.convert("HSV")
        arr = np.array(hsv)
        h, s, v = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        # Plant-tissue hue band — starts above typical skin-tone hue range
        plant_mask = (h >= 28) & (h <= 115) & (s >= 30) & (v >= 20)
        ratio = float(np.mean(plant_mask))
        return ratio
    except Exception as e:
        print(f"  ⚠️  Leaf coverage check error: {e}")
        return 1.0  # fail open

def imagenet_sanity_check(pil_img):
    """
    Runs the stock ImageNet MobileNetV2 and checks whether the top
    predictions plausibly correspond to a plant/vegetable/leaf, or
    clearly correspond to something else (person, electronics, animal).
    Returns (passes: bool, top_labels: list[str])
    """
    if sanity_model is None:
        return True, []  # gate unavailable — fail open, don't block users

    try:
        img      = pil_img.convert("RGB").resize((224, 224))
        arr      = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
        arr      = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        preds    = sanity_model.predict(arr, verbose=0)
        decoded  = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]
        labels   = [label.lower().replace("_", " ") for (_, label, _) in decoded]

        print(f"  ImageNet sanity top-5: {labels}")

        # Hard reject — top-3 clearly says "not a plant"
        for label in labels[:3]:
            if any(bad in label for bad in NON_PLANT_REJECT_KEYWORDS):
                return False, labels

        # Soft accept — at least one of top-5 looks plant-like
        for label in labels:
            if any(good in label for good in PLANT_KEYWORDS):
                return True, labels

        # Neither clearly plant nor clearly non-plant — fail open
        # (avoids rejecting genuine leaf photos just because ImageNet's
        # 1000 classes don't include an exact "tomato leaf" category)
        return True, labels

    except Exception as e:
        print(f"  ⚠️  ImageNet sanity check error: {e}")
        return True, []  # fail open

def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

def feature_similarity_check(img_array, predicted_crop, precomputed_embedding=None):
    """
    Gate 4 — compares this image's feature embedding against the
    pre-computed centroid for the crop the softmax classifier picked.
    If it doesn't genuinely resemble that crop's "fingerprint," reject
    it — this catches OOD images (e.g. mango) that softmax forced into
    a class it was confident about for the wrong reasons.

    Fails open (returns pass=True) if centroids or feature_extractor
    aren't available yet, so this never blocks users before
    build_centroids.py has been run.

    SPEED: pass precomputed_embedding (from the combined dual-output
    model) to avoid a second redundant forward pass through the same
    backbone layers. Falls back to running feature_extractor.predict()
    only if no precomputed embedding was supplied.

    Returns (passes: bool, similarity: float|None, best_match: str|None)
    """
    if not crop_centroids:
        return True, None, None

    if predicted_crop not in crop_centroids:
        return True, None, None  # no centroid for this class yet — fail open

    try:
        if precomputed_embedding is not None:
            embedding = precomputed_embedding
        elif feature_extractor is not None:
            embedding = feature_extractor.predict(img_array, verbose=0)[0]
        else:
            return True, None, None

        info       = crop_centroids[predicted_crop]
        target_sim = cosine_similarity(embedding, info["centroid"])
        threshold  = info.get("min_similarity", SIMILARITY_FALLBACK_MIN)

        all_sims = {
            name: cosine_similarity(embedding, c["centroid"])
            for name, c in crop_centroids.items()
        }
        best_match = max(all_sims, key=all_sims.get)
        print(f"  Feature similarity — {predicted_crop}: {target_sim:.3f} (floor: {threshold:.3f}) | best overall: {best_match} ({all_sims[best_match]:.3f})")

        if target_sim < threshold:
            return False, target_sim, best_match
        return True, target_sim, best_match

    except Exception as e:
        print(f"  ⚠️  Feature similarity check error: {e}")
        return True, None, None  # fail open on any error

def validate_image_quality(image_bytes):
    """
    Orchestrates all three no-retrain gates in order, cheapest first.
    Returns (is_valid: bool, reason: str | None, debug_info: dict)
    reason is one of: "blurry", "no_leaf", "not_a_plant", or None if valid.
    """
    debug = {}
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return False, "blurry", {"error": str(e)}

    # Gate 1: Blur / quality (cheapest — pure pixel math, no model)
    blurry, blur_score = is_blurry(pil_img)
    debug["blur_score"] = round(blur_score, 2)
    if blurry:
        print(f"  ❌ Rejected — too blurry (score: {blur_score:.1f}, threshold: {BLUR_EDGE_VARIANCE_THRESHOLD})")
        return False, "blurry", debug

    # Gate 2: Leaf/plant-tissue coverage (cheap — pixel math, no model)
    coverage = leaf_coverage_ratio(pil_img)
    debug["leaf_coverage"] = round(coverage, 3)
    if coverage < LEAF_GREEN_RATIO_THRESHOLD:
        print(f"  ❌ Rejected — low leaf coverage ({coverage*100:.1f}%, threshold: {LEAF_GREEN_RATIO_THRESHOLD*100:.0f}%)")
        return False, "no_leaf", debug

    # Gate 3: ImageNet sanity check (one extra forward pass, no retrain)
    # SPEED OPTIMIZATION: if leaf coverage is already comfortably high,
    # skip this gate — Gate 4 (feature similarity) still runs afterward
    # and provides a deeper semantic check, so this isn't unguarded.
    if coverage >= HIGH_CONFIDENCE_LEAF_RATIO:
        debug["imagenet_top5"] = "SKIPPED (high leaf coverage)"
    else:
        passes, labels = imagenet_sanity_check(pil_img)
        debug["imagenet_top5"] = labels
        if not passes:
            print(f"  ❌ Rejected — ImageNet sanity check failed: {labels}")
            return False, "not_a_plant", debug

    return True, None, debug

def detect_crop_and_disease(img_array):
    """
    Returns: (crop, disease, confidence)
    If unknown/unsupported: ("Unknown", "Not a supported crop", confidence)
    """

    # ---- Step 1: Crop Classifier ----
    if crop_classifier is not None and crop_classes:
        try:
            # SPEED OPTIMIZATION: if the combined dual-output model is
            # available, get BOTH the embedding (for Gate 4) and the
            # softmax prediction from a SINGLE forward pass, instead of
            # calling .predict() twice on overlapping layers.
            precomputed_embedding = None
            if crop_combined_model is not None:
                embedding_out, preds = crop_combined_model.predict(img_array, verbose=0)
                precomputed_embedding = embedding_out[0]
            else:
                preds = crop_classifier.predict(img_array, verbose=0)

            sorted_idx = np.argsort(preds[0])[::-1]   # descending order of probability
            max_prob = float(preds[0][sorted_idx[0]])
            second_prob = float(preds[0][sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
            margin   = max_prob - second_prob
            crop_idx = str(sorted_idx[0])
            detected = crop_classes.get(crop_idx)

            print(f"  Crop classifier: {detected} ({max_prob*100:.1f}%) | margin: {margin*100:.1f}pp")

            # Low confidence — not a supported crop
            if max_prob < CROP_CONF_THRESHOLD:
                print(f"  ⚠️  Low crop confidence ({max_prob*100:.1f}%) — rejecting")
                return "Unknown", "Not a supported crop", round(max_prob * 100, 2)

            # Ambiguous — top two classes are too close together.
            # This is what catches multi-leaf / mixed-crop / unclear-angle
            # images that would otherwise pass the raw confidence check.
            if margin < CROP_MARGIN_THRESHOLD:
                print(f"  ⚠️  Ambiguous crop prediction (margin {margin*100:.1f}pp < {CROP_MARGIN_THRESHOLD*100:.0f}pp) — rejecting")
                return "Unknown", "Ambiguous - multiple possible crops", round(max_prob * 100, 2)

            # "Other" class detected
            if detected == "Other" or detected not in loaded_models:
                print(f"  ⚠️  Unsupported crop detected: {detected}")
                return "Unknown", "Not a supported crop", round(max_prob * 100, 2)

            # ---- Gate 4: Feature Similarity Check ----
            # This is the strongest defense against mango/unknown-crop
            # confidently predicted as a known crop. Even if softmax says
            # "92% Tomato," this checks whether the image's actual feature
            # fingerprint genuinely resembles known Tomato images. Fails
            # open automatically if build_centroids.py hasn't been run yet.
            sim_passes, sim_score, sim_best = feature_similarity_check(img_array, detected, precomputed_embedding)
            if not sim_passes:
                print(f"  ⚠️  Feature similarity rejected — image doesn't resemble known {detected} fingerprint (score: {sim_score:.3f})")
                return "Unknown", "Ambiguous - multiple possible crops", round(max_prob * 100, 2)

            # ---- Step 2: Disease Detection ----
            d_preds  = loaded_models[detected].predict(img_array, verbose=0)
            d_max    = float(np.max(d_preds))
            d_idx    = str(np.argmax(d_preds))
            disease  = loaded_classes[detected].get(d_idx, "Unknown")

            print(f"  Disease: {disease} ({d_max*100:.1f}%)")

            # Low disease confidence
            if d_max < DISEASE_CONF_THRESHOLD:
                print(f"  ⚠️  Low disease confidence — uncertain result")
                return detected, "Uncertain - Please retake photo", round(d_max * 100, 2)

            return detected, disease, round(d_max * 100, 2)

        except Exception as e:
            print(f"  Classifier error: {e}")

    # ---- Fallback: Try all models ----
    print("  Using fallback — trying all models...")
    best_crop, best_disease, best_conf = None, None, 0.0

    for crop, model in loaded_models.items():
        try:
            preds = model.predict(img_array, verbose=0)
            conf  = float(np.max(preds))
            if conf > best_conf:
                best_conf    = conf
                best_crop    = crop
                best_disease = loaded_classes[crop].get(str(np.argmax(preds)), "Unknown")
        except:
            pass

    print(f"  Best fallback: {best_crop} → {best_disease} ({best_conf*100:.1f}%)")

    # Final confidence check
    if best_conf < CROP_CONF_THRESHOLD:
        return "Unknown", "Not a supported crop", round(best_conf * 100, 2)

    return best_crop, best_disease, round(best_conf * 100, 2)

# ============================================================
# WEATHER
# ============================================================
def get_weather(city="Lahore"):
    try:
        if not WEATHER_KEY:
            raise Exception("No key")
        url  = f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={WEATHER_KEY}"
        data = requests.get(url, timeout=5).json()
        return {
            "temp":      round(data["main"]["temp"]),
            "humidity":  data["main"]["humidity"],
            "condition": data["weather"][0]["description"].title(),
            "city":      data["name"],
        }
    except:
        return {
            "temp": 28, "humidity": 72,
            "condition": "Partly Cloudy", "city": city
        }

# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "message":         "Farm Vision AI API v3.3",
        "models_loaded":   list(loaded_models.keys()),
        "crop_classifier": crop_classifier is not None,
        "crop_classes":    list(crop_classes.values()) if crop_classes else [],
        "voice_support":   SR_AVAILABLE,
        "sanity_gate":      sanity_model is not None,
        "similarity_gate":  bool(crop_centroids),
        "thresholds": {
            "crop_confidence":    CROP_CONF_THRESHOLD,
            "disease_confidence": DISEASE_CONF_THRESHOLD,
            "crop_margin":         CROP_MARGIN_THRESHOLD,
            "blur_edge_variance":  BLUR_EDGE_VARIANCE_THRESHOLD,
            "leaf_green_ratio":    LEAF_GREEN_RATIO_THRESHOLD,
        },
        "status": "running",
    }

@app.post("/predict")
async def predict(
    file:     UploadFile = File(...),
    city:     str = "Lahore",
    language: str = "en",
):
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files allowed")

    image_bytes = await file.read()
    weather     = get_weather(city)

    # ---- Pre-Classification Validation Gate ----
    # Runs before the crop classifier ever sees the image. Catches blurry
    # photos, low-leaf-coverage images, and non-plant objects (people,
    # electronics, animals) using zero-retrain checks.
    is_valid, reject_reason, debug_info = validate_image_quality(image_bytes)
    if not is_valid:
        msgs = INVALID_IMAGE_MESSAGES.get(reject_reason, INVALID_IMAGE_MESSAGES["not_a_plant"])
        return {
            "crop":              "Invalid",
            "disease":           "Invalid Image",
            "confidence":        0,
            "cause":             msgs.get(language, msgs["en"]),
            "prevention":        "Supported crops: Potato, Tomato, Cucumber, Cauliflower",
            "treatment":         "N/A",
            "weather_note":      "N/A",
            "future_prediction": "N/A",
            "weather":           weather,
            "mode":              "rejected_pre_gate",
            "reject_reason":     reject_reason,
            "debug":             debug_info,
        }

    # ---- CNN Prediction ----
    if loaded_models:
        try:
            img_array           = preprocess_image(image_bytes)
            crop, disease, conf = detect_crop_and_disease(img_array)

            # Unknown / unsupported crop
            if crop == "Unknown":
                lang_msgs = {
                    "en":    "Please upload a clear image of Potato, Tomato, Cucumber or Cauliflower leaf only.",
                    "ur":    "براہ کرم صرف آلو، ٹماٹر، کھیرا یا پھول گوبھی کے پتے کی واضح تصویر اپ لوڈ کریں۔",
                    "roman": "Meherbani karke sirf Aloo, Tamatar, Kheera ya Phool Gobhi ke patte ki wazeh tasveer upload karein.",
                }
                # Ambiguous-margin rejections get their own clearer message
                if disease == "Ambiguous - multiple possible crops":
                    amb_msgs = INVALID_IMAGE_MESSAGES["ambiguous"]
                    return {
                        "crop":             "Unknown",
                        "disease":          "Ambiguous",
                        "confidence":       conf,
                        "cause":            amb_msgs.get(language, amb_msgs["en"]),
                        "prevention":       "Please retake with a single leaf clearly in frame.",
                        "treatment":        "N/A",
                        "weather_note":     "N/A",
                        "future_prediction":"N/A",
                        "weather":          weather,
                        "mode":             "rejected_ambiguous",
                    }
                return {
                    "crop":             "Unknown",
                    "disease":          "Not a supported crop",
                    "confidence":       conf,
                    "cause":            lang_msgs.get(language, lang_msgs["en"]),
                    "prevention":       "Supported crops: Potato, Tomato, Cucumber, Cauliflower",
                    "treatment":        "N/A",
                    "weather_note":     "N/A",
                    "future_prediction":"N/A",
                    "weather":          weather,
                    "mode":             "rejected",
                }

            # Uncertain result
            if disease == "Uncertain - Please retake photo":
                return {
                    "crop":             crop,
                    "disease":          "Uncertain",
                    "confidence":       conf,
                    "cause":            "Image is not clear enough for accurate detection.",
                    "prevention":       "Please retake photo in good lighting with the leaf clearly visible.",
                    "treatment":        "N/A",
                    "weather_note":     "N/A",
                    "future_prediction":"Retake photo for accurate results.",
                    "weather":          weather,
                    "mode":             "uncertain",
                }

            mode = "real_model"

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, str(e))

    else:
        # Demo mode
        import random
        pairs         = [
            ("Tomato",      "Early Blight"),
            ("Potato",      "Late Blight"),
            ("Cucumber",    "Downy Mildew"),
            ("Cauliflower", "Black Rot"),
            ("Tomato",      "Healthy"),
        ]
        crop, disease = random.choice(pairs)
        conf          = round(random.uniform(75, 95), 2)
        mode          = "demo"

    info = get_disease_info(disease, language)

    return {
        "crop":              crop,
        "disease":           disease,
        "confidence":        conf,
        "cause":             info["cause"],
        "prevention":        info["prevention"],
        "treatment":         info["treatment"],
        "weather_note":      info["weather_note"],
        "future_prediction": info["prediction"],
        "weather":           weather,
        "mode":              mode,
    }

# ---- Voice Query ----
@app.post("/voice-query")
async def voice_query(
    file:     UploadFile = File(...),
    language: str        = Form(default="en"),
):
    try:
        return await _voice_query_impl(file, language)
    except Exception as e:
        import traceback
        print(f"\n❌ VOICE QUERY CRASHED: {type(e).__name__}: {e}")
        traceback.print_exc()
        # Return a graceful JSON error instead of a raw 500 — this also
        # surfaces the actual exception type/message for debugging,
        # instead of just a generic "Voice processing failed" on the app.
        return {
            "crop":              "Invalid",
            "disease":           "Server Error",
            "confidence":        0,
            "voice_text":        "",
            "cause":             f"Voice processing crashed: {type(e).__name__}: {str(e)}",
            "prevention":        "N/A",
            "treatment":         "N/A",
            "weather_note":      "N/A",
            "future_prediction": "N/A",
            "weather":           {},
            "mode":              "voice_error",
        }

async def _voice_query_impl(file, language):
    audio_bytes     = await file.read()
    recognized_text = ""

    if SR_AVAILABLE:
        lang_map = {"en": "en-US", "ur": "ur-PK", "roman": "ur-PK"}
        sr_lang  = lang_map.get(language, "en-US")
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            recognizer = sr.Recognizer()
            with sr.AudioFile(tmp_path) as source:
                audio_data = recognizer.record(source)
            try:
                recognized_text = recognizer.recognize_google(audio_data, language=sr_lang)
                print(f"  🎤 Recognized via '{sr_lang}': \"{recognized_text}\"")
            except Exception as e1:
                print(f"  ⚠️  Recognition failed on '{sr_lang}' ({type(e1).__name__}) — retrying with en-US")
                recognized_text = recognizer.recognize_google(audio_data, language="en-US")
                print(f"  🎤 Recognized via 'en-US' (fallback): \"{recognized_text}\"")
            os.unlink(tmp_path)
        except Exception as e:
            print(f"  ❌ Voice recognition completely failed: {type(e).__name__}: {e}")
            recognized_text = ""

    # Detect crop from text
    text_lower = recognized_text.strip().lower()
    print(f"\n🎤 VOICE DEBUG — Raw recognized text: {recognized_text!r}")
    print(f"🎤 VOICE DEBUG — Lowercased: {text_lower!r}")

    # ---- Validation Gate: No speech detected ----
    # Previously this silently defaulted to "Tomato" when recognized_text
    # was empty (e.g. silence, background noise, unrecognized speech) —
    # meaning a 1-second silent recording would still return a confident
    # "Tomato" result. Now we explicitly detect this and reject it.
    if not text_lower:
        no_speech_msgs = {
            "en":    "No speech was detected. Please speak clearly and mention the crop name (e.g. Tomato, Potato, Cucumber, or Cauliflower).",
            "ur":    "کوئی آواز محسوس نہیں ہوئی۔ براہ کرم واضح طور پر بولیں اور فصل کا نام بتائیں (مثلاً ٹماٹر، آلو، کھیرا، یا گوبھی)۔",
            "roman": "Koi awaz mehsoos nahi hui. Meherbani karke wazeh tor par bolen aur fasal ka naam batayen (maslan Tamatar, Aloo, Kheera, ya Gobhi).",
        }
        weather = get_weather()
        return {
            "crop":              "Invalid",
            "disease":           "No Speech Detected",
            "confidence":        0,
            "voice_text":        recognized_text,
            "cause":             no_speech_msgs.get(language, no_speech_msgs["en"]),
            "prevention":        "Supported crops: Potato, Tomato, Cucumber, Cauliflower",
            "treatment":         "N/A",
            "weather_note":      "N/A",
            "future_prediction": "N/A",
            "weather":           weather,
            "mode":              "voice_rejected_no_speech",
        }

    crop_map   = {
        "Tomato":      ["tomato", "tamatar", "tamater", "ٹماٹر"],
        "Potato":      ["potato", "aloo", "alu", "آلو"],
        "Cucumber":    ["cucumber", "kheera", "khera", "khery", "khairy", "khaira", "kheri", "کھیرا", "khira"],
        "Cauliflower": ["cauliflower", "gobhi", "gobi", "phool gobhi", "گوبھی"],
    }
    detected_crop = None
    for crop, keywords in crop_map.items():
        matched_kw = [kw for kw in keywords if kw in text_lower]
        if matched_kw:
            print(f"🎤 VOICE DEBUG — Crop '{crop}' matched via keyword(s): {matched_kw}")
            detected_crop = crop
            break
    print(f"🎤 VOICE DEBUG — Final detected_crop: {detected_crop}")
    print(f"  📋 Text used for matching: \"{text_lower}\"  →  detected_crop: {detected_crop}")

    # ---- Validation Gate: Speech detected, but no known crop named ----
    # E.g. user said something unrelated ("hello", random words). Previously
    # this ALSO silently defaulted to "Tomato" — now it's explicitly rejected.
    if detected_crop is None:
        unrecognized_msgs = {
            "en":    f"I heard \"{recognized_text}\" but couldn't identify a supported crop name. Please mention Tomato, Potato, Cucumber, or Cauliflower.",
            "ur":    f"میں نے \"{recognized_text}\" سنا لیکن کوئی معاون فصل کا نام شناخت نہیں ہو سکا۔ براہ کرم ٹماٹر، آلو، کھیرا، یا گوبھی کا نام بتائیں۔",
            "roman": f"Maine \"{recognized_text}\" suna lekin koi supported fasal ka naam shanakht nahi ho saka. Meherbani karke Tamatar, Aloo, Kheera, ya Gobhi ka naam batayen.",
        }
        weather = get_weather()
        return {
            "crop":              "Invalid",
            "disease":           "Crop Not Recognized",
            "confidence":        0,
            "voice_text":        recognized_text,
            "cause":             unrecognized_msgs.get(language, unrecognized_msgs["en"]),
            "prevention":        "Supported crops: Potato, Tomato, Cucumber, Cauliflower",
            "treatment":         "N/A",
            "weather_note":      "N/A",
            "future_prediction": "N/A",
            "weather":           weather,
            "mode":              "voice_rejected_unrecognized",
        }

    # ---- Match disease/symptom keywords mentioned in the speech ----
    # Replaces the previous random.choice() — now the disease returned is
    # actually based on what the farmer said, not a random guess.
    valid_diseases = {
        "Tomato":      ["Early Blight", "Late Blight", "Leaf Mold", "Powdery Mildew", "Mosaic Virus", "Healthy"],
        "Potato":      ["Early Blight", "Late Blight", "Healthy"],
        "Cucumber":    ["Downy Mildew", "Anthracnose", "Healthy"],
        "Cauliflower": ["Black Rot", "Downy Mildew", "Healthy"],
    }
    # Keyword list per disease — English scientific terms + commonly used
    # Roman Urdu / Urdu terms. Ordered by specificity (checked most-specific
    # phrase first) to avoid e.g. generic "blight" matching before "early blight".
    DISEASE_KEYWORDS = {
        "Early Blight":   ["early blight", "alternaria"],
        "Late Blight":    ["late blight", "phytophthora"],
        "Leaf Mold":      ["leaf mold", "leaf mould"],
        "Powdery Mildew": ["powdery mildew", "phaphoondi", "phaphundi", "پھپھوندی"],
        "Mosaic Virus":   ["mosaic virus", "mosaic", "virus"],
        "Anthracnose":    ["anthracnose"],
        "Downy Mildew":   ["downy mildew"],
        "Black Rot":      ["black rot"],
        "Healthy":        ["healthy", "no disease", "theek hai", "ٹھیک ہے", "sehatmand", "صحت مند"],
    }
    # Ambiguous generic terms — used only if no specific match was found
    GENERIC_BLIGHT_TERMS = ["blight", "jhulsa", "جھلسا"]
    GENERIC_ROT_TERMS     = ["rot", "ganlna", "گلنا"]

    # ---- Symptom descriptions (not disease names) ----
    # Farmers describe what they SEE, not the scientific disease name.
    # Each entry: (phrase variants, target disease) — checked per-crop,
    # only applied if that disease is actually valid for the detected crop.
    SYMPTOM_KEYWORDS = [
        (["black spot", "black spots", "black mark", "black marks", "kaale nishan", "kaale daagh", "kaale dhabbay", "kale nishan", "کالے نشان", "کالے دھبے"], "Anthracnose"),
        (["black rot", "kaala sarna", "کالا گلنا"], "Black Rot"),
        (["white powder", "safed powder", "powder jaisa", "سفید پاؤڈر"], "Powdery Mildew"),
        (["yellow spot", "yellow spots", "zard dhabbay", "زرد دھبے"], "Mosaic Virus"),
        (["wilting", "sookna", "سوکھنا", "muraana"], "Late Blight"),
        (["fuzzy", "fluffy", "rooi jaisa", "روئی جیسا"], "Downy Mildew"),
    ]

    crop_diseases = valid_diseases.get(detected_crop, ["Healthy"])
    matched_disease = None

    # Pass 1: specific multi-word / unambiguous disease-name keyword match
    for d in crop_diseases:
        for kw in DISEASE_KEYWORDS.get(d, []):
            if kw in text_lower:
                matched_disease = d
                break
        if matched_disease:
            break

    # Pass 1.5: symptom DESCRIPTIONS (farmers describe what they see,
    # not the scientific disease name) — e.g. "black spots" -> Anthracnose,
    # but only if that disease is actually valid for this crop.
    if matched_disease is None:
        for phrases, target_disease in SYMPTOM_KEYWORDS:
            if target_disease in crop_diseases and any(p in text_lower for p in phrases):
                matched_disease = target_disease
                break

    # Pass 2: generic ambiguous terms (only if crop has just ONE matching
    # option, so it isn't actually ambiguous for THIS crop)
    if matched_disease is None:
        if any(t in text_lower for t in GENERIC_BLIGHT_TERMS):
            blight_options = [d for d in crop_diseases if "Blight" in d]
            if len(blight_options) == 1:
                matched_disease = blight_options[0]
            elif len(blight_options) > 1:
                matched_disease = "AMBIGUOUS_BLIGHT"
        elif any(t in text_lower for t in GENERIC_ROT_TERMS):
            rot_options = [d for d in crop_diseases if "Rot" in d]
            if rot_options:
                matched_disease = rot_options[0]

    weather = get_weather()

    # Ambiguous blight mentioned but crop has both Early and Late Blight —
    # cannot tell which without seeing the leaf. Be honest about this.
    if matched_disease == "AMBIGUOUS_BLIGHT":
        ambiguous_msgs = {
            "en":    f"You mentioned blight in {detected_crop}, but I can't tell if it's Early Blight or Late Blight from voice alone — they look different on the leaf. Please upload a photo for an accurate diagnosis.",
            "ur":    f"آپ نے {detected_crop} میں جھلسا بیماری کا ذکر کیا، لیکن آواز سے یہ معلوم نہیں ہو سکتا کہ یہ ابتدائی جھلسا ہے یا تاخیری جھلسا۔ براہ کرم درست تشخیص کے لیے تصویر اپ لوڈ کریں۔",
            "roman": f"Aap ne {detected_crop} mein jhulsa bimari ka zikar kiya, lekin awaz se yeh maloom nahi ho sakta ke yeh Early Blight hai ya Late Blight. Meherbani karke sahi tashkhees ke liye tasveer upload karein.",
        }
        return {
            "crop":              detected_crop,
            "disease":           "Ambiguous - Blight Type Unclear",
            "confidence":        0,
            "voice_text":        recognized_text,
            "cause":             ambiguous_msgs.get(language, ambiguous_msgs["en"]),
            "prevention":        "Upload a leaf photo for precise identification.",
            "treatment":         "N/A",
            "weather_note":      "N/A",
            "future_prediction": "N/A",
            "weather":           weather,
            "mode":              "voice_ambiguous_disease",
        }

    # Crop was named, but no disease or symptom was mentioned at all —
    # give general crop information instead of guessing a disease.
    if matched_disease is None:
        general_msgs = {
            "en":    f"You mentioned {detected_crop}, but didn't describe any symptoms. Common {detected_crop} diseases include: {', '.join(d for d in crop_diseases if d != 'Healthy')}. Please describe the symptoms you see, or upload a photo for an exact diagnosis.",
            "ur":    f"آپ نے {detected_crop} کا ذکر کیا، لیکن کوئی علامات نہیں بتائیں۔ {detected_crop} کی عام بیماریاں: {', '.join(d for d in crop_diseases if d != 'Healthy')}۔ براہ کرم علامات بتائیں یا تصویر اپ لوڈ کریں۔",
            "roman": f"Aap ne {detected_crop} ka zikar kiya, lekin koi alamat nahi batayi. {detected_crop} ki aam bimariyan: {', '.join(d for d in crop_diseases if d != 'Healthy')}. Meherbani karke alamat batayen ya tasveer upload karein.",
        }
        return {
            "crop":              detected_crop,
            "disease":           "No Symptoms Described",
            "confidence":        0,
            "voice_text":        recognized_text,
            "cause":             general_msgs.get(language, general_msgs["en"]),
            "prevention":        "Describe symptoms (spots, wilting, yellowing) or upload a photo.",
            "treatment":         "N/A",
            "weather_note":      "N/A",
            "future_prediction": "N/A",
            "weather":           weather,
            "mode":              "voice_general_info",
        }

    # ---- Matched a real disease from speech — return its actual info ----
    disease = matched_disease
    info    = get_disease_info(disease, language)

    return {
        "crop":              detected_crop,
        "disease":           disease,
        "confidence":        None,   # Not a CNN prediction — no confidence score to report
        "voice_text":        recognized_text,
        "cause":             info["cause"],
        "prevention":        info["prevention"],
        "treatment":         info["treatment"],
        "weather_note":      info["weather_note"],
        "future_prediction": info["prediction"],
        "weather":           weather,
        "mode":              "voice_query",
    }

@app.get("/weather/{city}")
def weather_endpoint(city: str):
    return get_weather(city)

# ============================================================
# STARTUP
# ============================================================
@app.on_event("startup")
async def startup():
    print("\n🌿 Farm Vision API v3.3 Starting...")
    load_models()
    load_sanity_model()
    print(f"\n📊 Status:")
    print(f"   Models:          {list(loaded_models.keys())}")
    print(f"   Crop Classifier: {'✅' if crop_classifier else '❌'}")
    print(f"   Sanity Gate:     {'✅' if sanity_model else '❌ (fails open — no rejection)'}")
    print(f"   Similarity Gate: {'✅' if crop_centroids else '❌ (run build_centroids.py to activate)'}")
    print(f"   Voice:           {'✅' if SR_AVAILABLE else '❌'}")
    print(f"   Crop Threshold:  {CROP_CONF_THRESHOLD*100}%")
    print(f"   Disease Thresh:  {DISEASE_CONF_THRESHOLD*100}%")
    print(f"   Crop Margin:     {CROP_MARGIN_THRESHOLD*100}pp")
    print(f"   Blur Threshold:  {BLUR_EDGE_VARIANCE_THRESHOLD}")
    print(f"   Leaf Coverage:   {LEAF_GREEN_RATIO_THRESHOLD*100}%")
    print(f"\n✅ API Ready!\n")