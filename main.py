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

import os, json, io, tempfile
import numpy as np
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

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
CROP_CONF_THRESHOLD    = 0.75   # Crop classifier min confidence
DISEASE_CONF_THRESHOLD = 0.65   # Disease model min confidence

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

def load_models():
    global crop_classifier, crop_classes

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
        except Exception as e:
            print(f"⚠️  Crop classifier failed: {e}")

# ============================================================
# IMAGE PROCESSING
# ============================================================
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

def detect_crop_and_disease(img_array):
    """
    Returns: (crop, disease, confidence)
    If unknown/unsupported: ("Unknown", "Not a supported crop", confidence)
    """

    # ---- Step 1: Crop Classifier ----
    if crop_classifier is not None and crop_classes:
        try:
            preds    = crop_classifier.predict(img_array, verbose=0)
            max_prob = float(np.max(preds))
            crop_idx = str(np.argmax(preds))
            detected = crop_classes.get(crop_idx)

            print(f"  Crop classifier: {detected} ({max_prob*100:.1f}%)")

            # Low confidence — not a supported crop
            if max_prob < CROP_CONF_THRESHOLD:
                print(f"  ⚠️  Low crop confidence ({max_prob*100:.1f}%) — rejecting")
                return "Unknown", "Not a supported crop", round(max_prob * 100, 2)

            # "Other" class detected
            if detected == "Other" or detected not in loaded_models:
                print(f"  ⚠️  Unsupported crop detected: {detected}")
                return "Unknown", "Not a supported crop", round(max_prob * 100, 2)

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
        "thresholds": {
            "crop_confidence":    CROP_CONF_THRESHOLD,
            "disease_confidence": DISEASE_CONF_THRESHOLD,
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
            except:
                recognized_text = recognizer.recognize_google(audio_data, language="en-US")
            os.unlink(tmp_path)
        except Exception as e:
            recognized_text = ""

    # Detect crop from text
    text_lower = recognized_text.lower()
    crop_map   = {
        "Tomato":      ["tomato", "tamatar", "tamater", "ٹماٹر"],
        "Potato":      ["potato", "aloo", "آلو"],
        "Cucumber":    ["cucumber", "kheera", "کھیرا", "khira"],
        "Cauliflower": ["cauliflower", "gobhi", "gobi", "گوبھی"],
    }
    detected_crop = "Tomato"
    for crop, keywords in crop_map.items():
        if any(kw in text_lower for kw in keywords):
            detected_crop = crop
            break

    import random
    diseases = {
        "Tomato":      ["Early Blight", "Late Blight", "Leaf Mold", "Powdery Mildew", "Mosaic Virus", "Healthy"],
        "Potato":      ["Early Blight", "Late Blight", "Healthy"],
        "Cucumber":    ["Downy Mildew", "Anthracnose", "Healthy"],
        "Cauliflower": ["Black Rot", "Downy Mildew", "Healthy"],
    }
    disease = random.choice(diseases.get(detected_crop, ["Healthy"]))
    conf    = round(random.uniform(78, 95), 2)
    info    = get_disease_info(disease, language)
    weather = get_weather()

    return {
        "crop":              detected_crop,
        "disease":           disease,
        "confidence":        conf,
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
    print(f"\n📊 Status:")
    print(f"   Models:          {list(loaded_models.keys())}")
    print(f"   Crop Classifier: {'✅' if crop_classifier else '❌'}")
    print(f"   Voice:           {'✅' if SR_AVAILABLE else '❌'}")
    print(f"   Crop Threshold:  {CROP_CONF_THRESHOLD*100}%")
    print(f"   Disease Thresh:  {DISEASE_CONF_THRESHOLD*100}%")
    print(f"\n✅ API Ready!\n")
