import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
DATASET_DIR   = "datasets"      # ← aapka folder naam "datasets" hai
MODEL_DIR     = "models"
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS        = 30
LEARNING_RATE = 0.0001

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# CROP CONFIG
# folder_name   → actual folder name in datasets/
# mode          → "split"  = train/val folders alag hain
#                 "auto"   = classes directly hain, auto 80/20 split
# train_folder  → (sirf "split" mode ke liye)
# val_folder    → (sirf "split" mode ke liye)
# ============================================================
CROP_CONFIG = {
    "Potato": {
        "folder_name":  "Potato",
        "mode":         "split",
        "train_folder": "Training",
        "val_folder":   "Validation",
    },
    "Tomato": {
        "folder_name":  "Tomato Diseases in Pakistan",
        "mode":         "split",
        "train_folder": "train",
        "val_folder":   "valid",
    },
    "Cucumber": {
        "folder_name":  "Cucumber",
        "mode":         "auto",   # classes directly hain
    },
    "Cauliflower": {
        "folder_name":  "Cauliflower",
        "mode":         "auto",   # classes directly hain
    },
}

# ============================================================
# DISEASE NAME MAPPING
# Training folder name → Display name (for app)
# ============================================================
DISEASE_DISPLAY = {
    # Potato
    "Early_Blight":         "Early Blight",
    "Late_Blight":          "Late Blight",
    "Healthy":              "Healthy",
    # Tomato
    "Early_blight":         "Early Blight",
    "Late_blight":          "Late Blight",
    "Leaf_Mold":            "Leaf Mold",
    "powdery_mildew":       "Powdery Mildew",
    "Tomato_mosaic_virus":  "Mosaic Virus",
    "healthy":              "Healthy",
    # Cucumber
    "Anthracnose lesions":  "Anthracnose",
    "Downy mildew":         "Downy Mildew",
    "Fresh leaf":           "Healthy",
    # Cauliflower
    "Black Rot":            "Black Rot",
    # "Downy mildew" already above
    # "Fresh leaf"   already above
}

# ============================================================
# CREATE GENERATORS — Split Mode
# (Potato, Tomato — alag train/val folders)
# ============================================================
def create_generators_split(crop_path, train_folder, val_folder):
    train_path = os.path.join(crop_path, train_folder)
    val_path   = os.path.join(crop_path, val_folder)

    if not os.path.exists(train_path):
        print(f"  ❌ Train folder nahi mila: {train_path}")
        return None, None
    if not os.path.exists(val_path):
        print(f"  ❌ Val folder nahi mila: {val_path}")
        return None, None

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        val_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )
    return train_gen, val_gen

# ============================================================
# CREATE GENERATORS — Auto Mode
# (Cucumber, Cauliflower — classes directly hain)
# 80% train, 20% validation auto split
# ============================================================
def create_generators_auto(crop_path):
    if not os.path.exists(crop_path):
        print(f"  ❌ Folder nahi mila: {crop_path}")
        return None, None

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=0.2,   # 20% val
    )
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
    )

    train_gen = train_datagen.flow_from_directory(
        crop_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        crop_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
    )
    return train_gen, val_gen

# ============================================================
# BUILD MODEL — MobileNetV2
# ============================================================
def build_model(num_classes):
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model, base_model

# ============================================================
# PLOT TRAINING
# ============================================================
def plot_training(h1, h2, crop_name):
    try:
        acc      = h1.history['accuracy']     + h2.history['accuracy']
        val_acc  = h1.history['val_accuracy'] + h2.history['val_accuracy']
        loss     = h1.history['loss']         + h2.history['loss']
        val_loss = h1.history['val_loss']     + h2.history['val_loss']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(acc, label='Train Accuracy')
        ax1.plot(val_acc, label='Val Accuracy')
        ax1.set_title(f'{crop_name} — Accuracy')
        ax1.legend(); ax1.grid(True)

        ax2.plot(loss, label='Train Loss')
        ax2.plot(val_loss, label='Val Loss')
        ax2.set_title(f'{crop_name} — Loss')
        ax2.legend(); ax2.grid(True)

        plt.tight_layout()
        save_path = os.path.join(MODEL_DIR, f"{crop_name}_plot.png")
        plt.savefig(save_path)
        plt.close()
        print(f"  ✅ Plot saved: {save_path}")
    except Exception as e:
        print(f"  ⚠️  Plot error (training continue hai): {e}")

# ============================================================
# TRAIN ONE CROP
# ============================================================
def train_crop(crop_name):
    print(f"\n{'='*55}")
    print(f"  🌿 Training: {crop_name}")
    print(f"{'='*55}")

    config      = CROP_CONFIG[crop_name]
    folder_name = config["folder_name"]
    mode        = config["mode"]
    crop_path   = os.path.join(DATASET_DIR, folder_name)

    if not os.path.exists(crop_path):
        print(f"  ⚠️  Skipping — folder nahi mila: {crop_path}")
        return

    # Generators
    if mode == "split":
        train_gen, val_gen = create_generators_split(
            crop_path,
            config["train_folder"],
            config["val_folder"],
        )
    else:  # auto
        train_gen, val_gen = create_generators_auto(crop_path)

    if train_gen is None:
        return

    num_classes   = len(train_gen.class_indices)
    class_indices = train_gen.class_indices

    print(f"  ✅ Classes found:  {list(class_indices.keys())}")
    print(f"  ✅ Train images:   {train_gen.samples}")
    print(f"  ✅ Val images:     {val_gen.samples}")
    print(f"  ✅ Num classes:    {num_classes}")

    # Save class map: index → display name
    idx_to_class = {}
    for class_name, idx in class_indices.items():
        display = DISEASE_DISPLAY.get(class_name, class_name)
        idx_to_class[str(idx)] = display

    class_file = os.path.join(MODEL_DIR, f"{crop_name}_classes.json")
    with open(class_file, 'w') as f:
        json.dump(idx_to_class, f, indent=2)
    print(f"  ✅ Class map saved: {class_file}")
    print(f"  📋 Map: {idx_to_class}")

    # Model
    model, base_model = build_model(num_classes)
    model_path = os.path.join(MODEL_DIR, f"{crop_name}_model.h5")

    callbacks = [
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # Phase 1 — Base frozen, top layers train
    print(f"\n  📌 Phase 1: Top layers training (base frozen)...")
    history1 = model.fit(
        train_gen,
        epochs=15,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    # Phase 2 — Fine tune last 30 layers
    print(f"\n  📌 Phase 2: Fine-tuning (last 30 layers)...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    history2 = model.fit(
        train_gen,
        epochs=EPOCHS,
        initial_epoch=len(history1.history['accuracy']),
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"\n  🎯 {crop_name} Final Accuracy: {val_acc*100:.2f}%")
    print(f"  ✅ Model saved: {model_path}")

    plot_training(history1, history2, crop_name)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🌿 Farm Vision — CNN Model Training")
    print("="*55)

    if not os.path.exists(DATASET_DIR):
        print(f"\n❌ '{DATASET_DIR}/' folder nahi mila!")
        print(f"   Make sure dataset folder 'datasets' naam ka hai")
        exit(1)

    print(f"\n📁 Checking datasets in '{DATASET_DIR}/'...")
    available = []
    for crop_name, config in CROP_CONFIG.items():
        path = os.path.join(DATASET_DIR, config["folder_name"])
        if os.path.exists(path):
            available.append(crop_name)
            print(f"  ✅ {crop_name:12} → {path}")
        else:
            print(f"  ⚠️  {crop_name:12} → NOT FOUND: {path}")

    if not available:
        print("\n❌ Koi bhi crop nahi mila! Dataset check karo.")
        exit(1)

    print(f"\n🚀 Training shuru: {available}")
    print("="*55)

    for crop in available:
        train_crop(crop)

    print("\n" + "="*55)
    print("  ✅ Training Complete!")
    print(f"  📁 Models saved in: {MODEL_DIR}/")
    print("\n  🚀 Ab FastAPI chalao:")
    print("  uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
    print("="*55 + "\n")