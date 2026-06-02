import os, json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Crop classifier dataset structure:
# crop_dataset/
# ├── Potato/     ← kuch images
# ├── Tomato/     ← kuch images  
# ├── Cucumber/   ← kuch images
# └── Cauliflower/ ← kuch images

CROP_DATASET = "crop_dataset"
MODEL_DIR = "models"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

os.makedirs(MODEL_DIR, exist_ok=True)

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    CROP_DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    CROP_DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

num_classes = len(train_gen.class_indices)
print(f"Crops found: {train_gen.class_indices}")

# Save crop classes
idx_to_crop = {str(v): k for k, v in train_gen.class_indices.items()}
with open(f"{MODEL_DIR}/crop_classes.json", 'w') as f:
    json.dump(idx_to_crop, f, indent=2)

# Build model
base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    callbacks=[
        ModelCheckpoint(f"{MODEL_DIR}/crop_classifier.h5", save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=5, restore_best_weights=True)
    ]
)

print("✅ Crop classifier saved!")