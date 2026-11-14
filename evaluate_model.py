
import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths and settings
DATASET_PATH = "PlantVillage/PlantVillage"
MODEL_PATH = "plant_disease_no_finetune.h5"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Load trained model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Prepare validation data (same split as original notebook)
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

validation_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=SEED
)

# Check class mapping
print("\nClass mapping:")
print(validation_generator.class_indices)

# Predict
print("\nPredicting validation images...")
predictions = model.predict(validation_generator, verbose=1)

y_pred = np.argmax(predictions, axis=1)
y_true = validation_generator.classes

class_names = list(validation_generator.class_indices.keys())

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print("\n========== MODEL EVALUATION ==========")
print(f"Validation images: {len(y_true)}")
print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
))

# Save confusion matrix
cm = confusion_matrix(y_true, y_pred)

np.savetxt(
    "confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d"
)

print("\nConfusion matrix saved as confusion_matrix.csv")
