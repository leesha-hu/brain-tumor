# fix_resnet_model.py
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(__file__)
OLD_MODEL = os.path.join(BASE_DIR, "models", "resnet50_final_finetuned.h5")
NEW_MODEL = os.path.join(BASE_DIR, "models", "resnet50_fixed.keras")

print("Loading old model...")
old_model = load_model(OLD_MODEL, compile=False)

print("Rebuilding model as Functional API...")
inputs = tf.keras.Input(shape=(224, 224, 3))
outputs = old_model(inputs)
fixed_model = tf.keras.Model(inputs, outputs)

fixed_model.trainable = False

print("Saving fixed model...")
fixed_model.save(NEW_MODEL)

print("✅ Model fixed and saved as:", NEW_MODEL)
