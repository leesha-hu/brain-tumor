import tensorflow as tf
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def rebuild_model(model_folder, output_name):
    folder_path = os.path.join(MODELS_DIR, model_folder)
config_path = os.path.join(folder_path, "config.json")
    weights_path = os.path.join(folder_path, "model.weights.h5")

    print("Looking for:", config_path)
    print("Looking for:", weights_path)

    # load architecture
    with open(config_path, "r") as f:
        config = json.load(f)

    model = tf.keras.models.model_from_json(json.dumps(config))

    # load weights
    model.load_weights(weights_path)

    # save unified keras model INSIDE models/
    output_path = os.path.join(MODELS_DIR, output_name)
    model.save(output_path)

    print(f"✅ Saved model → {output_path}")

# ---------- DenseNet ----------
rebuild_model("densenet", "densenet_model.keras")

# ---------- EfficientNet ----------
rebuild_model("efficient", "efficientnet_model.keras")
