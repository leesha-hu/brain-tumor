# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import login_user, logout_user, login_required, current_user
from extensions import db, login_manager
from models import User
from datetime import datetime
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
from sqlalchemy import or_, and_, desc
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from flask import make_response
import pdfkit
import matplotlib
matplotlib.use('Agg')
PDF_CONFIG = pdfkit.configuration(
     wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
)
def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "devkey"
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///brain_chat.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)
    login_manager.init_app(app)

    # -------------------------------------------------
    # PATHS
    # -------------------------------------------------
    BASE_DIR = os.path.dirname(__file__)
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
    MODEL_FOLDER = os.path.join(BASE_DIR, "models")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)

    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

    def allowed_file(filename):
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    def load_model_from_folder(folder):
        with open(os.path.join(folder, "config.json"), "r") as f:
            model = model_from_json(f.read())
        model.load_weights(os.path.join(folder, "model.weights.h5"))
        model.trainable = False
        return model

    MODEL_REGISTRY = {
        "densenet": load_model_from_folder(os.path.join(MODEL_FOLDER, "densenet")),
        "efficientnet": load_model_from_folder(os.path.join(MODEL_FOLDER, "efficient")),
        "xception": load_model_from_folder(os.path.join(MODEL_FOLDER, "xception")),
       
    } 
    resnet_path = os.path.join(MODEL_FOLDER, "resnet50_best.keras")
    try:
        resnet_model = tf.keras.models.load_model(resnet_path)
        resnet_model.trainable = False
        print("✅ ResNet50 model loaded successfully.")
    except Exception as e:
        print("❌ ERROR while loading ResNet50 model.")
        raise e

  
    print("\n📌 ResNet50 MODEL SUMMARY:")
    resnet_model.summary()

    
    MODEL_REGISTRY["resnet50"] = resnet_model


    CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
   
    resnet_base = resnet_model.get_layer("resnet50")

   
    last_conv_layer = resnet_base.get_layer("conv5_block3_out")

    
    x = last_conv_layer.output
    x = resnet_model.get_layer("global_average_pooling2d")(x)
    x = resnet_model.get_layer("dense")(x)
    x = resnet_model.get_layer("dropout")(x)
    predictions = resnet_model.get_layer("dense_1")(x)

 
    GRADCAM_RESNET_MODEL = tf.keras.Model(
        inputs=resnet_base.input,
        outputs=[last_conv_layer.output, predictions]
    )

    def make_gradcam_heatmap_resnet50(img_array, pred_index=None):

        with tf.GradientTape() as tape:
            conv_outputs, predictions = GRADCAM_RESNET_MODEL(
                img_array, training=False
            )

            if pred_index is None:
                pred_index = tf.argmax(predictions[0])

            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            raise RuntimeError("Grad-CAM failed: gradients are None")

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.reduce_max(heatmap) + 1e-10

        return heatmap.numpy()


    
    def make_gradcam_heatmap_xception(img_array, model, pred_index=None):
        last_conv_layer_name = "cam_conv"

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output
            ]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array, training=False)

            # ✅ CRITICAL FIX
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]

            if pred_index is None:
                pred_index = tf.argmax(predictions[0])

            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.reduce_max(heatmap) + 1e-8

        return heatmap.numpy()


    
    def make_gradcam_heatmap_densenet(img_array, model, pred_index):
        last_conv_layer = None
        for layer in reversed(model.layers):
            try:
                if len(layer.output.shape) == 4:
                    last_conv_layer = layer
                    break
            except Exception:
                continue

        if last_conv_layer is None:
            return None

        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[last_conv_layer.output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array, training=False)
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.reduce_max(heatmap) + 1e-10

        return heatmap.numpy()


    def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
        data_aug_layer = model.get_layer("sequential_1")
        base_model = model.get_layer(last_conv_layer_name)

        classifier_layers = []
        found = False
        for layer in model.layers:
            if layer.name == last_conv_layer_name:
                found = True
                continue
            if found:
                if isinstance(
                    layer,
                    (
                        tf.keras.layers.GlobalAveragePooling2D,
                        tf.keras.layers.Dense,
                        tf.keras.layers.Dropout,
                    ),
                ):
                    classifier_layers.append(layer)

        with tf.GradientTape() as tape:
            augmented = data_aug_layer(img_array, training=False)
            preprocessed = tf.keras.applications.efficientnet.preprocess_input(augmented)

            conv_outputs = base_model(preprocessed, training=False)
            tape.watch(conv_outputs)

            x = conv_outputs
            for layer in classifier_layers:
                x = layer(x)

            preds = x
            if pred_index is None:
                pred_index = tf.argmax(preds[0])

            loss = preds[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.reduce_max(heatmap) + 1e-10

        return heatmap.numpy()

   
    def overlay_gradcam(img_path, heatmap, out_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    
    def preprocess_image(img_path, model_name, model):
        h, w = model.input_shape[1], model.input_shape[2]

        if model_name == "densenet":
            
            img = Image.open(img_path).convert("L").resize((w, h))
            x = np.array(img).astype("float32") / 255.0
            x = np.expand_dims(x, axis=-1)   # (H,W,1)

        else:
            
            img = Image.open(img_path).convert("RGB").resize((w, h))
            x = np.array(img).astype("float32")

            if model_name == "efficientnet":
                x = eff_preprocess(x)
            elif model_name == "xception":
                x = xception_preprocess(x)
            elif model_name == "resnet50":
                x = resnet_preprocess(x)

        return np.expand_dims(x, axis=0)  
    def get_preprocess_function(model_name):

        if model_name == "efficientnet":
            return eff_preprocess

        elif model_name == "xception":
            return xception_preprocess

        elif model_name == "resnet50":
            return resnet_preprocess

        elif model_name == "densenet":
            return lambda x: x / 255.0

        else:
            return lambda x: x
    def shap_to_saliency(shap_values, class_idx):

        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 5:
            shap_values = np.squeeze(shap_values, axis=0)
            s = shap_values[..., class_idx]
            return np.sum(np.abs(s), axis=-1)

        else:
            return np.abs(np.squeeze(shap_values))

    def generate_shap_image(image_path, model_name, idx, save_dir="static/results"):

        import shap
        import matplotlib.pyplot as plt
        from tensorflow.keras.preprocessing import image

        model = MODEL_REGISTRY[model_name]
        IMG_SIZE = (model.input_shape[1], model.input_shape[2])

        preprocess_input = get_preprocess_function(model_name)

        # -------------------------------
        # LOAD IMAGE
        # -------------------------------
        img = image.load_img(image_path, target_size=IMG_SIZE)
        img_arr = image.img_to_array(img)

        # 🔥 IMPORTANT FIX (for grayscale models like densenet)
        if model_name == "densenet":
            img_arr = np.expand_dims(img_arr.squeeze(), axis=-1)

        img_batch = np.expand_dims(img_arr, axis=0)
        img_batch = preprocess_input(img_batch)

        # -------------------------------
        # SHAP (WORKING VERSION)
        # -------------------------------
        masker = shap.maskers.Image("inpaint_telea", img_arr.shape)

        explainer = shap.Explainer(
            model,
            masker=masker,
            output_names=CLASS_NAMES
        )

        shap_values = explainer(
            img_batch,
            max_evals=50,
            batch_size=10
        ).values

        # -------------------------------
        # SALIENCY MAP
        # -------------------------------
        saliency_map = shap_to_saliency(shap_values, idx)

        # -------------------------------
        # PROCESS SHAP OUTPUT
        # -------------------------------
        shap_list = []

        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 5:
            shap_values = np.squeeze(shap_values, axis=0)

            for i in range(shap_values.shape[-1]):
                s = shap_values[..., i]
                s = np.sum(s, axis=-1, keepdims=True)
                s = np.repeat(s, 3, axis=-1)
                shap_list.append(s)

        else:
            s = np.squeeze(shap_values)
            if s.ndim == 3:
                if s.shape[-1] == 1:
                    s = np.repeat(s, 3, axis=-1)
                shap_list.append(s)

        # -------------------------------
        # SAVE IMAGE
        # -------------------------------
        os.makedirs(save_dir, exist_ok=True)

        name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(save_dir, f"shap_{model_name}_{name}.png")

        # 🔥 IMPORTANT FIX (normalize image for display)
        display_img = img_arr.copy()
        display_img = display_img - display_img.min()
        display_img = display_img / (display_img.max() + 1e-8)

        if display_img.shape[-1] == 1:
            display_img = np.repeat(display_img, 3, axis=-1)

        plt.figure()
        shap.image_plot(shap_list, display_img, show=False)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        return output_path, saliency_map, img_batch
    
    def generate_explanation_metrics(model, img_array, saliency_map, model_name, save_dir):

        import matplotlib.pyplot as plt
        from scipy.stats import spearmanr

        os.makedirs(save_dir, exist_ok=True)

        H, W = img_array.shape[1], img_array.shape[2]

        pred_class_idx = int(np.argmax(model.predict(img_array, verbose=0)))

        # -----------------------------
        # Helper
        # -----------------------------
        def model_confidence(img):
            return float(model.predict(img, verbose=0)[0][pred_class_idx])

        # -----------------------------
        # Deletion Test
        # -----------------------------
        def deletion_test():
            img = img_array.copy()
            saliency = saliency_map.flatten()
            order = np.argsort(-saliency)

            confidences = []
            flat_img = img.reshape(-1, 3)

            for i in range(50):
                k = int((i + 1) / 50 * len(order))
                flat_img[order[:k]] = 0
                step = flat_img.reshape(1, H, W, 3)
                confidences.append(model_confidence(step))

            return confidences

        # -----------------------------
        # Insertion Test
        # -----------------------------
        def insertion_test():
            saliency = saliency_map.flatten()
            order = np.argsort(-saliency)

            blank = np.zeros_like(img_array)
            flat_blank = blank.reshape(-1, 3)
            flat_img = img_array.reshape(-1, 3)

            confidences = []

            for i in range(50):
                k = int((i + 1) / 50 * len(order))
                flat_blank[order[:k]] = flat_img[order[:k]]
                step = flat_blank.reshape(1, H, W, 3)
                confidences.append(model_confidence(step))

            return confidences

        # -----------------------------
        # Sensitivity-N (optional)
        # -----------------------------
        def sensitivity_n():
            flat_sal = saliency_map.flatten()
            flat_img = img_array.reshape(-1, 3)

            scores, deltas = [], []
            baseline = model_confidence(img_array)

            for _ in range(150):
                idx = np.random.choice(len(flat_sal), size=200, replace=False)

                masked = flat_img.copy()
                masked[idx] = 0
                masked_img = masked.reshape(1, H, W, 3)

                conf = model_confidence(masked_img)

                deltas.append(baseline - conf)
                scores.append(flat_sal[idx].sum())

            return spearmanr(scores, deltas).correlation

        # -----------------------------
        # RUN
        # -----------------------------
        deletion_curve = deletion_test()
        insertion_curve = insertion_test()
        sens_n = sensitivity_n()

        # -----------------------------
        # SAVE PLOTS
        # -----------------------------
        import time
        unique_id = str(int(time.time()))

        deletion_path = os.path.join(save_dir, f"deletion_{model_name}_{unique_id}.png")
        insertion_path = os.path.join(save_dir, f"insertion_{model_name}_{unique_id}.png")

        # Deletion plot
        plt.figure()
        plt.plot(deletion_curve)
        plt.title("Deletion Curve")
        plt.xlabel("Steps")
        plt.ylabel("Confidence")
        plt.savefig(deletion_path)
        plt.close()

        # Insertion plot
        plt.figure()
        plt.plot(insertion_curve)
        plt.title("Insertion Curve")
        plt.xlabel("Steps")
        plt.ylabel("Confidence")
        plt.savefig(insertion_path)
        plt.close()

        return {
            "deletion_image": deletion_path,
            "insertion_image": insertion_path,
            "sensitivity_n": float(sens_n)
        }
    TUMOR_PRECAUTIONS = {
    "glioma": [
        "Consult doctor regularly",
        "Consult a neuro-oncologist immediately",
        "Avoid excessive screen exposure",
        "Maintain adequate sleep and hydration",
        "Strictly follow MRI follow-up schedule",
        "Avoid self-medication"
    ],
    "meningioma": [
        "Consult doctor regularly",
        "Regular imaging follow-up",
        "Avoid head trauma",
        "Monitor blood pressure",
        "Report worsening headaches immediately"
    ],
    "pituitary": [
        "Consult doctor regularly",
        "Monitor hormonal changes",
        "Regular endocrinology checkups",
        "Avoid stress",
        "Do not ignore vision changes"
    ],
    "no_tumor": [
        "Maintain a healthy lifestyle",
        "Regular medical checkups",
        "Balanced diet and exercise"
    ]
    }
    
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    
    @app.route("/")
    def home():
        return render_template("home.html")

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":

            # Check if email already exists
            if User.query.filter_by(email=request.form["email"]).first():
                flash("Email already registered. Please login.", "danger")
                return redirect(url_for("register"))

            # Create doctor user (role fixed)
            user = User(
                username=request.form["username"],
                email=request.form["email"],
                role="doctor",
                hospital=request.form.get("hospital"),
                experience_years=int(request.form.get("experience_years", 0)),
            )

            # Set password
            user.set_password(request.form["password"])

            # Save to DB
            db.session.add(user)
            db.session.commit()

            flash("Doctor registration successful", "success")
            return redirect(url_for("login"))

        return render_template("register.html")


    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            ue = request.form["username_or_email"]
            pw = request.form["password"]

            user = User.query.filter(
                (User.username == ue) | (User.email == ue)
            ).first()

            # Doctor-only validation
            if not user or not user.check_password(pw) or user.role != "doctor":
                flash("Invalid doctor credentials", "danger")
                return redirect(request.url)

            login_user(user)
            return redirect(url_for("doctor_dashboard"))

        return render_template("login.html")

    @app.route("/explainability")
    @login_required
    def explainability():
        return render_template("explainability_loading.html")

    @app.route("/doctor")
    @login_required
    def doctor_dashboard():
        return render_template("doctor_dashboard.html")

    @app.route("/profile")
    @login_required
    def profile():
        return render_template("profile.html")

    @app.route("/diagnostic_report", methods=["POST"])
    @login_required
    def diagnostic_report():

            tumor_type = request.form.get("label")
            confidence = request.form.get("confidence")
            orig_url = request.form.get("orig_url")
            result_url = request.form.get("result_url")
            model_used = request.form.get("model_used")

            if not tumor_type or not confidence:
                    flash("Prediction data missing. Please try again.", "danger")
                    return redirect(url_for("doctor_dashboard"))

            precautions = TUMOR_PRECAUTIONS.get(
                    tumor_type.lower(),
                    ["Consult a medical professional immediately"]
            )

            return render_template(
                "report.html",
                tumor_type=tumor_type,
                confidence=confidence,
                orig_url=orig_url,
                result_url=result_url,
                model_used=model_used,
                precautions=precautions,
                doctor_name=current_user.username,   # ✅ ADD
                report_date=datetime.now().strftime("%d %B %Y")  # ✅ ADD
               )
    

    @app.route("/download_report_pdf", methods=["POST"])
    @login_required
    def download_report_pdf():

        tumor_type = request.form.get("tumor_type")
        confidence = request.form.get("confidence")
        orig_url = request.form.get("orig_url")
        result_url = request.form.get("result_url")
        orig_url = request.host_url.rstrip("/") + orig_url
        result_url = request.host_url.rstrip("/") + result_url
        precautions = TUMOR_PRECAUTIONS.get(
            tumor_type.lower(), []
        )

        html = render_template(
            "report.html",
            tumor_type=tumor_type,
            confidence=confidence,
            orig_url=orig_url,
            result_url=result_url,
            precautions=precautions
        )

        options = {
           "page-size": "A4",
           "encoding": "UTF-8",

           "enable-local-file-access": "",

    # 🔥 THESE TWO LINES ARE WHY HEADER IS SHOWING
           "print-media-type": "",
           "background": "",

    # 🔥 THIS PREVENTS IMAGES FROM BEING SKIPPED
           "load-error-handling": "ignore",
           "load-media-error-handling": "ignore",

           "quiet": ""
           }


        pdf = pdfkit.from_string(
            html,
            False,
            options=options,
            configuration=PDF_CONFIG
        )

        response = make_response(pdf)
        response.headers["Content-Type"] = "application/pdf"
        response.headers["Content-Disposition"] = (
            "attachment; filename=diagnostic_report.pdf"
        )

        return response


    @app.route("/upload_predict", methods=["POST"])
    @login_required
    def upload_predict():
        model_name = request.form.get("model_name")
        model = MODEL_REGISTRY.get(model_name)

        file = request.files["image"]
        filename = secure_filename(file.filename)
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(img_path)

        x = preprocess_image(img_path, model_name, model)
        preds = model.predict(x)
        idx = int(np.argmax(preds[0]))

        if model_name == "densenet":
            heatmap = make_gradcam_heatmap_densenet(x, model, idx)
        elif model_name == "efficientnet":
            heatmap = make_gradcam_heatmap(x, model, "efficientnetb4", idx)
        elif model_name == "xception":
            heatmap = make_gradcam_heatmap_xception(x, model, idx)
        elif model_name == "resnet50":
            heatmap = make_gradcam_heatmap_resnet50(x, idx)
        else:
            heatmap = None

        result_name = f"gradcam_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_name)

        if heatmap is not None:
            overlay_gradcam(img_path, heatmap, result_path)
        
        session["last_prediction"] = {
    "filename": filename,
    "model_name": model_name,
    "label": CLASS_NAMES[idx],
    "confidence": float(preds[0][idx]),
    "result_name": result_name
     }
        return render_template(
            "prediction_result.html",
            orig_url=url_for("static", filename=f"uploads/{filename}"),
            result_url=url_for("static", filename=f"results/{result_name}"),
            label=CLASS_NAMES[idx],
            confidence=round(float(preds[0][idx]) * 100, 2),
            model_used=model_name.capitalize(),
        )
        
    @app.route("/run_explainability")
    @login_required
    def run_explainability():

        data = session.get("last_prediction")
        if not data:
            return "No prediction data found"

        img_path = os.path.join("static", "uploads", data["filename"])
        model_name = data["model_name"]
        model = MODEL_REGISTRY[model_name]

        idx = CLASS_NAMES.index(data["label"])

        # SHAP
        try:
            shap_path, saliency_map, img_batch = generate_shap_image(
        img_path,
        model_name,
        idx
    )
        except Exception as e:
            return f"SHAP failed: {str(e)}"

        # METRICS
        metrics = generate_explanation_metrics(
            model,
            img_batch,
            saliency_map,
            model_name,
            "static/results"
        )
        shap_url = url_for(
        "static",
          filename=shap_path.replace("static\\", "").replace("\\", "/")
    )

        deletion_url = url_for(
          "static",
         filename=metrics["deletion_image"].replace("static\\", "").replace("\\", "/")
     )

     insertion_url = url_for(
        "static",
         filename=metrics["insertion_image"].replace("static\\", "").replace("\\", "/")
    )
        return render_template(
    "explainability_result.html",
    shap_url=shap_url,
    deletion_url=deletion_url,
    insertion_url=insertion_url,
    sensitivity=metrics["sensitivity_n"]
    )
    @app.route("/logout")
    @login_required
    def logout():
        logout_user()
        return redirect(url_for("home"))

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
