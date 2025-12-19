# app.py
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from extensions import db, login_manager
from models import User, Message, LastRead
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


# -------------------------------------------------
# APP FACTORY
# -------------------------------------------------
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

    # -------------------------------------------------
    # LOAD MODELS
    # -------------------------------------------------
    def load_model_from_folder(folder):
        with open(os.path.join(folder, "config.json"), "r") as f:
            model = model_from_json(f.read())
        model.load_weights(os.path.join(folder, "model.weights.h5"))
        model.trainable = False
        return model

    MODEL_REGISTRY = {
        "densenet": load_model_from_folder(os.path.join(MODEL_FOLDER, "densenet")),
        "efficientnet": load_model_from_folder(os.path.join(MODEL_FOLDER, "efficient")),
    }

    CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

    # -------------------------------------------------
    # GRAD-CAM
    # -------------------------------------------------
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

    # -------------------------------------------------
    # IMAGE PREPROCESS
    # -------------------------------------------------
    def preprocess_image(img_path, model_name, model):
        h, w = model.input_shape[1], model.input_shape[2]

        if model_name == "densenet":
            img = Image.open(img_path).convert("L").resize((w, h))
            x = np.array(img).astype("float32") / 255.0
            x = np.expand_dims(x, axis=-1)
        else:
            img = Image.open(img_path).convert("RGB").resize((w, h))
            x = np.array(img).astype("float32")
            x = eff_preprocess(x)

        return np.expand_dims(x, axis=0)

    # -------------------------------------------------
    # LOGIN
    # -------------------------------------------------
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # -------------------------------------------------
    # ROUTES
    # -------------------------------------------------
    @app.route("/")
    def home():
        return render_template("home.html")

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            if User.query.filter_by(email=request.form["email"]).first():

               flash("Email already registered. Please login.", "danger")
               return redirect(url_for("register"))
            user = User(
                username=request.form["username"],
                email=request.form["email"],
                role=request.form["role"],
            )

            user.set_password(request.form["password"])
            if request.form["role"] == "doctor":
            
                user.hospital = request.form.get("hospital")
                user.experience_years = int(request.form.get("experience_years", 0))
            db.session.add(user)
            db.session.commit()
            flash("Registration successful", "success")
            return redirect(url_for("login"))
        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            ue = request.form["username_or_email"]
            pw = request.form["password"]
            role = request.form["role"]

            user = User.query.filter(
                (User.username == ue) | (User.email == ue)
            ).first()

            if not user or not user.check_password(pw) or user.role != role:
                flash("Invalid credentials", "danger")
                return redirect(request.url)

            login_user(user)
            return redirect(
                url_for("patient_dashboard") if role == "patient"
                else url_for("doctor_dashboard")
            )

        return render_template("login.html")

    @app.route("/patient")
    @login_required
    def patient_dashboard():
        return render_template("patient_dashboard.html")

    @app.route("/doctor")
    @login_required
    def doctor_dashboard():
        return render_template("doctor_dashboard.html")

    @app.route("/find_doctors")
    @login_required
    def find_doctors():
        doctors = User.query.filter_by(role="doctor").all()
        return render_template("find_doctors.html", doctors=doctors)

    @app.route("/tips")
    @login_required
    def tips():
        return render_template(
            "tips.html",
            tips=[
                "Follow regular medical checkups",
                "Avoid excessive screen time",
                "Maintain proper sleep",
                "Reduce stress",
                "Eat healthy food",
            ],
        )

    @app.route("/send_message/<int:peer_id>", methods=["POST"])
    @login_required
    def send_message(peer_id):
        msg = Message(
            sender_id=current_user.id,
            receiver_id=peer_id,
            content=request.form["content"],
        )
        db.session.add(msg)
        db.session.commit()
        return redirect(url_for("chat", peer_id=peer_id))

    @app.route("/profile")
    @login_required
    def profile():
        return render_template("profile.html")

    @app.route("/chats")
    @login_required
    def chats():
        msgs = Message.query.filter(
            or_(
                Message.sender_id == current_user.id,
                Message.receiver_id == current_user.id,
            )
        ).order_by(Message.timestamp.desc()).all()

        chat_map = {}
        for msg in msgs:
            peer_id = (
                msg.receiver_id
                if msg.sender_id == current_user.id
                else msg.sender_id
            )
            if peer_id not in chat_map:
                chat_map[peer_id] = msg

        chats = []
        for peer_id, last_msg in chat_map.items():
            peer = User.query.get(peer_id)
            chats.append({"peer": peer, "last_message": last_msg})

        return render_template("chats.html", chats=chats)

    @app.route("/chat/<int:peer_id>")
    @login_required
    def chat(peer_id):
        peer = User.query.get_or_404(peer_id)
        messages = Message.query.filter(
            or_(
                and_(
                    Message.sender_id == current_user.id,
                    Message.receiver_id == peer_id,
                ),
                and_(
                    Message.sender_id == peer_id,
                    Message.receiver_id == current_user.id,
                ),
            )
        ).order_by(Message.timestamp.asc()).all()

        return render_template("chat.html", peer=peer, messages=messages)

    # -------------------------------------------------
    # UPLOAD + PREDICT
    # -------------------------------------------------
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
        else:
            heatmap = None

        result_name = f"gradcam_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_name)

        if heatmap is not None:
            overlay_gradcam(img_path, heatmap, result_path)

        return render_template(
            "prediction_result.html",
            orig_url=url_for("static", filename=f"uploads/{filename}"),
            result_url=url_for("static", filename=f"results/{result_name}"),
            label=CLASS_NAMES[idx],
            confidence=round(float(preds[0][idx]) * 100, 2),
            model_used=model_name.capitalize(),
        )

    @app.route("/logout")
    @login_required
    def logout():
        logout_user()
        return redirect(url_for("home"))

    return app


# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
