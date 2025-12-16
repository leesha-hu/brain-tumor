# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
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
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Conv2D


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "devkey"
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///brain_chat.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # ---------- INIT EXTENSIONS ----------
    db.init_app(app)
    login_manager.init_app(app)

    # ---------- PATHS ----------
    BASE_DIR = os.path.dirname(__file__)
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
    MODEL_FOLDER = os.path.join(BASE_DIR, "models")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)

    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

    def allowed_file(filename):
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    # ---------- MODEL REGISTRY ----------
    MODEL_REGISTRY = {}

    def load_model_from_folder(folder_path):
        with open(os.path.join(folder_path, "config.json"), "r") as f:
            model = model_from_json(f.read())

        model.load_weights(os.path.join(folder_path, "model.weights.h5"))
        return model

    MODEL_REGISTRY["densenet"] = load_model_from_folder(
        os.path.join(MODEL_FOLDER, "densenet")
    )

    MODEL_REGISTRY["efficientnet"] = load_model_from_folder(
        os.path.join(MODEL_FOLDER, "efficient")
    )

    # ---------- GRAD-CAM ----------
    def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index):
        last_conv_layer = model.get_layer(last_conv_layer_name)

        grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)

        # ✅ FIX: handle list output
            if isinstance(predictions, list):
                preds = predictions[0]
            else:
                preds = predictions

            loss = preds[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(weights * conv_outputs[0], axis=-1)

        cam = tf.maximum(cam, 0)
        cam /= tf.reduce_max(cam) + 1e-10

        return cam.numpy()


    def overlay_heatmap_on_image(img_path, heatmap, out_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # ---------- LOGIN LOADER ----------
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # ---------- HOME ----------
    @app.route("/")
    def home():
        return render_template("home.html")

    # ---------- REGISTER ----------
    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            username = request.form["username"].strip()
            email = request.form["email"].strip()
            password = request.form["password"]
            role = request.form["role"]

            if role not in ("patient", "doctor"):
                flash("Invalid role selected", "danger")
                return redirect(request.url)

            if User.query.filter_by(email=email).first():
                flash("Email already registered", "danger")
                return redirect(request.url)

            user = User(username=username, email=email, role=role)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()

            flash("Registration successful. Please login.", "success")
            return redirect(url_for("login"))

        return render_template("register.html")

    # ---------- LOGIN ----------
    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            ue = request.form["username_or_email"]
            pw = request.form["password"]
            role = request.form["role"]

            user = User.query.filter(
                (User.username == ue) | (User.email == ue)
            ).first()

            if not user or not user.check_password(pw):
                flash("Invalid credentials", "danger")
                return redirect(request.url)

            if user.role != role:
                flash("Incorrect role selected", "danger")
                return redirect(request.url)

            login_user(user)
            return redirect(
                url_for("patient_dashboard") if user.role == "patient"
                else url_for("doctor_dashboard")
            )

        return render_template("login.html")

    # ---------- FIND DOCTORS ----------
    @app.route("/find_doctors")
    @login_required
    def find_doctors():
        doctors = User.query.filter_by(role="doctor").all()
        return render_template("find_doctors.html", doctors=doctors)

    # ---------- DASHBOARDS ----------
    @app.route("/patient")
    @login_required
    def patient_dashboard():
        return render_template("patient_dashboard.html")

    @app.route("/doctor")
    @login_required
    def doctor_dashboard():
        return render_template("doctor_dashboard.html")

    # ---------- TIPS ----------
    @app.route("/tips")
    @login_required
    def tips():
        tips_list = [
            "Follow regular medical checkups",
            "Avoid excessive screen time",
            "Maintain a healthy sleep schedule",
            "Consult doctors before taking medication",
            "Reduce stress and follow a balanced diet"
        ]
        return render_template("tips.html", tips=tips_list)

    # ---------- CHAT LIST ----------
    @app.route("/chats")
    @login_required
    def chats():
        peers = User.query.filter(User.role != current_user.role).all()
        chat_list = []

        for p in peers:
            last_msg = Message.query.filter(
                ((Message.sender_id == current_user.id) & (Message.receiver_id == p.id)) |
                ((Message.sender_id == p.id) & (Message.receiver_id == current_user.id))
            ).order_by(Message.timestamp.desc()).first()

            if not last_msg:
                continue

            lr = LastRead.query.filter_by(user_id=current_user.id, peer_id=p.id).first()
            unread = Message.query.filter(
                Message.sender_id == p.id,
                Message.receiver_id == current_user.id,
                Message.timestamp > (lr.last_read if lr else datetime.min)
            ).count()

            chat_list.append({
                "peer": p,
                "last": last_msg.timestamp,
                "unread": unread
            })

        chat_list.sort(key=lambda x: x["last"], reverse=True)
        return render_template("chats.html", chats=chat_list)

    # ---------- CHAT ----------
    @app.route("/chat/<int:peer_id>")
    @login_required
    def chat(peer_id):
        peer = User.query.get_or_404(peer_id)

        lr = LastRead.query.filter_by(
            user_id=current_user.id,
            peer_id=peer_id
        ).first()

        if lr:
            lr.last_read = datetime.utcnow()
        else:
            db.session.add(
                LastRead(
                    user_id=current_user.id,
                    peer_id=peer_id,
                    last_read=datetime.utcnow()
                )
            )

        db.session.commit()
        return render_template("chat.html", peer=peer)

    # ---------- UPLOAD + PREDICT ----------
    @app.route("/upload_predict", methods=["POST"])
    @login_required
    def upload_predict():
        model_name = request.form.get("model_name")
        model = MODEL_REGISTRY.get(model_name)

        if model is None:
            flash("Selected model not available", "danger")
            return redirect(request.referrer)

        file = request.files.get("image")
        if not file or file.filename == "":
            flash("No file selected", "danger")
            return redirect(request.referrer)

        if not allowed_file(file.filename):
            flash("Invalid file type", "danger")
            return redirect(request.referrer)

        filename = secure_filename(
            f"{current_user.id}_{int(datetime.utcnow().timestamp())}_{file.filename}"
        )
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        h, w = model.input_shape[1], model.input_shape[2]
        img = Image.open(save_path).convert("L").resize((w, h))

        x = np.array(img).astype("float32") / 255.0
        x = np.expand_dims(x, axis=-1)
        x = np.expand_dims(x, axis=0)

        preds = model.predict(x)
        idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][idx])

        CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
        label = CLASS_NAMES[idx]

        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, Conv2D):
                last_conv_layer = layer.name
                break

        heatmap = make_gradcam_heatmap(x, model, last_conv_layer, idx)

        result_name = f"gradcam_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_name)
        overlay_heatmap_on_image(save_path, heatmap, result_path)

        return render_template(
            "prediction_result.html",
            orig_url=url_for("static", filename=f"uploads/{filename}"),
            result_url=url_for("static", filename=f"results/{result_name}"),
            label=label,
            confidence=round(confidence * 100, 2),
            model_used=model_name.capitalize()
        )

    # ---------- PROFILE ----------
    @app.route("/profile")
    @login_required
    def profile():
        return render_template("profile.html")

    # ---------- LOGOUT ----------
    @app.route("/logout")
    @login_required
    def logout():
        logout_user()
        return redirect(url_for("home"))

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
