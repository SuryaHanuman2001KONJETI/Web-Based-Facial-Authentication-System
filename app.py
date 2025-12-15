import os
import json
import base64
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect

app = Flask(__name__)
app.secret_key = "face_auth_secret_key"

# ----- PATHS -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FACES_DIR = os.path.join(DATA_DIR, "faces")
STATIC_FACES_DIR = os.path.join(BASE_DIR, "static", "faces")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
MODEL_FILE = os.path.join(DATA_DIR, "face_model.xml")
LABELS_FILE = os.path.join(DATA_DIR, "labels.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(STATIC_FACES_DIR, exist_ok=True)

if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump({}, f)

# ----- OPENCV MODELS -----
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
recognizer = cv2.face.LBPHFaceRecognizer_create()


# ----- UTILS -----

def load_users():
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(data):
    with open(USERS_FILE, "w") as f:
        json.dump(data, f, indent=4)


def load_labels():
    if not os.path.exists(LABELS_FILE):
        return {}
    with open(LABELS_FILE, "r") as f:
        return json.load(f)


def save_labels(label_map):
    with open(LABELS_FILE, "w") as f:
        json.dump(label_map, f, indent=4)


def decode_image(data_url: str):
    """Convert data:image/png;base64,... to OpenCV BGR image."""
    if "," in data_url:
        encoded = data_url.split(",", 1)[1]
    else:
        encoded = data_url
    binary = base64.b64decode(encoded)
    arr = np.frombuffer(binary, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def preprocess_face(img):
    """Detect first face, crop and resize."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(80, 80)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face_roi = gray[y:y + h, x:x + w]
    resized = cv2.resize(face_roi, (200, 200))
    return resized


def train_model():
    """Train LBPH model and label map from all registered users."""
    users = load_users()
    if not users:
        return

    images = []
    labels = []
    label_map = {}

    # Keep a stable label order
    usernames = sorted(users.keys())
    for idx, username in enumerate(usernames):
        face_file = users[username]["face"]
        img_path = os.path.join(FACES_DIR, face_file)
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        images.append(img)
        labels.append(idx)
        label_map[str(idx)] = username

    if not images:
        return

    recognizer.train(images, np.array(labels))
    recognizer.write(MODEL_FILE)
    save_labels(label_map)


# ----- ROUTES (PAGES) -----

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register")
def register_page():
    return render_template("register.html")


@app.route("/login")
def login_page():
    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/login")
    return render_template("dashboard.html", user=session["user"])


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# ----- API: REGISTER -----

@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json()
    name = (data.get("name") or "").strip()
    age = (data.get("age") or "").strip()
    email = (data.get("email") or "").strip()
    image_data = data.get("image")

    if not name or not age or not email or not image_data:
        return jsonify({"ok": False, "error": "All fields and face are required."}), 400

    users = load_users()
    if name in users:
        return jsonify({"ok": False, "error": "This user is already registered ❌"}), 400

    img = decode_image(image_data)
    face = preprocess_face(img)
    if face is None:
        return jsonify({"ok": False, "error": "Face not detected. Move closer and try again."}), 400

    # Save face image
    safe_name = "".join(c for c in name if c.isalnum() or c in ("_", "-"))
    filename = f"{safe_name}.jpg"
    cv2.imwrite(os.path.join(FACES_DIR, filename), face)
    cv2.imwrite(os.path.join(STATIC_FACES_DIR, filename), face)

    # Save user info
    users[name] = {
        "name": name,
        "age": age,
        "email": email,
        "face": filename
    }
    save_users(users)

    # Retrain model
    train_model()

    return jsonify({"ok": True, "message": "User registered successfully ✅"})


# ----- API: LOGIN (WITH LIVENESS) -----

@app.route("/api/login", methods=["POST"])
def api_login():
    users = load_users()
    if not users or not os.path.exists(MODEL_FILE):
        return jsonify({"ok": False, "error": "No registered users yet."}), 400

    data = request.get_json()
    img1_data = data.get("image1")
    img2_data = data.get("image2")

    if not img1_data or not img2_data:
        return jsonify({"ok": False, "error": "Missing camera frames."}), 400

    img1 = decode_image(img1_data)
    img2 = decode_image(img2_data)

    # Liveness: movement between frames
    diff = cv2.absdiff(img1, img2)
    mean_diff = float(diff.mean())
    if mean_diff < 7.0:
        return jsonify({
            "ok": False,
            "error": "Liveness failed. Please blink or move your head clearly."
        }), 403

    face = preprocess_face(img2)
    if face is None:
        return jsonify({"ok": False, "error": "Face not detected. Try again."}), 400

    # Load model + label map
    recognizer.read(MODEL_FILE)
    label_map = load_labels()
    users_dict = load_users()

    label, confidence = recognizer.predict(face)
    username = label_map.get(str(label))

    if username and confidence < 80.0:
        user = users_dict[username]
        user["last_login"] = datetime.now().strftime("%d %b %Y %H:%M")
        session["user"] = user
        # Fake match percentage just for UI feel
        match_percent = max(50, int(100 - min(confidence, 90)))
        return jsonify({"ok": True, "match": match_percent})

    return jsonify({"ok": False, "error": "Face not recognized."}), 401


if __name__ == "__main__":
    app.run(debug=True)
