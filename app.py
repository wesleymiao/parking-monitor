import os
import json
import uuid
import datetime
import logging
import numpy as np
import cv2
import requests as http_requests
from flask import Flask, request, abort, send_from_directory, jsonify

app = Flask(__name__)

DEFAULT_KEY = str(uuid.uuid5(uuid.NAMESPACE_DNS, "parking-monitor"))
API_KEY = os.environ.get("API_KEY", DEFAULT_KEY)
DINGTALK_WEBHOOK = os.environ.get("DINGTALK_WEBHOOK", "")

log = logging.getLogger(__name__)
previous_open_spots = None
if os.environ.get("WEBSITE_SITE_NAME"):
    # Azure App Service — use persistent storage outside the app directory
    UPLOAD_DIR = "/home/uploads"
else:
    # Local development
    UPLOAD_DIR = os.path.join(app.root_path, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
CONFIG_DIR = os.path.join(UPLOAD_DIR, "config")
os.makedirs(CONFIG_DIR, exist_ok=True)
CONFIG_FILE = os.path.join(CONFIG_DIR, "spots.json")
REFERENCE_FILE = os.path.join(CONFIG_DIR, "reference.jpg")
DIFF_THRESHOLD = int(os.environ.get("DIFF_THRESHOLD", "40"))
GMT8 = datetime.timezone(datetime.timedelta(hours=8))
DEPLOY_TIME = datetime.datetime.now(GMT8).strftime("%Y-%m-%d %H:%M:%S")


def load_spots():
    if os.path.isfile(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return []


def save_spots(spots):
    with open(CONFIG_FILE, "w") as f:
        json.dump(spots, f)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/info")
def info():
    return jsonify({"deploy_time": DEPLOY_TIME})


@app.route("/calibrate")
def calibrate():
    return send_from_directory("static", "calibrate.html")


@app.route("/config/reference", methods=["GET", "POST"])
def config_reference():
    key = request.headers.get("X-API-Key") or request.form.get("api_key")
    if key != API_KEY:
        abort(401)
    if request.method == "POST":
        if "file" in request.files:
            data = request.files["file"].read()
        else:
            data = request.data
        with open(REFERENCE_FILE, "wb") as f:
            f.write(data)
        return "OK", 200
    if not os.path.isfile(REFERENCE_FILE):
        return "No reference image", 404
    return send_from_directory(CONFIG_DIR, "reference.jpg")


@app.route("/config/spots", methods=["GET", "POST"])
def config_spots():
    if request.method == "POST":
        key = request.headers.get("X-API-Key") or (request.json.get("api_key") if request.is_json else request.form.get("api_key"))
        if key != API_KEY:
            abort(401)
        spots = request.json.get("spots", [])
        save_spots(spots)
        return jsonify({"count": len(spots)}), 200
    return jsonify(load_spots())


@app.route("/upload", methods=["POST"])
def upload():
    key = request.headers.get("X-API-Key") or request.form.get("api_key")
    if key != API_KEY:
        abort(401)

    # Handle multipart form upload (from HTML form)
    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400
        data = file.read()
    # Handle raw binary upload (from ESP32-CAM)
    elif request.content_type == "image/jpeg":
        data = request.data
    else:
        return "No image provided", 400

    if not data:
        return "Empty image", 400

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"parking_{timestamp}.jpg"
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(data)

    result = detect_open_spots(filepath)
    image_url = f"{request.host_url}images/{filename}"
    notify_if_changed(result, image_url)

    return jsonify({"filename": filename, "size": len(data), "detection": result}), 200


def detect_open_spots(image_path):
    spots = load_spots()
    if not spots or not os.path.isfile(REFERENCE_FILE):
        log.warning("Detection not calibrated — no reference or spots defined")
        return {"total": 0, "open": [], "occupied": [], "error": "not calibrated"}

    ref = cv2.imread(REFERENCE_FILE)
    img = cv2.imread(image_path)
    if ref is None or img is None:
        return {"total": 0, "open": [], "occupied": [], "error": "cannot read images"}

    # Resize img to match reference if needed
    if ref.shape[:2] != img.shape[:2]:
        img = cv2.resize(img, (ref.shape[1], ref.shape[0]))

    # Convert to grayscale and blur to reduce noise
    ref_gray = cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    img_gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    open_spots = []
    occupied_spots = []

    for spot in spots:
        x = int(spot["x"] * ref.shape[1])
        y = int(spot["y"] * ref.shape[0])
        w = int(spot["w"] * ref.shape[1])
        h = int(spot["h"] * ref.shape[0])

        ref_roi = ref_gray[y:y+h, x:x+w]
        img_roi = img_gray[y:y+h, x:x+w]

        if ref_roi.size == 0 or img_roi.size == 0:
            continue

        # Compute mean absolute difference
        diff = cv2.absdiff(ref_roi, img_roi)
        mean_diff = float(np.mean(diff))

        spot_id = spot["id"]
        if mean_diff > DIFF_THRESHOLD:
            occupied_spots.append(spot_id)
        else:
            open_spots.append(spot_id)

        log.info(f"Spot {spot_id}: diff={mean_diff:.1f} threshold={DIFF_THRESHOLD} -> {'occupied' if mean_diff > DIFF_THRESHOLD else 'open'}")

    return {
        "total": len(spots),
        "open": sorted(open_spots),
        "occupied": sorted(occupied_spots),
    }


def notify_if_changed(result, image_url):
    """Send DingTalk notification when open spots change."""
    global previous_open_spots
    current_open = result["open"]

    if previous_open_spots is not None and set(current_open) == set(previous_open_spots):
        log.info("No change in open spots, skipping notification")
        return

    previous_open_spots = current_open

    if not current_open:
        title = "Parking Update"
        text = f"### 🅿️ Parking Update\n\n**No open spots.**\n\n![image]({image_url})"
    else:
        spots = ", ".join(f"#{s}" for s in current_open)
        title = f"{len(current_open)}/{result['total']} spots open"
        text = f"### 🅿️ Parking Update\n\n**{len(current_open)}/{result['total']}** spots open: {spots}\n\n![image]({image_url})"

    log.info(f"Notification: {title}")
    send_dingtalk(title, text)


def send_dingtalk(title, text):
    if not DINGTALK_WEBHOOK:
        log.warning("DingTalk not configured, skipping notification")
        return
    try:
        resp = http_requests.post(DINGTALK_WEBHOOK, json={
            "msgtype": "markdown",
            "markdown": {
                "title": title,
                "text": text,
            },
        }, timeout=10)
        log.info(f"DingTalk response: {resp.status_code} {resp.text}")
    except Exception as e:
        log.error(f"DingTalk send failed: {e}")


@app.route("/images")
def list_images():
    files = sorted(os.listdir(UPLOAD_DIR), reverse=True)
    images = [f for f in files if f.endswith(".jpg")]
    return jsonify(images)


@app.route("/images/<filename>", methods=["GET", "DELETE"])
def get_image(filename):
    if request.method == "DELETE":
        key = request.headers.get("X-API-Key")
        if key != API_KEY:
            abort(401)
        filepath = os.path.join(UPLOAD_DIR, filename)
        if not os.path.isfile(filepath):
            return "Not found", 404
        os.remove(filepath)
        return "Deleted", 200
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/images/latest")
def latest_image():
    files = sorted(os.listdir(UPLOAD_DIR), reverse=True)
    images = [f for f in files if f.endswith(".jpg")]
    if not images:
        return "No images yet", 404
    return send_from_directory(UPLOAD_DIR, images[0])
