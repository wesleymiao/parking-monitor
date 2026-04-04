import os
import uuid
import random
import datetime
import logging
import requests as http_requests
from flask import Flask, request, abort, send_from_directory, jsonify

app = Flask(__name__)

DEFAULT_KEY = str(uuid.uuid5(uuid.NAMESPACE_DNS, "parking-monitor"))
API_KEY = os.environ.get("API_KEY", DEFAULT_KEY)
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
TOTAL_SPOTS = int(os.environ.get("TOTAL_SPOTS", "6"))

log = logging.getLogger(__name__)
previous_open_spots = None
if os.environ.get("WEBSITE_SITE_NAME"):
    # Azure App Service — use persistent storage outside the app directory
    UPLOAD_DIR = "/home/uploads"
else:
    # Local development
    UPLOAD_DIR = os.path.join(app.root_path, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


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
    notify_if_changed(result, filename)

    return jsonify({"filename": filename, "size": len(data), "detection": result}), 200


def detect_open_spots(image_path):
    """Mock detection — returns random results. Replace with real CV later."""
    open_spots = random.sample(range(1, TOTAL_SPOTS + 1), random.randint(0, TOTAL_SPOTS))
    open_spots.sort()
    return {
        "total": TOTAL_SPOTS,
        "open": open_spots,
        "occupied": [s for s in range(1, TOTAL_SPOTS + 1) if s not in open_spots],
    }


def notify_if_changed(result, filename):
    """Send Telegram notification when open spots change."""
    global previous_open_spots
    current_open = result["open"]

    if previous_open_spots is not None and set(current_open) == set(previous_open_spots):
        log.info("No change in open spots, skipping notification")
        return

    previous_open_spots = current_open

    if not current_open:
        message = f"🅿️ Parking Update ({filename})\nNo open spots."
    else:
        spots = ", ".join(f"#{s}" for s in current_open)
        message = f"🅿️ Parking Update ({filename})\n{len(current_open)}/{result['total']} spots open: {spots}"

    log.info(f"Notification: {message}")
    send_telegram(message)


def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram not configured, skipping notification")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = http_requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
        }, timeout=10)
        log.info(f"Telegram response: {resp.status_code}")
    except Exception as e:
        log.error(f"Telegram send failed: {e}")


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
