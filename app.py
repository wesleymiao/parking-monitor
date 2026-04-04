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
DINGTALK_WEBHOOK = os.environ.get("DINGTALK_WEBHOOK", "")
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
GMT8 = datetime.timezone(datetime.timedelta(hours=8))
DEPLOY_TIME = datetime.datetime.now(GMT8).strftime("%Y-%m-%d %H:%M:%S")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/info")
def info():
    return jsonify({"deploy_time": DEPLOY_TIME})


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
    """Mock detection — returns random results. Replace with real CV later."""
    open_spots = random.sample(range(1, TOTAL_SPOTS + 1), random.randint(0, TOTAL_SPOTS))
    open_spots.sort()
    return {
        "total": TOTAL_SPOTS,
        "open": open_spots,
        "occupied": [s for s in range(1, TOTAL_SPOTS + 1) if s not in open_spots],
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
