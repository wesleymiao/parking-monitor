import os
import json
import uuid
import datetime
import logging
import requests as http_requests
from flask import Flask, request, abort, send_from_directory, jsonify
from detector import detect as detect_vehicles

import sys

log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.addFilter(lambda r: r.levelno < logging.ERROR)
stdout_handler.setFormatter(logging.Formatter(log_format))

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.ERROR)
stderr_handler.setFormatter(logging.Formatter(log_format))

logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, stderr_handler])

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
METADATA_FILE = os.path.join(CONFIG_DIR, "metadata.json")
GMT8 = datetime.timezone(datetime.timedelta(hours=8))
DEPLOY_TIME = datetime.datetime.now(GMT8).strftime("%Y-%m-%d %H:%M:%S")


def load_metadata():
    if os.path.isfile(METADATA_FILE):
        with open(METADATA_FILE) as f:
            return json.load(f)
    return {}


def save_metadata(meta):
    with open(METADATA_FILE, "w") as f:
        json.dump(meta, f)


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
    log.info(f"POST /upload from {request.remote_addr}, content_type={request.content_type}")
    key = request.headers.get("X-API-Key") or request.form.get("api_key")
    if key != API_KEY:
        log.warning(f"401 Unauthorized from {request.remote_addr}")
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

    meta = load_metadata()
    source = "ESP32-CAM" if request.content_type == "image/jpeg" else "Web"
    meta[filename] = {
        "source": source,
        "ip": request.remote_addr,
        "time": datetime.datetime.now(GMT8).strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_metadata(meta)

    log.info(f"Saved {filename} ({len(data)} bytes) from {source}, running detection...")
    result = detect_open_spots(filepath)
    log.info(f"Detection result: {result}")
    labeled = result.get("labeled_image", filename)
    image_url = f"{request.host_url}images/{labeled}"
    notify_if_changed(result, image_url)

    return jsonify({"filename": filename, "size": len(data), "detection": result}), 200


def detect_open_spots(image_path):
    spots = load_spots()
    if not spots:
        log.warning("Detection not calibrated — no spots defined")
        return {"total": 0, "open": [], "occupied": [], "error": "not calibrated"}
    return detect_vehicles(image_path, spots)


def notify_if_changed(result, image_url):
    """Send DingTalk notification for every upload."""
    current_open = result["open"]

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
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 100, type=int)

    files = sorted(os.listdir(UPLOAD_DIR), reverse=True)
    images = [f for f in files if f.endswith(".jpg")]
    total = len(images)
    start = (page - 1) * per_page
    page_images = images[start:start + per_page]

    meta = load_metadata()
    result = []
    for name in page_images:
        is_labeled = "_labeled" in name
        original = name.replace("_labeled", "")
        info = meta.get(original, {})
        result.append({
            "filename": name,
            "source": "Labeled" if is_labeled else info.get("source", "Unknown"),
            "time": info.get("time", ""),
        })
    return jsonify({"images": result, "page": page, "total": total, "pages": (total + per_page - 1) // per_page})


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
