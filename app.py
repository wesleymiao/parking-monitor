import os
import json
import uuid
import datetime
import logging
import requests as http_requests
from flask import Flask, request, abort, send_from_directory, jsonify
from detector import detect as detect_vehicles, get_available_models, get_current_model

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
last_daily_notification = None
previous_had_open = False
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
SETTINGS_FILE = os.path.join(CONFIG_DIR, "settings.json")
TIMELINE_FILE = os.path.join(CONFIG_DIR, "timeline.json")
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


def load_settings():
    defaults = {"detect_start": 6, "detect_end": 18, "model": "yolov8l", "confidence": 0.1}
    if os.path.isfile(SETTINGS_FILE):
        with open(SETTINGS_FILE) as f:
            saved = json.load(f)
            defaults.update(saved)
    return defaults


def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f)


def load_timeline():
    if os.path.isfile(TIMELINE_FILE):
        with open(TIMELINE_FILE) as f:
            return json.load(f)
    return []


def save_timeline(timeline):
    with open(TIMELINE_FILE, "w") as f:
        json.dump(timeline, f)


def append_timeline(status, time_str):
    """Append a detection event. Keep last 3 days only."""
    timeline = load_timeline()
    timeline.append({"status": status, "time": time_str})
    # Prune entries older than 3 days
    cutoff = (datetime.datetime.now(GMT8) - datetime.timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S")
    timeline = [e for e in timeline if e["time"] >= cutoff]
    save_timeline(timeline)


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


@app.route("/config/settings", methods=["GET", "POST"])
def config_settings():
    if request.method == "POST":
        key = request.headers.get("X-API-Key") or (request.json.get("api_key") if request.is_json else request.form.get("api_key"))
        if key != API_KEY:
            abort(401)
        settings = load_settings()
        settings["detect_start"] = request.json.get("detect_start", settings["detect_start"])
        settings["detect_end"] = request.json.get("detect_end", settings["detect_end"])
        if "model" in request.json:
            settings["model"] = request.json["model"]
        if "confidence" in request.json:
            settings["confidence"] = float(request.json["confidence"])
        save_settings(settings)
        return jsonify(settings), 200
    settings = load_settings()
    settings["available_models"] = get_available_models()
    settings["current_model"] = get_current_model()
    return jsonify(settings)


@app.route("/timeline")
def timeline():
    """Return timeline segments for the past 3 days."""
    events = load_timeline()
    if not events:
        return jsonify([])

    # Group consecutive same-status events into segments
    segments = []
    current = {"status": events[0]["status"], "start": events[0]["time"], "end": events[0]["time"], "count": 1}
    for e in events[1:]:
        if e["status"] == current["status"]:
            current["end"] = e["time"]
            current["count"] += 1
        else:
            segments.append(current)
            current = {"status": e["status"], "start": e["time"], "end": e["time"], "count": 1}
    segments.append(current)
    return jsonify(segments)


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

    # Validate JPEG: must start with FF D8 and end with FF D9
    if len(data) < 4 or data[:2] != b'\xff\xd8' or data[-2:] != b'\xff\xd9':
        log.warning(f"Discarded incomplete JPEG ({len(data)} bytes)")
        return "Incomplete or invalid JPEG", 400

    import time as time_mod
    now_str = datetime.datetime.now(GMT8).strftime("%Y-%m-%d %H:%M:%S")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    source = "ESP32-CAM" if request.content_type == "image/jpeg" else "Web"

    # Save temp file for detection
    temp_filename = f"parking_{timestamp}.jpg"
    temp_filepath = os.path.join(UPLOAD_DIR, temp_filename)
    with open(temp_filepath, "wb") as f:
        f.write(data)

    log.info(f"Received {temp_filename} ({len(data)} bytes) from {source}")
    settings = load_settings()
    hour = datetime.datetime.now(GMT8).hour
    if settings["detect_start"] <= hour < settings["detect_end"]:
        log.info("Running detection...")
        t0 = time_mod.time()
        model_name = settings.get("model", "yolov8l")
        confidence = settings.get("confidence", 0.1)
        result = detect_open_spots(temp_filepath, model_name, confidence)
        inference_ms = int((time_mod.time() - t0) * 1000)
        log.info(f"Detection result: {result} (inference: {inference_ms}ms)")

        labeled = result.get("labeled_image", temp_filename)
        image_url = f"{request.host_url}images/{labeled}"
        notify_if_changed(result, image_url)

        # Store metadata for labeled image
        meta = load_metadata()
        meta[labeled] = {
            "source": source,
            "model": model_name,
            "open": result.get("open", []),
            "occupied": result.get("occupied", []),
            "time": now_str,
        }
        save_metadata(meta)

        # Record timeline
        has_open = len(result.get("open", [])) > 0
        append_timeline("open" if has_open else "occupied", now_str)

        # Remove original image, keep only labeled
        if labeled != temp_filename and os.path.isfile(temp_filepath):
            os.remove(temp_filepath)
    else:
        log.info(f"Outside detection hours (current: {hour}:00 GMT+8), skipping")
        result = {"skipped": True}
        append_timeline("not_monitored", now_str)
        # Remove temp file since no detection
        if os.path.isfile(temp_filepath):
            os.remove(temp_filepath)

    return jsonify({"filename": temp_filename, "size": len(data), "detection": result}), 200


def detect_open_spots(image_path, model_name="yolov8l", confidence=0.1):
    spots = load_spots()
    if not spots:
        log.warning("Detection not calibrated — no spots defined")
        return {"total": 0, "open": [], "occupied": [], "error": "not calibrated"}
    return detect_vehicles(image_path, spots, model_name=model_name, confidence=confidence)


def notify_if_changed(result, image_url):
    """Send DingTalk notification on state change or daily summary."""
    global last_daily_notification, previous_had_open
    current_open = result["open"]
    today = datetime.datetime.now(GMT8).date()
    has_open = len(current_open) > 0
    state_changed = has_open != previous_had_open
    needs_daily = last_daily_notification != today

    if state_changed and has_open:
        # Transition: no open -> some open
        spots = ", ".join(f"#{s}" for s in current_open)
        title = f"Spots available! {len(current_open)}/{result['total']}"
        text = f"### 🅿️ Spots Available!\n\n**{len(current_open)}/{result['total']}** spots open: {spots}\n\n![image]({image_url})"
        previous_had_open = True
    elif state_changed and not has_open:
        # Transition: some open -> no open
        title = "All spots taken"
        text = f"### 🅿️ All Spots Taken\n\n**All {result['total']} spots are now occupied.**\n\n![image]({image_url})"
        previous_had_open = False
    elif needs_daily:
        # Daily summary
        if has_open:
            spots = ", ".join(f"#{s}" for s in current_open)
            title = f"Daily: {len(current_open)}/{result['total']} open"
            text = f"### 🅿️ Daily Summary\n\n**{len(current_open)}/{result['total']}** spots open: {spots}\n\n![image]({image_url})"
        else:
            title = "Daily: all spots occupied"
            text = f"### 🅿️ Daily Summary\n\n**All {result['total']} spots occupied.** Detection is running normally.\n\n![image]({image_url})"
    else:
        log.info("No state change, skipping notification")
        return

    log.info(f"Notification: {title}")
    send_dingtalk(title, text)
    last_daily_notification = today


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
    filter_type = request.args.get("filter", "all")

    files = sorted(os.listdir(UPLOAD_DIR), reverse=True)
    images = [f for f in files if f.endswith(".jpg")]
    meta = load_metadata()

    # Apply filter
    filtered = []
    for name in images:
        is_labeled = "_labeled" in name
        info = meta.get(name, {})
        if not info:
            original = name.replace("_labeled", "")
            info = meta.get(original, {})
        has_open = len(info.get("open", [])) > 0
        has_occupied = len(info.get("occupied", [])) > 0

        if filter_type == "open" and not (is_labeled and has_open):
            continue
        elif filter_type == "occupied" and not (is_labeled and not has_open and has_occupied):
            continue
        elif filter_type == "original" and is_labeled:
            continue

        filtered.append(name)

    total = len(filtered)
    start = (page - 1) * per_page
    page_images = filtered[start:start + per_page]

    result = []
    for name in page_images:
        is_labeled = "_labeled" in name
        info = meta.get(name, {})
        if not info:
            original = name.replace("_labeled", "")
            info = meta.get(original, {})
        result.append({
            "filename": name,
            "source": info.get("source", "Labeled" if is_labeled else "Unknown"),
            "model": info.get("model", ""),
            "open": info.get("open", []),
            "occupied": info.get("occupied", []),
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
