import os
import uuid
import datetime
from flask import Flask, request, abort, send_from_directory, jsonify

app = Flask(__name__)

DEFAULT_KEY = str(uuid.uuid5(uuid.NAMESPACE_DNS, "parking-monitor"))
API_KEY = os.environ.get("API_KEY", DEFAULT_KEY)
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

    return jsonify({"filename": filename, "size": len(data)}), 200


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
