"""
Parking spot detection using YOLOv8n ONNX model.
Detects vehicles and checks overlap with calibrated spot regions.
"""

import os
import logging
import numpy as np
import cv2
import onnxruntime as ort
import requests as dl_requests

log = logging.getLogger(__name__)

if os.environ.get("WEBSITE_SITE_NAME"):
    MODEL_DIR = "/home/model"
else:
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

BLOB_BASE = "https://parkingyolomodels.blob.core.windows.net/models"
AVAILABLE_MODELS = {
    "yolov8n": f"{BLOB_BASE}/yolov8n.onnx",
    "yolov8s": f"{BLOB_BASE}/yolov8s.onnx",
    "yolov8m": f"{BLOB_BASE}/yolov8m.onnx",
    "yolov8l": f"{BLOB_BASE}/yolov8l.onnx",
    "yolo11m": f"{BLOB_BASE}/yolo11m.onnx",
    "yolo11l": f"{BLOB_BASE}/yolo11l.onnx",
}

# COCO classes that are vehicles
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 8: "boat"}

COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
    49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush",
}

INPUT_SIZE = 640
CONF_THRESHOLD = 0.1
IOU_THRESHOLD = 0.45
OVERLAP_THRESHOLD = 0.0  # any overlap means occupied

# Azure Computer Vision
AZURE_CV_ENABLED = False  # set to True to re-enable Azure CV dual detection
AZURE_CV_ENDPOINT = os.environ.get("AZURE_CV_ENDPOINT", "")
AZURE_CV_KEY = os.environ.get("AZURE_CV_KEY", "")
AZURE_CV_VEHICLE_TAGS = {"car", "truck", "bus", "motorcycle", "vehicle", "van", "suv", "taxi", "minivan",
                         "land vehicle", "wheeled vehicle", "automobile", "motor vehicle"}

_session = None
_current_model = None


def _download_model(model_name):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{model_name}.onnx")
    if os.path.isfile(model_path):
        return model_path
    url = AVAILABLE_MODELS[model_name]
    log.info(f"Downloading {model_name} from {url}...")
    resp = dl_requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(model_path + ".tmp", "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    os.rename(model_path + ".tmp", model_path)
    log.info(f"Model saved ({os.path.getsize(model_path)} bytes)")
    return model_path


def _get_session(model_name="yolov8l"):
    global _session, _current_model
    if _session is None or _current_model != model_name:
        model_path = _download_model(model_name)
        _session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        _current_model = model_name
        log.info(f"Loaded model: {model_name}")
    return _session


def get_current_model():
    return _current_model or "yolov8l"


def get_available_models():
    return list(AVAILABLE_MODELS.keys())


def _preprocess(img):
    """Resize and normalize image for YOLOv8 input."""
    h, w = img.shape[:2]
    scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Pad to 640x640
    padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    # HWC -> NCHW, normalize to 0-1
    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    return blob, scale, 0, 0  # scale, pad_x, pad_y


def _postprocess(output, scale, conf_threshold=CONF_THRESHOLD, vehicle_only=True):
    """Extract detections from YOLOv8 output."""
    # output shape: [1, 84, 8400] -> transpose to [8400, 84]
    predictions = output[0].transpose()

    boxes = []
    scores = []
    class_ids = []

    for pred in predictions:
        x_center, y_center, width, height = pred[:4]
        class_scores = pred[4:]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence < conf_threshold:
            continue
        if vehicle_only and class_id not in VEHICLE_CLASSES:
            continue

        # Convert from center to corner format, scale back to original
        x1 = (x_center - width / 2) / scale
        y1 = (y_center - height / 2) / scale
        x2 = (x_center + width / 2) / scale
        y2 = (y_center + height / 2) / scale

        boxes.append([x1, y1, x2, y2])
        scores.append(confidence)
        class_ids.append(class_id)

    if not boxes:
        return []

    # NMS
    boxes_arr = np.array(boxes, dtype=np.float32)
    scores_arr = np.array(scores, dtype=np.float32)
    # Convert to x, y, w, h for cv2.dnn.NMSBoxes
    nms_boxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
    indices = cv2.dnn.NMSBoxes(nms_boxes, scores_arr.tolist(), conf_threshold, IOU_THRESHOLD)

    class_map = VEHICLE_CLASSES if vehicle_only else COCO_CLASSES
    results = []
    for i in np.array(indices).flatten():
        results.append({
            "box": boxes[i],
            "score": scores[i],
            "class": class_map.get(class_ids[i], f"class_{class_ids[i]}"),
        })

    return results


def _box_overlap(spot_box, vehicle_box):
    """Compute fraction of spot area covered by vehicle."""
    sx1, sy1, sx2, sy2 = spot_box
    vx1, vy1, vx2, vy2 = vehicle_box

    ix1 = max(sx1, vx1)
    iy1 = max(sy1, vy1)
    ix2 = min(sx2, vx2)
    iy2 = min(sy2, vy2)

    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    spot_area = (sx2 - sx1) * (sy2 - sy1)
    if spot_area <= 0:
        return 0.0
    return intersection / spot_area


def _box_overlap_iou(box_a, box_b):
    """Compute IoU between two boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return intersection / (area_a + area_b - intersection)


def _detect_azure_cv(image_path):
    """Detect vehicles using Azure Computer Vision 4.0. Returns list of detections or empty list on failure."""
    if not AZURE_CV_ENDPOINT or not AZURE_CV_KEY:
        return []

    try:
        with open(image_path, "rb") as f:
            image_data = f.read()

        url = f"{AZURE_CV_ENDPOINT.rstrip('/')}/computervision/imageanalysis:analyze?api-version=2024-02-01&features=objects"
        resp = dl_requests.post(
            url,
            headers={
                "Ocp-Apim-Subscription-Key": AZURE_CV_KEY,
                "Content-Type": "application/octet-stream",
            },
            data=image_data,
            timeout=30,
        )

        if resp.status_code != 200:
            log.warning(f"Azure CV returned {resp.status_code}: {resp.text[:200]}")
            return []

        result = resp.json()
        log.info(f"Azure CV response keys: {list(result.keys())}")
        if "objectsResult" not in result:
            log.warning(f"Azure CV response (no objectsResult): {str(result)[:500]}")
        all_objects = result.get("objectsResult", {}).get("values", [])
        log.info(f"Azure CV raw: {len(all_objects)} objects: {[o.get('tags', [{}])[0].get('name', '?') for o in all_objects]}")
        vehicles = []
        for obj in all_objects:
            tags = obj.get("tags", [])
            if not tags:
                continue
            tag_name = tags[0]["name"].lower()
            tag_conf = tags[0]["confidence"]
            if tag_name in AZURE_CV_VEHICLE_TAGS:
                bb = obj["boundingBox"]
                vehicles.append({
                    "box": [bb["x"], bb["y"], bb["x"] + bb["w"], bb["y"] + bb["h"]],
                    "score": tag_conf,
                    "class": f"az:{tag_name}",
                })

        log.info(f"Azure CV detected {len(vehicles)} vehicles: {[v['class'] for v in vehicles]}")
        return vehicles

    except Exception as e:
        log.warning(f"Azure CV failed: {e}")
        return []


def detect(image_path, spots, model_name=None, confidence=None):
    """
    Detect open/occupied parking spots.

    Args:
        image_path: path to the image file
        spots: list of spot dicts with id, x, y, w, h (normalized 0-1)
        model_name: which YOLO model to use
        confidence: confidence threshold (default CONF_THRESHOLD)

    Returns:
        dict with total, open, occupied lists and vehicle detections
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"total": 0, "open": [], "occupied": [], "error": "cannot read image"}

    h, w = img.shape[:2]
    session = _get_session(model_name or "yolov8l")

    # Run YOLO inference
    input_name = session.get_inputs()[0].name
    conf = confidence if confidence is not None else CONF_THRESHOLD

    blob, scale, _, _ = _preprocess(img)
    output = session.run(None, {input_name: blob})
    vehicles = _postprocess(output[0], scale, conf)

    log.info(f"YOLO detected {len(vehicles)} vehicles: {[v['class'] for v in vehicles]}")

    # Run Azure Computer Vision and merge results
    if AZURE_CV_ENABLED:
        azure_vehicles = _detect_azure_cv(image_path)
    else:
        azure_vehicles = []
        log.info("Azure CV disabled, skipping")
    all_vehicles = vehicles + azure_vehicles

    log.info(f"Total detections (YOLO only): {len(all_vehicles)}")

    open_spots = []
    occupied_spots = []

    for spot in spots:
        if spot.get("enabled") is False:
            log.info(f"Spot {spot['id']}: disabled, skipping")
            continue
        sx1 = spot["x"] * w
        sy1 = spot["y"] * h
        sx2 = sx1 + spot["w"] * w
        sy2 = sy1 + spot["h"] * h
        spot_box = [sx1, sy1, sx2, sy2]

        is_occupied = False
        for v in all_vehicles:
            if _box_overlap(spot_box, v["box"]) > 0:
                is_occupied = True
                log.info(f"Spot {spot['id']}: occupied ({v['class']}, overlap={_box_overlap(spot_box, v['box']):.2f})")
                break

        if is_occupied:
            occupied_spots.append(spot["id"])
        else:
            open_spots.append(spot["id"])
            log.info(f"Spot {spot['id']}: open")

    # Draw labeled image
    labeled_path = image_path.replace(".jpg", "_labeled.jpg")
    _draw_labels(img, spots, all_vehicles, open_spots, occupied_spots, labeled_path)

    return {
        "total": len(spots),
        "open": sorted(open_spots),
        "occupied": sorted(occupied_spots),
        "vehicles": len(all_vehicles),
        "labeled_image": os.path.basename(labeled_path),
    }


def _draw_debug(img, objects, output_path, variant_name=""):
    """Draw ALL detected objects for debugging."""
    labeled = img.copy()
    colors = {}
    for obj in objects:
        cls = obj["class"]
        if cls not in colors:
            h = hash(cls) % 180
            colors[cls] = (int(50 + h * 0.8), int(100 + (h * 3) % 155), int(200 - h % 150))
        color = colors[cls]
        x1, y1, x2, y2 = [int(c) for c in obj["box"]]
        cv2.rectangle(labeled, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} {obj['score']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(labeled, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(labeled, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if variant_name:
        cv2.putText(labeled, f"DEBUG: {variant_name} ({len(objects)} objects)", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(output_path, labeled)


def _draw_labels(img, spots, vehicles, open_ids, occupied_ids, output_path, variant_name=""):
    """Draw spot regions and vehicle boxes on the image."""
    labeled = img.copy()
    h, w = labeled.shape[:2]

    # Draw vehicle bounding boxes (orange for YOLO, blue for Azure CV)
    for v in vehicles:
        x1, y1, x2, y2 = [int(c) for c in v["box"]]
        is_azure = v["class"].startswith("az:")
        color = (255, 140, 0) if is_azure else (0, 165, 255)
        cv2.rectangle(labeled, (x1, y1), (x2, y2), color, 2)
        label = f"{v['class']} {v['score']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(labeled, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(labeled, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw open spot regions only
    for spot in spots:
        spot_id = spot["id"]
        if spot_id not in open_ids:
            continue

        sx1 = int(spot["x"] * w)
        sy1 = int(spot["y"] * h)
        sx2 = sx1 + int(spot["w"] * w)
        sy2 = sy1 + int(spot["h"] * h)

        color = (0, 200, 0)
        cv2.rectangle(labeled, (sx1, sy1), (sx2, sy2), color, 3)
        overlay = labeled.copy()
        cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), color, -1)
        cv2.addWeighted(overlay, 0.15, labeled, 0.85, 0, labeled)

        label = f"#{spot_id} OPEN"
        cv2.putText(labeled, label, (sx1 + 4, sy2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw variant name in top-left corner
    if variant_name:
        label = f"variant: {variant_name}"
        cv2.putText(labeled, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(output_path, labeled)
