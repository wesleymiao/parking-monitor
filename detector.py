"""
Parking spot detection using YOLOv8n ONNX model.
Detects vehicles and checks overlap with calibrated spot regions.
"""

import os
import logging
import numpy as np
import cv2
import onnxruntime as ort

log = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "yolov8n.onnx")

# COCO classes that are vehicles
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

INPUT_SIZE = 640
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
OVERLAP_THRESHOLD = 0.3  # fraction of spot area that must be covered by a vehicle

_session = None


def _get_session():
    global _session
    if _session is None:
        _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return _session


def _preprocess(img):
    """Resize and normalize image for YOLOv8 input."""
    h, w = img.shape[:2]
    scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    # Pad to 640x640
    padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    # HWC -> NCHW, normalize to 0-1
    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    return blob, scale, 0, 0  # scale, pad_x, pad_y


def _postprocess(output, scale):
    """Extract vehicle detections from YOLOv8 output."""
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

        if confidence < CONF_THRESHOLD:
            continue
        if class_id not in VEHICLE_CLASSES:
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
    indices = cv2.dnn.NMSBoxes(nms_boxes, scores_arr.tolist(), CONF_THRESHOLD, IOU_THRESHOLD)

    results = []
    for i in np.array(indices).flatten():
        results.append({
            "box": boxes[i],
            "score": scores[i],
            "class": VEHICLE_CLASSES[class_ids[i]],
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


def detect(image_path, spots):
    """
    Detect open/occupied parking spots.

    Args:
        image_path: path to the image file
        spots: list of spot dicts with id, x, y, w, h (normalized 0-1)

    Returns:
        dict with total, open, occupied lists and vehicle detections
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"total": 0, "open": [], "occupied": [], "error": "cannot read image"}

    h, w = img.shape[:2]
    session = _get_session()

    # Run inference
    blob, scale, _, _ = _preprocess(img)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: blob})
    vehicles = _postprocess(output[0], scale)

    log.info(f"Detected {len(vehicles)} vehicles: {[v['class'] for v in vehicles]}")

    open_spots = []
    occupied_spots = []

    for spot in spots:
        sx1 = spot["x"] * w
        sy1 = spot["y"] * h
        sx2 = sx1 + spot["w"] * w
        sy2 = sy1 + spot["h"] * h
        spot_box = [sx1, sy1, sx2, sy2]

        is_occupied = False
        for v in vehicles:
            overlap = _box_overlap(spot_box, v["box"])
            if overlap >= OVERLAP_THRESHOLD:
                is_occupied = True
                log.info(f"Spot {spot['id']}: occupied ({v['class']}, overlap={overlap:.2f})")
                break

        if is_occupied:
            occupied_spots.append(spot["id"])
        else:
            open_spots.append(spot["id"])

    # Draw labeled image
    labeled_path = image_path.replace(".jpg", "_labeled.jpg")
    _draw_labels(img, spots, vehicles, open_spots, occupied_spots, labeled_path)

    return {
        "total": len(spots),
        "open": sorted(open_spots),
        "occupied": sorted(occupied_spots),
        "vehicles": len(vehicles),
        "labeled_image": os.path.basename(labeled_path),
    }


def _draw_labels(img, spots, vehicles, open_ids, occupied_ids, output_path):
    """Draw spot regions and vehicle boxes on the image."""
    labeled = img.copy()
    h, w = labeled.shape[:2]

    # Draw vehicle bounding boxes
    for v in vehicles:
        x1, y1, x2, y2 = [int(c) for c in v["box"]]
        cv2.rectangle(labeled, (x1, y1), (x2, y2), (0, 165, 255), 2)
        label = f"{v['class']} {v['score']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(labeled, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 165, 255), -1)
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

    cv2.imwrite(output_path, labeled)
