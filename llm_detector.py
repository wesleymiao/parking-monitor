"""
Parking spot detection using Azure OpenAI GPT-4.1 vision API.
Draws labeled bounding boxes on the image and asks the LLM to determine occupancy.
"""

import os
import re
import json
import base64
import logging
import cv2
import numpy as np

log = logging.getLogger(__name__)

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")


def _draw_spot_labels(img, spots):
    """Draw clearly labeled bounding boxes for each enabled spot."""
    labeled = img.copy()
    h, w = labeled.shape[:2]

    for spot in spots:
        if spot.get("enabled") is False:
            continue
        sid = spot["id"]
        sx1 = int(spot["x"] * w)
        sy1 = int(spot["y"] * h)
        sx2 = sx1 + int(spot["w"] * w)
        sy2 = sy1 + int(spot["h"] * h)

        # Bright yellow rectangle
        cv2.rectangle(labeled, (sx1, sy1), (sx2, sy2), (0, 255, 255), 3)

        # Label with spot ID
        label = f"#{sid}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.9
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        cv2.rectangle(labeled, (sx1, sy1 - th - 10), (sx1 + tw + 8, sy1), (0, 0, 0), -1)
        cv2.putText(labeled, label, (sx1 + 4, sy1 - 6), font, scale, (0, 255, 255), thickness)

    return labeled


def _encode_image_base64(img):
    """Encode a cv2 image as base64 JPEG."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


def _call_openai_vision(image_b64, spot_ids, deployment=None):
    """Send the annotated image to Azure OpenAI and get occupancy results."""
    from openai import AzureOpenAI

    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version="2024-12-01-preview",
    )

    spot_list = ", ".join(f"#{s}" for s in spot_ids)

    response = client.chat.completions.create(
        model=deployment or AZURE_OPENAI_DEPLOYMENT,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                    },
                },
                {
                    "type": "text",
                    "text": (
                        f"This image shows a parking lot with labeled parking spots: {spot_list}. "
                        "Each spot is outlined with a yellow rectangle and labeled with its ID number. "
                        "For each labeled spot, determine whether it is OCCUPIED (a vehicle is parked in it) or OPEN (empty/no vehicle). "
                        'Respond with ONLY valid JSON: {"spots": [{"id": 1, "status": "occupied"}, {"id": 2, "status": "open"}, ...]}'
                    ),
                },
            ],
        }],
    )

    return response.choices[0].message.content


def _parse_response(response_text, spot_ids):
    """Parse LLM JSON response into open/occupied lists."""
    text = response_text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        log.warning(f"LLM response has no JSON: {text[:200]}")
        return [], list(spot_ids)

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as e:
        log.warning(f"LLM JSON parse error: {e}, response: {text[:200]}")
        return [], list(spot_ids)

    open_spots = []
    occupied_spots = []
    seen = set()

    for entry in data.get("spots", []):
        sid = entry.get("id")
        status = entry.get("status", "").lower()
        if sid not in spot_ids:
            continue
        seen.add(sid)
        if status == "open":
            open_spots.append(sid)
        else:
            occupied_spots.append(sid)

    for sid in spot_ids:
        if sid not in seen:
            occupied_spots.append(sid)

    return sorted(open_spots), sorted(occupied_spots)


def _draw_result_labels(img, spots, open_ids, occupied_ids, output_path, model_label="gpt-4.1"):
    """Draw result labels on the image for all spots."""
    labeled = img.copy()
    h, w = labeled.shape[:2]

    for spot in spots:
        spot_id = spot["id"]
        sx1 = int(spot["x"] * w)
        sy1 = int(spot["y"] * h)
        sx2 = sx1 + int(spot["w"] * w)
        sy2 = sy1 + int(spot["h"] * h)

        is_open = spot_id in open_ids
        color = (0, 200, 0) if is_open else (0, 0, 255)
        cv2.rectangle(labeled, (sx1, sy1), (sx2, sy2), color, 3)
        overlay = labeled.copy()
        cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), color, -1)
        cv2.addWeighted(overlay, 0.15, labeled, 0.85, 0, labeled)

        label = f"#{spot_id} OPEN" if is_open else f"#{spot_id}"
        cv2.putText(labeled, label, (sx1 + 4, sy2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(labeled, model_label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(output_path, labeled)


def detect(image_path, spots, deployment=None):
    """
    Detect open/occupied parking spots using Azure OpenAI vision.

    Returns dict with same shape as detector.detect().
    """
    deploy_name = deployment or AZURE_OPENAI_DEPLOYMENT
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY:
        log.error("AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_KEY not configured")
        return {"total": len(spots), "open": [], "occupied": [], "error": "Azure OpenAI not configured"}

    img = cv2.imread(image_path)
    if img is None:
        return {"total": 0, "open": [], "occupied": [], "error": "cannot read image"}

    enabled_spots = [s for s in spots if s.get("enabled") is not False]
    spot_ids = set(s["id"] for s in enabled_spots)

    annotated = _draw_spot_labels(img, enabled_spots)
    image_b64 = _encode_image_base64(annotated)

    try:
        log.info(f"Calling Azure OpenAI ({deploy_name}) vision for {len(enabled_spots)} spots...")
        response_text = _call_openai_vision(image_b64, sorted(spot_ids), deployment=deploy_name)
        log.info(f"LLM response: {response_text[:300]}")

        open_spots, occupied_spots = _parse_response(response_text, spot_ids)
        log.info(f"LLM result: {len(open_spots)} open, {len(occupied_spots)} occupied")

        for sid in open_spots:
            log.info(f"Spot {sid}: open ({deploy_name})")
        for sid in occupied_spots:
            log.info(f"Spot {sid}: occupied ({deploy_name})")

    except Exception as e:
        log.error(f"Azure OpenAI vision failed: {e}")
        open_spots = []
        occupied_spots = sorted(spot_ids)

    labeled_path = image_path.replace(".jpg", "_labeled.jpg")
    _draw_result_labels(img, enabled_spots, open_spots, occupied_spots, labeled_path, deploy_name)

    return {
        "total": len(enabled_spots),
        "open": open_spots,
        "occupied": occupied_spots,
        "vehicles": 0,
        "labeled_image": os.path.basename(labeled_path),
    }
