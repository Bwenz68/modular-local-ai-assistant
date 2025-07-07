"""
assistant.vision
-----------------
Minimal YOLOv8 wrapper for the local AI Assistant container.

Usage (inside the container or host):
    python -m assistant.vision <image-or-video-path-or-URL>
"""

import sys
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"      # auto-downloads on first run

def run(source: str) -> None:
    model = YOLO(MODEL_PATH)
    # Force GPU 0 if available; otherwise CPU fallback
    device = 0 if model.device.type == "cuda" else "cpu"
    results = model(source, device=device)

    r = results[0]
    names = model.names
    detections = [names[int(cls_id)] for cls_id in r.boxes.cls]
    print(f"Detected: {detections}")

    # Save annotated image/video to runs/detect/
    if isinstance(source, (str, Path)) and Path(source).is_file():
        print("Result saved to:", results.save_dir)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m assistant.vision <image-or-video-path-or-URL>")
        sys.exit(1)
    run(sys.argv[1])
