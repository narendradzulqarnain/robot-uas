"""
YOLO-based person detection module.
"""

from ultralytics import YOLO
import numpy as np
import torch

class PersonDetector:
    def __init__(self, config):
        # Load pretrained YOLOv8n model (tidak menggunakan fine-tuned model)
        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Menggunakan model YOLO pretrained standar
            self.model = YOLO('yolov8n.pt')
            self.model.to(self.device)
            print(f"Loaded pretrained YOLO model on {self.device}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None

    def detect(self, frame):
        """
        Run person detection on a frame.
        Returns: list of bounding boxes [(x1, y1, x2, y2), ...]
        """
        if self.model is None:
            return []
        results = self.model(frame, device=self.device)
        bboxes = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # class 0 is 'person' in COCO dataset
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bboxes.append((x1, y1, x2, y2))
        return bboxes