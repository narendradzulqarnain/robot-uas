"""
YOLO-based person detection module.
"""

from ultralytics import YOLO
import numpy as np
import torch

class PersonDetector:
    def __init__(self, config):
        # Load YOLOv11n model (pretrained or fine-tuned)
        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = YOLO('runs/train/yolov11n_custom/weights/best.pt')
            self.model.to(self.device)
        except Exception as e:
            print(f"Error loading or training YOLOv11n model: {e}")
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
                if cls == 0:  # class 0 is 'person'
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bboxes.append((x1, y1, x2, y2))
        return bboxes
