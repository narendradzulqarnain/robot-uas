"""
YOLO-based person detection module.
"""

from ultralytics import YOLO
import numpy as np

class PersonDetector:
    def __init__(self, config):
        # Load YOLOv11n model (pretrained or fine-tuned)
        try:
            # self.model = YOLO('yolo11n.pt')
            # # Fine-tune directly if needed
            # self.model.train(
            #     data="dataset-train/person detection/data.yaml",
            #     epochs=50,
            #     batch=16,
            #     imgsz=640,
            #     lr0=0.001,
            #     patience=10,
            #     project="runs/train",
            #     name="yolov11n_custom",
            #     exist_ok=True
            # )
            # After training, you can load the best weights if you want inference only:
            self.model = YOLO('runs/train/yolov11n_custom/weights/best.pt')
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
        results = self.model(frame)
        bboxes = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # class 0 is 'person'
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bboxes.append((x1, y1, x2, y2))
        return bboxes
