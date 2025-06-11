"""
MTCNN-based face detection module.
"""

from facenet_pytorch import MTCNN
import numpy as np

class FaceDetector:
    def __init__(self, config):
        self.mtcnn = MTCNN(keep_all=True, device='cpu')

    def detect(self, person_crop):
        """
        Detect faces in a cropped person image.
        Returns: list of face bounding boxes [(x1, y1, x2, y2), ...]
        """
        boxes, _ = self.mtcnn.detect(person_crop)
        if boxes is None:
            return []
        bboxes = [tuple(map(int, box)) for box in boxes]
        return bboxes
