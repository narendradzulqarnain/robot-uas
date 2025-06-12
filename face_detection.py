"""
RetinaFace-based face detection module using insightface.
"""

import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

class FaceDetector:
    def __init__(self, config=None):
        # Initialize RetinaFace detector using insightface FaceAnalysis
        self.detector = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.detector.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 for CPU

    def detect(self, image):
        """
        Detect faces in an image using RetinaFace.
        Returns: list of face bounding boxes [(x1, y1, x2, y2), ...]
        """
        if image is None or image.size == 0:
            return []
        try:
            faces = self.detector.get(image)
            bboxes = []
            for face in faces:
                x1, y1, x2, y2 = map(int, face.bbox)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)
                bboxes.append((x1, y1, x2, y2))
            return bboxes
        except Exception as e:
            print(f"Error in RetinaFace face detection: {e}")
            return []