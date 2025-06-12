"""
Face detection module using OpenCV's YuNet.
"""

import numpy as np
import torch
import cv2

class FaceDetector:
    def __init__(self, config=None):
        # Use YuNet from OpenCV for face detection
        self.model_path = './face_detection_yunet_2023mar.onnx'
        try:
            self.face_detector = cv2.FaceDetectorYN_create(
                self.model_path,
                "",
                (300, 300),
                score_threshold=0.5
            )
        except Exception as e:
            print(f"Error loading YuNet model: {e}")
            self.face_detector = None

    def detect(self, image):
        """
        Detect faces in an image using YuNet.
        Returns: list of face bounding boxes [(x1, y1, x2, y2), ...]
        """
        if image is None or image.size == 0 or self.face_detector is None:
            return []
        h, w = image.shape[:2]
        self.face_detector.setInputSize((w, h))
        try:
            retval, faces = self.face_detector.detect(image)
            bboxes = []
            if faces is not None and len(faces) > 0:
                for face in faces:
                    x, y, w, h, score = face[:5]
                    if score < 0.5:
                        continue
                    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                    bboxes.append((x1, y1, x2, y2))
            return bboxes
        except Exception as e:
            print(f"Error in YuNet face detection: {e}")
            return []


