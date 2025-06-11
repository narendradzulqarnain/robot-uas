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
        # Preprocess the image
        import cv2
        
        # Convert to RGB if grayscale
        if len(person_crop.shape) == 2:
            person_crop = cv2.cvtColor(person_crop, cv2.COLOR_GRAY2RGB)
        elif person_crop.shape[2] == 4:  # RGBA
            person_crop = cv2.cvtColor(person_crop, cv2.COLOR_RGBA2RGB)
        
        # Apply histogram equalization to improve contrast
        yuv = cv2.cvtColor(person_crop, cv2.COLOR_RGB2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        person_crop = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        boxes, _ = self.mtcnn.detect(person_crop)
        if boxes is None:
            return []
        bboxes = [tuple(map(int, box)) for box in boxes]
        return bboxes
