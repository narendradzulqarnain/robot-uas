"""
MTCNN-based face detection module.
"""

import numpy as np
import cv2
from facenet_pytorch import MTCNN
import torch

def nms(bboxes, iou_threshold=0.4):
    if len(bboxes) == 0:
        return []
    bboxes = np.array(bboxes)
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = areas.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return [tuple(map(int, bboxes[i])) for i in keep]


class FaceDetector:
    def __init__(self, config=None):
        # Use CPU device for MTCNN (more compatible)
        self.device = torch.device('cpu')
        
        # Initialize MTCNN with optimized parameters
        self.mtcnn = MTCNN(
            keep_all=True,                    # Return all detected faces
            min_face_size=30,                 # Minimum face size in pixels
            thresholds=[0.6, 0.7, 0.8],       # Detection thresholds for the 3 stages
            factor=0.709,                     # Scale factor between image pyramids
            post_process=True,                # Apply post-processing
            device=self.device
        )
        
    def detect(self, image):
        """
        Detect faces in an image using MTCNN.
        Returns: list of face bounding boxes [(x1, y1, x2, y2), ...]
        """
        # Check if image is valid
        if image is None or image.size == 0:
            return []
            
        # Ensure we have a valid image
        try:
            # MTCNN expects RGB format
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run detection (boxes will be in format [x1, y1, x2, y2])
            boxes, probs = self.mtcnn.detect(rgb_image)
            
            # No faces detected
            if boxes is None:
                print("MTCNN: No faces detected")
                return []
                
            bboxes = []
            # Process detected faces
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                # Skip low confidence detections
                if prob < 0.8:
                    continue
                    
                # Convert to integers and ensure coordinates are valid
                x1, y1, x2, y2 = map(int, box.tolist())
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1 or (x2-x1)*(y2-y1) < 900:  # Minimum 30x30 pixels
                    continue
                    
                bboxes.append((x1, y1, x2, y2))
            
            print(f"MTCNN detected {len(bboxes)} faces with confidence")
            return bboxes
            
        except Exception as e:
            print(f"Error in MTCNN face detection: {e}")
            return []
            
    def detect_with_landmarks(self, image):
        """
        Detect faces and their landmarks using MTCNN.
        Returns: 
            - list of face bounding boxes [(x1, y1, x2, y2), ...]
            - list of landmarks (5 points per face)
        """
        if image is None or image.size == 0:
            return [], []
            
        try:
            # Convert to RGB for MTCNN
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run detection with landmarks
            boxes, probs, landmarks = self.mtcnn.detect(rgb_image, landmarks=True)
            
            if boxes is None:
                return [], []
                
            bboxes = []
            valid_landmarks = []
            
            # Process results
            for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
                if prob < 0.8:
                    continue
                    
                # Convert coordinates to integers
                x1, y1, x2, y2 = map(int, box.tolist())
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                bboxes.append((x1, y1, x2, y2))
                valid_landmarks.append(landmark)
            
            return bboxes, valid_landmarks
            
        except Exception as e:
            print(f"Error in MTCNN landmark detection: {e}")
            return [], []