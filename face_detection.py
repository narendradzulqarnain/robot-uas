"""
MediaPipe-based face detection module.
"""

import numpy as np
import mediapipe as mp
import cv2

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
        # MediaPipe does not support CUDA, so fallback to CPU
        self.face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    def detect(self, person_crop):
        """
        Detect faces in a cropped person image using MediaPipe Face Detection.
        Returns: list of face bounding boxes [(x1, y1, x2, y2), ...]
        """
        results = self.face_detection.process(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
        bboxes = []
        if results.detections:
            h, w, _ = person_crop.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x1 = int(bboxC.xmin * w)
                y1 = int(bboxC.ymin * h)
                x2 = int((bboxC.xmin + bboxC.width) * w)
                y2 = int((bboxC.ymin + bboxC.height) * h)
                bboxes.append((x1, y1, x2, y2))
        # Apply NMS to remove overlapping boxes
        bboxes = nms(bboxes, iou_threshold=0.2)
        return bboxes
