"""
Utility functions for robot-uas project.
"""

import cv2

def draw_bboxes(frame, bboxes, labels=None):
    """
    Draw bounding boxes and labels on a frame.
    """
    if labels is None:
        labels = [None] * len(bboxes)
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if label == 'person' or label is None else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(frame, str(label), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# Add more utility functions as needed
