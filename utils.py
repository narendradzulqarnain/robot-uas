"""
Utility functions for robot-uas project.
"""

import cv2

def label_to_color(label):
    if label == 'person':
        return (0, 255, 0)  # green for generic person
    # Hash label to a color (avoid too dark/bright)
    h = abs(hash(str(label)))
    r = 50 + (h % 180)
    g = 50 + ((h // 180) % 180)
    b = 50 + ((h // (180*180)) % 180)
    return (int(b), int(g), int(r))

def draw_bboxes(frame, bboxes, labels=None):
    """
    Draw bounding boxes and labels on a frame.
    Each label gets a unique color.
    """
    if labels is None:
        labels = [None] * len(bboxes)
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        color = label_to_color(label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(frame, str(label), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# Add more utility functions as needed
