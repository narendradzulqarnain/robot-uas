"""
Central configuration for robot-uas project.
"""

class Config:
    # Path to your fine-tuned YOLOv11n weights
    PERSON_MODEL_PATH = "runs/detect/train/weights/best.pt"
    # Path to your face recognition database
    FACE_DB_PATH = "dataset-train/face recognition/Face Recognition - 2.v3-grayscaled-4-people.folder/train/"
    # Default video file for testing
    VIDEO_FILE = "test.mp4"
    # Add more config values as needed