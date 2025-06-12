"""
Central configuration for robot-uas project.
"""

class Config:
    # Path to your fine-tuned YOLOv11n weights
    PERSON_MODEL_PATH = "runs/detect/train/weights/best.pt"
    # Path to your face recognition database
    FACE_DB_PATH = "dataset-train/face recognition/Face Recognition - 2.v9i.folder/train/"
    # Default video file for testing
    # VIDEO_FILE = "keliling_kelas_data_train.mp4"
    VIDEO_FILE = "train.mp4"
    # VIDEO_FILE = "test-mini.mp4"
    # Add more config values as needed