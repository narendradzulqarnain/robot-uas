"""
Central configuration for robot-uas project.
"""

class Config:
    # Path to your fine-tuned YOLOv11n weights
    PERSON_MODEL_PATH = "runs/detect/train/weights/best.pt"
    # Path to your face recognition database
    FACE_DB_PATH = "dataset-train/face recognition/Face Recognition - 2.v5-grayscaled-5-people.folder/train/"
    # Default video file for testing
    # VIDEO_FILE = "keliling_kelas_data_train.mp4"
    VIDEO_FILE = "keliling_kelas_data_test - lower fps.mp4"
    # VIDEO_FILE = "test-mini.mp4"
    # Add more config values as needed