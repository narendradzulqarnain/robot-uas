"""
Main entry point for the robot-uas project.
Handles input mode selection and coordinates the detection pipeline.
"""

import cv2
import sys

import numpy as np
from config import Config
from person_detection import PersonDetector
from crowd_density import CrowdDensityCalculator
from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from auto_record import AutoRecorder
from utils import draw_bboxes


def select_input_mode():
    print("Select input mode:")
    print("1. RTMP stream")
    print("2. Webcam")
    print("3. Video file")
    mode = input("Enter mode (1/2/3): ").strip()
    if mode == '1':
        rtmp_url = input("Enter RTMP URL: ").strip()
        return cv2.VideoCapture(rtmp_url)
    elif mode == '2':
        return cv2.VideoCapture(0)
    elif mode == '3':
        video_path = Config.VIDEO_FILE
        print(f"Using video file: {video_path}")
        return cv2.VideoCapture(video_path)
    else:
        print("Invalid mode. Exiting.")
        sys.exit(1)

def nms(bboxes, iou_threshold=0.3):
    """Non-Maximum Suppression untuk menghilangkan deteksi duplikat"""
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

def main():
    cap = select_input_mode()
    if not cap.isOpened():
        print("Failed to open video source.")
        return

    # Initialize modules
    config = Config()
    person_detector = PersonDetector(config)
    crowd_calculator = CrowdDensityCalculator(config)
    face_detector = FaceDetector(config)
    face_recognizer = FaceRecognizer(config)
    auto_recorder = AutoRecorder(config)
    
    # Performance variables
    process_every_n_frames = 5  # Hanya proses 1 dari setiap N frame
    frame_count = 0
    last_detection_results = {
        'person_bboxes': [],
        'person_labels': [],
        'face_bboxes': [],
        'face_labels': [],
        'density': 0,
        'crowded': False
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        process_current_frame = (frame_count % process_every_n_frames == 0)
        
        # Proses deteksi hanya pada beberapa frame
        if process_current_frame:
            # ------ PERSON DETECTION ------
            person_bboxes = person_detector.detect(frame)
            person_labels = ["person"] * len(person_bboxes)
            
            # ------ FACE DETECTION ------
            face_bboxes = face_detector.detect(frame)
            face_bboxes = nms(face_bboxes, iou_threshold=0.3)
            
            # ------ FACE RECOGNITION ------
            face_labels = []
            for bbox in face_bboxes:
                x1, y1, x2, y2 = bbox
                face_img = frame[y1:y2, x1:x2]
                if face_img is not None and face_img.size > 0:
                    label = face_recognizer.recognize(face_img)
                    face_labels.append(label if label else "Unknown")
                else:
                    face_labels.append("Unknown")

            # ------ CROWD DENSITY ------
            density, crowded = crowd_calculator.calculate(person_bboxes)
            
            # Update last detection results
            last_detection_results = {
                'person_bboxes': person_bboxes,
                'person_labels': person_labels,
                'face_bboxes': face_bboxes,
                'face_labels': face_labels,
                'density': density,
                'crowded': crowded
            }
        
        # Always use the most recent detection results for display
        auto_recorder.update(last_detection_results['crowded'], frame)
        draw_bboxes(frame, last_detection_results['face_bboxes'], last_detection_results['face_labels'])
        
        # Show crowd density info
        cv2.putText(frame, f"Density: {last_detection_results['density']:.2f} | Crowded: {last_detection_results['crowded']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0,0,255) if last_detection_results['crowded'] else (0,255,0), 2)
        
        cv2.imshow("robot-uas", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
