"""
Main entry point for the robot-uas project.
Handles input mode selection and coordinates the detection pipeline.
"""

import cv2
import sys
from config import Config
from person_detection import PersonDetector
from crowd_density import CrowdDensityCalculator
from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from auto_record import AutoRecorder
from utils import draw_bboxes
import time


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

    prev_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # 1. Person detection (YOLO)
        person_bboxes = person_detector.detect(frame)
        person_labels = ["person"] * len(person_bboxes)

        # 2. Face detection on the whole frame
        face_bboxes = face_detector.detect(frame)
        face_crops = []
        valid_face_indices = []
        for idx, (x1, y1, x2, y2) in enumerate(face_bboxes):
            face_img = frame[y1:y2, x1:x2]
            # Only add valid crops
            if face_img is not None and face_img.size > 0 and face_img.shape[0] > 0 and face_img.shape[1] > 0:
                face_crops.append(face_img)
                valid_face_indices.append(idx)
        # Batch face recognition for all valid faces
        face_labels = ["Unknown"] * len(face_bboxes)
        if face_crops:
            batch_labels = face_recognizer.recognize_batch(face_crops)
            for i, label in zip(valid_face_indices, batch_labels):
                face_labels[i] = label

        # 3. Crowd density calculation
        # density, crowded = crowd_calculator.calculate(person_bboxes)

        # # 4. Auto recording logic
        # auto_recorder.update(crowded, frame)

        # 5. Visualization
        # draw_bboxes(frame, person_bboxes, person_labels)
        draw_bboxes(frame, face_bboxes, face_labels)

        # Calculate and show FPS
        if frame_count % 10 == 0:
            curr_time = time.time()
            fps = 10 / (curr_time - prev_time)
            prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        

        # cv2.putText(frame, f"Density: {density:.2f} | Crowded: {crowded}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if crowded else (0,255,0), 2)
        cv2.imshow("robot-uas", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
