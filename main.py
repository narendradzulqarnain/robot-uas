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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Person detection (YOLO)
        person_bboxes = person_detector.detect(frame)
        person_labels = []
        face_bboxes = []
        face_labels = []

        # 2. For each person, detect face and recognize
        for bbox in person_bboxes:
            x1, y1, x2, y2 = bbox
            person_crop = frame[y1:y2, x1:x2]
            faces = face_detector.detect(person_crop)
            for fx1, fy1, fx2, fy2 in faces:
                # Adjust face bbox to original frame coordinates
                abs_face = (x1+fx1, y1+fy1, x1+fx2, y1+fy2)
                face_bboxes.append(abs_face)
                face_img = frame[abs_face[1]:abs_face[3], abs_face[0]:abs_face[2]]
                # Check if face_img is valid before recognition
                if face_img is not None and face_img.size > 0 and face_img.shape[0] > 0 and face_img.shape[1] > 0:
                    label = face_recognizer.recognize(face_img)
                else:
                    label = None
                face_labels.append(label if label else "Unknown")
                person_labels.append(label if label else "person")
            if not faces:
                person_labels.append("person")

        # 3. Crowd density calculation
        density, crowded = crowd_calculator.calculate(person_bboxes)

        # 4. Auto recording logic
        auto_recorder.update(crowded, frame)

        # 5. Visualization
        # draw_bboxes(frame, person_bboxes, person_labels)
        draw_bboxes(frame, face_bboxes, face_labels)
        cv2.putText(frame, f"Density: {density:.2f} | Crowded: {crowded}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if crowded else (0,255,0), 2)
        cv2.imshow("robot-uas", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
