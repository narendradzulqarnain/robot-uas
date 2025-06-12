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
    show_person_detection = True
    show_face_detection = True
    show_density_heatmap = True
    prev_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        display_frame = frame.copy()
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
        density, crowded = crowd_calculator.calculate(person_bboxes)
        if show_density_heatmap and len(person_bboxes) > 1:
            # Apply density heatmap visualization
            display_frame = crowd_calculator.draw_crowd_indicator(display_frame, person_bboxes)
            
            # Display density information with color coding
            status_color = (0, 0, 255) if crowded else (0, 255, 0)  # Red if crowded, green otherwise
            status_text = f"Density: {density:.2f} | {'CROWDED' if crowded else 'Normal'}"
            
            cv2.putText(display_frame, status_text, 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    status_color, 2)
        
        
        # 5. Visualization
        if show_person_detection and person_bboxes:
            draw_bboxes(display_frame, person_bboxes, person_labels)
            
        if show_face_detection and face_bboxes:
            draw_bboxes(display_frame, face_bboxes, face_labels)

        # Calculate and show FPS
        if frame_count % 10 == 0:
            curr_time = time.time()
            fps = 10 / (curr_time - prev_time)
            prev_time = curr_time
        cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        # Auto recording logic
        auto_recorder.update(crowded, frame, display_frame)
        # Add recording indicator if recording
        if auto_recorder.recording:
            # Get frame dimensions
            h, w = display_frame.shape[:2]
            
            # Draw red blinking "REC" indicator with circle
            if int(time.time()) % 2 == 0:  # Blink every second
                # Position for bottom right (with padding)
                circle_x = w - 20
                circle_y = h - 20
                text_x = w - 100
                text_y = h - 25
                
                # Draw red circle
                cv2.circle(display_frame, (circle_x, circle_y), 10, (0, 0, 255), -1)
                
        status_text = []
        if show_person_detection:
            status_text.append("Person: ON")
        if show_face_detection:
            status_text.append("Face: ON")
        if show_density_heatmap:
            status_text.append("Density: ON")
        # if auto_recorder and auto_recorder.recording:
        #     status_text.append("REC")
        status_str = " | ".join(status_text)
        cv2.putText(frame, status_str, 
                  (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                  0.7, (0, 165, 255), 2)
        # cv2.putText(frame, f"Density: {density:.2f} | Crowded: {crowded}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if crowded else (0,255,0), 2)
        cv2.imshow("robot-uas", display_frame)
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            show_person_detection = not show_person_detection
            print(f"Person detection visualization: {'ON' if show_person_detection else 'OFF'}")
        elif key == ord('f'):
            show_face_detection = not show_face_detection
            print(f"Face detection visualization: {'ON' if show_face_detection else 'OFF'}")
        elif key == ord('d'):
            show_density_heatmap = not show_density_heatmap
            print(f"Density heatmap: {'ON' if show_density_heatmap else 'OFF'}")

    if auto_recorder.recording:
        auto_recorder.stop_recording()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
