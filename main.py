"""
Face recognition module for robot-uas project.
Optimized for fast video playback with RetinaFace detector.
"""
import cv2
import sys
import time
from config import Config
from face_detection import FaceDetector
from face_recognition import FaceRecognizer
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
    face_detector = FaceDetector(config)
    face_recognizer = FaceRecognizer(config)

    frame_counter = 0
    process_every_n_frames = 3  # Proses setiap 3 frame
    target_width = 480          # Resize frame untuk deteksi

    # FPS calculation
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0

    face_bboxes = []
    face_labels = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        fps_counter += 1

        # Calculate FPS every second
        if time.time() - fps_start_time > 1.0:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()

        display_frame = frame.copy()

        # Only process every nth frame for detection
        if frame_counter % process_every_n_frames == 0:
            h, w = frame.shape[:2]
            scale = target_width / w
            small_frame = cv2.resize(frame, (target_width, int(h * scale)))
            faces = face_detector.detect(small_frame)
            faces_original_size = []
            for fx1, fy1, fx2, fy2 in faces:
                x1 = int(fx1 / scale)
                y1 = int(fy1 / scale)
                x2 = int(fx2 / scale)
                y2 = int(fy2 / scale)
                faces_original_size.append((x1, y1, x2, y2))
            face_bboxes = []
            face_labels = []
            for fx1, fy1, fx2, fy2 in faces_original_size:
                valid_fx1 = max(0, fx1)
                valid_fy1 = max(0, fy1)
                valid_fx2 = min(frame.shape[1], fx2)
                valid_fy2 = min(frame.shape[0], fy2)
                if valid_fx2 <= valid_fx1 or valid_fy2 <= valid_fy1:
                    continue
                padding = int((valid_fx2 - valid_fx1) * 0.1)
                display_fx1 = max(0, valid_fx1 - padding)
                display_fy1 = max(0, valid_fy1 - padding)
                display_fx2 = min(frame.shape[1], valid_fx2 + padding)
                display_fy2 = min(frame.shape[0], valid_fy2 + padding)
                try:
                    face_img = frame[valid_fy1:valid_fy2, valid_fx1:valid_fx2]
                    if face_img is None or face_img.size == 0 or face_img.shape[0] == 0 or face_img.shape[1] == 0:
                        continue
                    label = face_recognizer.recognize(face_img)
                    face_bboxes.append((display_fx1, display_fy1, display_fx2, display_fy2))
                    face_labels.append(label if label else "Unknown")
                except Exception as e:
                    print(f"Error processing face: {e}")

        # Draw bounding boxes and labels
        draw_bboxes(display_frame, face_bboxes, face_labels)

        # Display performance metrics
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Face Recognition", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()