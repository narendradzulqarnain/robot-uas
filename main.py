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
                label = face_recognizer.recognize(face_img)
                face_labels.append(label if label else "Unknown")
                person_labels.append(label if label else "person")
            if not faces:
                person_labels.append("person")
            # person_labels.append("person")
        # 3. Crowd density calculation
        density, crowded = crowd_calculator.calculate(person_bboxes)

        # 4. Auto recording logic
        auto_recorder.update(crowded, frame)

        # 5. Visualization
        frame = draw_crowd_heatmap(frame, person_bboxes, crowd_calculator.dist_threshold)
        # draw_bboxes(frame, person_bboxes, person_labels)
        # draw_crowd_visualization(frame, person_bboxes, crowd_calculator.dist_threshold)
        draw_bboxes(frame, face_bboxes, face_labels)
        cv2.putText(frame, f"Density: {density:.2f} | Crowded: {crowded}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if crowded else (0,255,0), 2)
        # Display recording status
        if auto_recorder.recording:
            # Calculate elapsed time
            elapsed = time.time() - auto_recorder.start_time
            
            # Add red "REC" indicator with recording time
            cv2.putText(frame, f"REC {int(elapsed)}s", 
                       (frame.shape[1] - 160, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2)
            
            # Add flashing red circle indicator
            if int(elapsed * 2) % 2 == 0:  # Flash every half second
                cv2.circle(frame, (frame.shape[1] - 180, 25), 8, (0, 0, 255), -1)
        
        
        cv2.imshow("robot-uas", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def draw_crowd_visualization(frame, person_bboxes, dist_threshold=120):
    """
    Draw visualization of crowd proximity:
    - Red lines between people who are too close
    - Colored circles indicating crowd density for each person
    """
    if len(person_bboxes) < 2:
        return frame
        
    # Calculate centers of each person
    centers = []
    for x1, y1, x2, y2 in person_bboxes:
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        centers.append((center_x, center_y))
    
    # Calculate distances between all pairs
    import numpy as np
    from scipy.spatial.distance import cdist
    
    centers_np = np.array(centers)
    dists = cdist(centers_np, centers_np)
    
    # Draw lines between close people
    for i in range(len(centers)):
        # Count how many people are close to this person
        close_count = sum(1 for d in dists[i] if 0 < d < dist_threshold)
        
        # Color code based on crowding (green=0, yellow=1-2, orange=3-4, red=5+)
        if close_count == 0:
            color = (0, 255, 0)  # Green
            radius = 10
        elif close_count <= 2:
            color = (0, 255, 255)  # Yellow
            radius = 15
        elif close_count <= 4:
            color = (0, 165, 255)  # Orange
            radius = 20
        else:
            color = (0, 0, 255)  # Red
            radius = 25
            
        # Draw a circle around each person indicating their crowd level
        cv2.circle(frame, centers[i], radius, color, 2)
        
        # Draw connections between close people
        for j in range(i+1, len(centers)):
            if 0 < dists[i][j] < dist_threshold:
                # Draw a line connecting close people
                cv2.line(frame, centers[i], centers[j], (0, 0, 255), 1)
                
                # Draw the distance value
                mid_x = (centers[i][0] + centers[j][0]) // 2
                mid_y = (centers[i][1] + centers[j][1]) // 2
                cv2.putText(frame, f"{int(dists[i][j])}", (mid_x, mid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame
def draw_crowd_heatmap(frame, person_bboxes, dist_threshold=120, alpha=0.4):
    """
    Draw a heatmap overlay showing crowd density areas
    """
    if len(person_bboxes) < 2:
        return frame
        
    import numpy as np
    import cv2
    
    # Create a blank heatmap of the same size as the frame
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
    
    # Calculate centers of each person
    centers = []
    for x1, y1, x2, y2 in person_bboxes:
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        centers.append((center_x, center_y))
    
    # Add gaussian blob around each person
    for center in centers:
        x, y = center
        cv2.circle(heatmap, (x, y), dist_threshold, 1.0, -1)
        
    # Normalize heatmap values to 0-1
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    # Apply gaussian blur to smooth the heatmap
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    
    # Convert heatmap to colormap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Create a simple overlay with constant alpha
    result = frame.copy()
    cv2.addWeighted(heatmap_colored, float(alpha), result, 1.0 - float(alpha), 0, result)
    
    return result
if __name__ == "__main__":
    main()
