"""
Test script for face recognition using video input.
Tests the face recognition component in isolation for better performance analysis.
"""

import cv2
import os
import time
import argparse
import numpy as np
from config import Config
from face_detection import FaceDetector
from face_recognition import FaceRecognizer

def draw_results(frame, face_boxes, face_labels, fps=None):
    """Draw bounding boxes and labels on the frame"""
    for (x1, y1, x2, y2), label in zip(face_boxes, face_labels):
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label with background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Display FPS if available
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

def test_video(config, video_path, output_path=None, skip_frames=1, display=True):
    """Test face recognition on a video file"""
    print(f"Starting face recognition test on video: {video_path}")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Initialize output video if requested
    out = None
    if output_path:
        out = cv2.VideoWriter(output_path, 
                             cv2.VideoWriter_fourcc(*'mp4v'), 
                             fps/skip_frames, 
                             (width, height))
    
    # Initialize face detection and recognition
    face_detector = FaceDetector(config)
    face_recognizer = FaceRecognizer(config)
    
    # Statistics
    frame_count = 0
    total_proc_time = 0
    total_faces = 0
    
    processing_times = []
    face_counts = []
    frame_times = []
    
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Skip frames if needed
        if frame_count % skip_frames != 0:
            frame_count += 1
            continue
            
        frame_start_time = time.time()
        
        # Detect faces
        face_boxes = face_detector.detect(frame)
        
        # Recognize each face
        face_labels = []
        for x1, y1, x2, y2 in face_boxes:
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            label = face_recognizer.recognize(face_img)
            face_labels.append(label if label else "Unknown")
        
        # Calculate processing time
        frame_proc_time = time.time() - frame_start_time
        processing_times.append(frame_proc_time)
        
        # Update statistics
        total_proc_time += frame_proc_time
        total_faces += len(face_boxes)
        face_counts.append(len(face_boxes))
        frame_times.append(time.time())
        
        # Calculate real-time FPS (moving average over last 10 frames)
        if len(frame_times) > 10:
            recent_fps = 10 / (frame_times[-1] - frame_times[-11])
        else:
            recent_fps = 1.0 / frame_proc_time if frame_proc_time > 0 else 0
        
        # Draw results
        result_frame = draw_results(frame.copy(), face_boxes, face_labels, recent_fps)
        
        # Write to output video
        if out:
            out.write(result_frame)
        
        # Display
        if display:
            cv2.imshow("Face Recognition Test", result_frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        
        # Print progress
        frame_count += 1
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            elapsed = time.time() - start_time
            eta = (elapsed / frame_count) * (total_frames - frame_count) if frame_count > 0 else 0
            print(f"Progress: {progress:.1f}% | ETA: {eta:.1f}s | Faces detected: {total_faces}")
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # Calculate statistics
    avg_proc_time = total_proc_time / (frame_count / skip_frames) if frame_count > 0 else 0
    avg_faces_per_frame = total_faces / (frame_count / skip_frames) if frame_count > 0 else 0
    
    print("\n=== Face Recognition Test Results ===")
    print(f"Total frames processed: {frame_count // skip_frames}")
    print(f"Total faces detected: {total_faces}")
    print(f"Average faces per frame: {avg_faces_per_frame:.2f}")
    print(f"Average processing time: {avg_proc_time:.3f} seconds per frame")
    print(f"Average FPS: {1/avg_proc_time:.1f}")
    
    # Calculate more detailed statistics if we have enough data
    if processing_times:
        print("\n=== Performance Analysis ===")
        print(f"Min processing time: {min(processing_times):.3f} seconds")
        print(f"Max processing time: {max(processing_times):.3f} seconds")
        print(f"Median processing time: {np.median(processing_times):.3f} seconds")
        
        # Processing time correlation with face count
        if len(face_counts) > 10 and max(face_counts) > 0:
            print("\n=== Performance by Face Count ===")
            for faces in range(max(face_counts) + 1):
                times = [t for t, fc in zip(processing_times, face_counts) if fc == faces]
                if times:
                    print(f"  {faces} faces: {np.mean(times):.3f}s average ({len(times)} frames)")
    
    return {
        'frames': frame_count // skip_frames,
        'total_faces': total_faces,
        'avg_faces_per_frame': avg_faces_per_frame,
        'avg_proc_time': avg_proc_time,
        'avg_fps': 1/avg_proc_time if avg_proc_time > 0 else 0,
        'processing_times': processing_times,
        'face_counts': face_counts
    }

def main():
    parser = argparse.ArgumentParser(description="Face Recognition Video Test Tool")
    parser.add_argument('--video', type=str, required=True, 
                        help='Path to input video file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output video file (optional)')
    parser.add_argument('--skip', type=int, default=1,
                        help='Process only every Nth frame (default: 1)')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display video (useful for batch processing)')
    
    args = parser.parse_args()
    config = Config()
    
    test_video(config, args.video, args.output, args.skip, not args.no_display)

if __name__ == "__main__":
    main()