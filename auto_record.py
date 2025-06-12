"""
Automatic recording logic module with visualization options.
"""

import cv2
import time
import os

class AutoRecorder:
    def __init__(self, config):
        self.recording = False
        self.writer = None
        self.start_time = None
        self.min_duration = 10  # seconds
        self.out_dir = 'recordings'
        os.makedirs(self.out_dir, exist_ok=True)
        self.fps = 20
        self.frame_size = None
        self.last_frame_time = None
        # New flag for recording visualized frames vs raw frames
        self.record_visualized = True  # Default to recording frames with visualizations
        self.manual_recording = False  # Flag for user-triggered recording
        
        # Add cooldown parameters to prevent spam recording
        self.last_recording_end = 0
        self.cooldown_period = 10  # Wait at least 30 seconds between auto recordings
        self.crowd_stability_frames = 0  # Count consecutive crowded frames
        self.stability_threshold = 15  # Require 15 consecutive crowded frames to start recording

    def start_recording(self, frame):
        """Start recording manually"""
        now = time.time()
        self.start_time = now
        self.frame_size = (frame.shape[1], frame.shape[0])
        filename = time.strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(self.out_dir, f"manual_{filename}.mp4")
        self.writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.frame_size)
        self.recording = True
        self.manual_recording = True
        self.last_frame_time = now
        print(f"Recording started: {out_path}")

    def stop_recording(self):
        """Stop recording manually"""
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.recording = False
        self.manual_recording = False
        self.last_recording_end = time.time()  # Track when recording ended
        print(f"Recording stopped after {time.time() - self.start_time:.1f} seconds")

    def toggle_visual_mode(self):
        """Toggle between recording raw frames or visualized frames"""
        self.record_visualized = not self.record_visualized
        mode = "VISUAL (with detections)" if self.record_visualized else "RAW (without detections)"
        print(f"Recording mode: {mode}")
        return self.record_visualized
        
    def update(self, crowded, raw_frame, visual_frame=None):
        """
        Update recorder state and write frames.
        
        Args:
            crowded: Boolean indicating if crowd is detected
            raw_frame: Original frame without visualizations
            visual_frame: Frame with visualizations (boxes, labels, etc.)
        """
        now = time.time()
        frame_to_write = visual_frame if (self.record_visualized and visual_frame is not None) else raw_frame
        
        # Auto recording mode (based on crowd)
        if not self.manual_recording:
            # Update crowd stability counter
            if crowded:
                self.crowd_stability_frames += 1
            else:
                self.crowd_stability_frames = 0
                
            # Check if we should start recording (stability check + cooldown check)
            cooldown_ok = (now - self.last_recording_end) > self.cooldown_period
            stability_ok = self.crowd_stability_frames >= self.stability_threshold
            
            if crowded and not self.recording and cooldown_ok and stability_ok:
                self.start_time = now
                self.frame_size = (raw_frame.shape[1], raw_frame.shape[0])
                out_path = os.path.join(self.out_dir, f"auto_{time.strftime('%Y%m%d-%H%M%S')}.mp4")
                self.writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.frame_size)
                self.recording = True
                self.last_frame_time = now
                print(f"Auto recording started: {out_path}")
                
            if self.recording:
                self.writer.write(frame_to_write)
                self.last_frame_time = now
                
                # Stop auto recording if not crowded for a sustained period (5 seconds)
                if not crowded and (now - self.start_time) > self.min_duration:
                    # We'll use a hysteresis approach - require multiple non-crowded frames to stop
                    # This prevents jittering when crowd state changes rapidly
                    if not hasattr(self, 'uncrowded_since'):
                        self.uncrowded_since = now
                    elif now - self.uncrowded_since > 5:  # 5 seconds of no crowd to stop
                        self.writer.release()
                        self.writer = None
                        self.recording = False
                        self.last_recording_end = now
                        if hasattr(self, 'uncrowded_since'):
                            delattr(self, 'uncrowded_since')
                        print(f"Auto recording stopped after {now - self.start_time:.1f} seconds")
                else:
                    # Reset the uncrowded timer if crowd is detected again
                    if hasattr(self, 'uncrowded_since'):
                        delattr(self, 'uncrowded_since')
        
        # Manual recording mode
        elif self.recording:  # Manual recording is active
            self.writer.write(frame_to_write)
            self.last_frame_time = now
            
        # Safety: release writer if recording is stuck
        if self.recording and self.last_frame_time and (now - self.last_frame_time) > 30:
            self.writer.release()
            self.writer = None
            self.recording = False
            self.manual_recording = False
            self.last_recording_end = now
            print("Recording timeout - stopped")