"""
Automatic recording logic module.
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
        self.out_dir ='recordings'
        os.makedirs(self.out_dir, exist_ok=True)
        self.fps = 20
        self.frame_size = None
        self.last_frame_time = None
        
        # Anti-spam parameters
        self.last_recording_end_time = None
        self.recording_cooldown = 5  # seconds between recordings
        self.crowd_state_history = []
        self.debounce_window = 10  # frames
        self.crowd_threshold = 0.6
        self.uncrowd_threshold = 0.4
        
    def update(self, crowded, frame):
        now = time.time()
        
        # Update crowd state history
        self.crowd_state_history.append(crowded)
        if len(self.crowd_state_history) > self.debounce_window:
            self.crowd_state_history.pop(0)
            
        # Calculate stable state
        crowd_ratio = sum(self.crowd_state_history) / len(self.crowd_state_history)
        stable_crowded = crowd_ratio >= self.crowd_threshold
        stable_not_crowded = crowd_ratio <= self.uncrowd_threshold
        
        # Check if we're in cooldown period
        in_cooldown = (self.last_recording_end_time and 
                      now - self.last_recording_end_time < self.recording_cooldown)
        
        # Start recording if conditions are met
        if stable_crowded and not self.recording and not in_cooldown:
            self.start_time = now
            self.frame_size = (frame.shape[1], frame.shape[0])
            out_path = os.path.join(self.out_dir, f"rec_{int(now)}.mp4")
            self.writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.frame_size)
            self.recording = True
            self.last_frame_time = now
            print(f"Started recording: {out_path}")
            
        if self.recording:
            self.writer.write(frame)
            self.last_frame_time = now
            
            # Stop recording if consistently not crowded and min duration passed
            if stable_not_crowded and (now - self.start_time) > self.min_duration:
                self.writer.release()
                self.writer = None
                self.recording = False
                self.last_recording_end_time = now
                print(f"Stopped recording after {now - self.start_time:.1f} seconds")
                
        # Safety: release writer if recording is stuck
        if self.recording and self.last_frame_time and (now - self.last_frame_time) > 30:
            self.writer.release()
            self.writer = None
            self.recording = False
            self.last_recording_end_time = now
            print("Recording stopped due to timeout")
        
    def get_status(self):
        """
        Get the current recording status information.
        Returns a dictionary with status details.
        """
        now = time.time()
        status = {
            'is_recording': self.recording,
            'elapsed_time': now - self.start_time if self.recording else 0,
            'in_cooldown': False,
            'cooldown_remaining': 0,
            'crowd_ratio': sum(self.crowd_state_history) / max(len(self.crowd_state_history), 1)
        }
        
        # Add cooldown information if applicable
        if self.last_recording_end_time:
            cooldown_elapsed = now - self.last_recording_end_time
            if cooldown_elapsed < self.recording_cooldown:
                status['in_cooldown'] = True
                status['cooldown_remaining'] = self.recording_cooldown - cooldown_elapsed
                
        return status