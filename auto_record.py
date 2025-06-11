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
        self.out_dir = 'recordings'
        os.makedirs(self.out_dir, exist_ok=True)
        self.fps = 20
        self.frame_size = None
        self.last_frame_time = None

    def update(self, crowded, frame):
        now = time.time()
        if crowded and not self.recording:
            self.start_time = now
            self.frame_size = (frame.shape[1], frame.shape[0])
            out_path = os.path.join(self.out_dir, f"rec_{int(now)}.mp4")
            self.writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.frame_size)
            self.recording = True
            self.last_frame_time = now
        if self.recording:
            self.writer.write(frame)
            self.last_frame_time = now
            # Stop recording if not crowded and minimum duration has passed
            if not crowded and (now - self.start_time) > self.min_duration:
                self.writer.release()
                self.writer = None
                self.recording = False
        # Safety: release writer if recording is stuck (e.g., no new frames for a while)
        if self.recording and self.last_frame_time and (now - self.last_frame_time) > 30:
            self.writer.release()
            self.writer = None
            self.recording = False
