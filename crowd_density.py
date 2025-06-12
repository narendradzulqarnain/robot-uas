"""
Crowd density calculation module with visualization.
"""

import numpy as np
import cv2
from scipy.spatial.distance import cdist

class CrowdDensityCalculator:
    def __init__(self, config):
        self.dist_threshold = 75  # pixel distance threshold for 'crowded'
        self.crowd_count_threshold = 3  # number of close people to trigger 'crowded'
        
        # Additional parameters for visualization
        self.heatmap_radius = int(self.dist_threshold * 1.2)  # Slightly larger than threshold
        self.heatmap_intensity = 0.6  # Alpha blending factor for heatmap

    def calculate(self, person_bboxes):
        """
        Calculate crowd density from person bounding boxes.
        Returns: density value, crowded (bool)
        """
        if len(person_bboxes) < 2:
            return 0, False
        centers = np.array([[(x1+x2)//2, (y1+y2)//2] for x1, y1, x2, y2 in person_bboxes])
        dists = cdist(centers, centers)
        close_pairs = np.sum((dists < self.dist_threshold) & (dists > 0)) // 2
        density = close_pairs / max(1, len(person_bboxes))
        crowded = close_pairs >= self.crowd_count_threshold
        return density, crowded
        
    def draw_crowd_highlight(self, frame, person_bboxes):
        """
        Highlight crowded areas with colored overlays
        """
        if len(person_bboxes) < 2:
            return frame
            
        result = frame.copy()
        h, w = frame.shape[:2]
        
        # Create a blank overlay
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Extract person centers
        centers = []
        for x1, y1, x2, y2 in person_bboxes:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            centers.append((center_x, center_y))
            
        # Draw circles at each center
        for center in centers:
            cv2.circle(overlay, center, self.dist_threshold // 2, (0, 0, 255), -1)
        
        # Apply Gaussian blur to create smooth areas
        overlay = cv2.GaussianBlur(overlay, (99, 99), 0)
        
        # Overlay with transparency
        cv2.addWeighted(overlay, 0.3, result, 1.0, 0, result)
        
        return result
    
    def draw_crowd_indicator(self, frame, person_bboxes):
        """
        Simple traffic light style indicator for crowd density
        """
        result = frame.copy()
        h, w = frame.shape[:2]
        
        # Get density value
        density, crowded = self.calculate(person_bboxes)
        
        # Draw a simple indicator in the corner
        indicator_size = 100
        padding = 10
        
        # Background
        cv2.rectangle(
            result,
            (w - indicator_size - padding, padding),
            (w - padding, indicator_size + padding),
            (50, 50, 50),
            -1
        )
        
        # Select color based on density
        if crowded:
            color = (0, 0, 255)  # Red
        elif density > 0.3:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 255, 0)  # Green
        
        # Draw circle indicator
        cv2.circle(
            result,
            (w - indicator_size//2 - padding, indicator_size//2 + padding),
            indicator_size//3,
            color,
            -1
        )
        
        # Add text
        cv2.putText(
            result,
            f"Crowd: {len(person_bboxes)}",
            (w - indicator_size - padding + 5, indicator_size + padding - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        return result