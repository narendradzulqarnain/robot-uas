"""
Crowd density calculation module.
"""

import numpy as np
from scipy.spatial.distance import cdist

class CrowdDensityCalculator:
    def __init__(self, config):
        self.dist_threshold = 75  # pixel distance threshold for 'crowded'
        self.crowd_count_threshold = 3  # number of close people to trigger 'crowded'

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
