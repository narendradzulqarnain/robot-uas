"""
Script to rebuild face recognition cache.
Run this after adding new face images to force rebuild of embeddings and thresholds.
"""

from config import Config
from face_recognition import FaceRecognizer
import os
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description='Rebuild face recognition cache')
    parser.add_argument('--delete', action='store_true', 
                       help='Delete existing cache instead of rebuilding')
    args = parser.parse_args()
    
    cache_path = "models/face_recognition_cache.pkl"
    
    # Delete cache if requested
    if args.delete:
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"Deleted cache file: {cache_path}")
        else:
            print("No cache file found to delete.")
        return
    
    # Create custom config with caching enabled
    config = Config()
    config.USE_CACHED_EMBEDDINGS = True
    
    # Force rebuild by deleting existing cache
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"Removed old cache file: {cache_path}")
    
    print("Building face recognition database and thresholds...")
    start_time = time.time()
    
    # This will rebuild and save the cache
    face_recognizer = FaceRecognizer(config)
    
    # Report completion
    elapsed = time.time() - start_time
    print(f"Completed cache rebuild in {elapsed:.2f} seconds")
    print(f"Cache saved to: {cache_path}")

if __name__ == "__main__":
    main()