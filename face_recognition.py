"""
FaceNet-based face recognition module.
"""

from facenet_pytorch import InceptionResnetV1
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import os
import cv2

from facenet_pytorch import InceptionResnetV1
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import os
import cv2
import pickle
import time

class FaceRecognizer:
    def __init__(self, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("using " + self.device)
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.face_db = {}  # {person_name: [embedding1, embedding2, ...]}
        self.threshold = 0.65
        self.person_thresholds = {}  # {person_name: threshold}
        self.db_path = config.FACE_DB_PATH
        
        # Cache settings
        self.cache_path = "models/face_recognition_cache.pkl"
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        
        # Try to load from cache first
        if hasattr(config, 'USE_CACHED_EMBEDDINGS') and config.USE_CACHED_EMBEDDINGS:
            if self._load_from_cache():
                print("Successfully loaded face embeddings and thresholds from cache.")
                return
        
        # If loading from cache fails or is disabled, compute from scratch
        self._load_face_db_and_validate()
        
        # Save to cache for future use
        if hasattr(config, 'USE_CACHED_EMBEDDINGS') and config.USE_CACHED_EMBEDDINGS:
            self._save_to_cache()
            
    def _load_from_cache(self):
        """Try to load embeddings and thresholds from cache file"""
        if not os.path.exists(self.cache_path):
            print("Cache file not found. Computing embeddings from scratch.")
            return False
            
        try:
            # Check if database directory has been modified since cache was created
            db_modified = max(os.path.getmtime(os.path.join(root, f)) 
                          for root, _, files in os.walk(self.db_path) 
                          for f in files if f.lower().endswith(('.jpg', '.png')))
            cache_created = os.path.getmtime(self.cache_path)
            
            if db_modified > cache_created:
                print("Face database has been modified since cache was created. Rebuilding cache.")
                return False
                
            # Load the cache
            with open(self.cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                
            # Verify cache compatibility
            if cached_data.get('db_path') != self.db_path:
                print("Database path has changed. Rebuilding cache.")
                return False
                
            # Load embeddings and thresholds
            self.face_db = cached_data['face_db']
            self.person_thresholds = cached_data['person_thresholds']
            self.threshold = cached_data.get('threshold', self.threshold)
            
            print(f"Loaded face database from cache ({len(self.face_db)} persons):")
            for person, embeddings in self.face_db.items():
                thresh = self.person_thresholds.get(person, self.threshold)
                print(f"  {person}: {len(embeddings)} embeddings, threshold={thresh:.2f}")
                
            return True
            
        except Exception as e:
            print(f"Error loading cache: {e}")
            return False
            
    def _save_to_cache(self):
        """Save embeddings and thresholds to cache file"""
        try:
            cache_data = {
                'face_db': self.face_db,
                'person_thresholds': self.person_thresholds,
                'threshold': self.threshold,
                'db_path': self.db_path,
                'created': time.time()
            }
            
            with open(self.cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
                
            print(f"Face recognition data saved to cache: {self.cache_path}")
            return True
            
        except Exception as e:
            print(f"Error saving to cache: {e}")
            return False

    def _load_face_db_and_validate(self):
        print("Loading face database and validating with per-person threshold tuning...")
        all_embs = {}
        val_embs = {}
        # For each folder (person) in db_path, compute mean embedding from all images
        for person_name in os.listdir(self.db_path):
            person_dir = os.path.join(self.db_path, person_name)
            if not os.path.isdir(person_dir):
                continue
            img_paths = [os.path.join(person_dir, img) for img in os.listdir(person_dir) if img.lower().endswith(('.jpg','.png'))]
            if len(img_paths) < 2:
                continue
            train_imgs, val_imgs = train_test_split(img_paths, test_size=0.3, random_state=42)
            # Train embeddings (no augmentation)
            embs = []
            for img_path in train_imgs:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                if img.shape[0] != 160 or img.shape[1] != 160:
                    img = self._pad_to_160(img)
                img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float() / 255.0
                img_tensor = img_tensor.to(self.device)
                img_tensor = img_tensor.repeat(1, 3, 1, 1)
                with torch.no_grad():
                    emb = self.model(img_tensor).squeeze().cpu().numpy()
                embs.append(emb)
            if embs:
                all_embs[person_name] = embs
            # Validation embeddings
            val_embs[person_name] = []
            for img_path in val_imgs:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                if img.shape[0] != 160 or img.shape[1] != 160:
                    img = self._pad_to_160(img)
                img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float() / 255.0
                img_tensor = img_tensor.to(self.device)
                img_tensor = img_tensor.repeat(1, 3, 1, 1)
                with torch.no_grad():
                    emb = self.model(img_tensor).squeeze().cpu().numpy()
                val_embs[person_name].append(emb)
        # Per-person threshold tuning
        print("Validation set sizes:")
        for person, emb_list in val_embs.items():
            print(f"  {person}: {len(emb_list)} embeddings")
        self.person_thresholds = {}
        for person, emb_list in val_embs.items():
            best_acc = 0
            best_thresh = self.threshold
            for thresh in np.arange(0.8, 1.0, 0.01):
                correct = 0
                total = 0
                for emb in emb_list:
                    pred = self._recognize_embedding(emb, threshold=thresh, restrict_to=all_embs)
                    if pred == person:
                        correct += 1
                    total += 1
                acc = correct / total if total > 0 else 0
                if acc > best_acc:
                    best_acc = acc
                    best_thresh = thresh
            self.person_thresholds[person] = best_thresh
            print(f"  {person}: best threshold={best_thresh:.2f}, acc={best_acc*100:.2f}%")
        # After validation, fit on all data (train+val) for deployment (no augmentation)
        all_data_embs = {}
        for person_name in os.listdir(self.db_path):
            person_dir = os.path.join(self.db_path, person_name)
            if not os.path.isdir(person_dir):
                continue
            img_paths = [os.path.join(person_dir, img) for img in os.listdir(person_dir) if img.lower().endswith(('.jpg','.png'))]
            embs = []
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                if img.shape[0] != 160 or img.shape[1] != 160:
                    img = self._pad_to_160(img)
                img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float() / 255.0
                img_tensor = img_tensor.to(self.device)
                img_tensor = img_tensor.repeat(1, 3, 1, 1)
                with torch.no_grad():
                    emb = self.model(img_tensor).squeeze().cpu().numpy()
                embs.append(emb)
            if embs:
                all_data_embs[person_name] = embs
        self.face_db = all_data_embs

    def _pad_to_160(self, img):
        h, w = img.shape[:2]
        scale = min(160 / h, 160 / w)
        nh, nw = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (nw, nh))
        top = (160 - nh) // 2
        bottom = 160 - nh - top
        left = (160 - nw) // 2
        right = 160 - nw - left
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        return img_padded

    def _recognize_embedding(self, embedding, threshold=None, restrict_to=None):
        """
        Recognize a face embedding against the face database using cosine similarity.
        If restrict_to is provided, only search in that dict (used for validation).
        """
        db = restrict_to if restrict_to is not None else self.face_db
        max_sim = -1
        best_match = None
        for person_name, embs in db.items():
            for db_emb in embs:
                db_emb_norm = db_emb / np.linalg.norm(db_emb)
                emb_norm = embedding / np.linalg.norm(embedding)
                sim = np.dot(emb_norm, db_emb_norm)
                if sim > max_sim:
                    max_sim = sim
                    best_match = person_name
        # Use per-person threshold if available
        if best_match is not None:
            person_thresh = self.person_thresholds.get(best_match, self.threshold)
            if max_sim >= (threshold if threshold is not None else person_thresh):
                return best_match
        return "Unknown"

    def recognize(self, img):
        """
        Recognize a face in the given image.
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if img.shape[0] != 160 or img.shape[1] != 160:
            img = self._pad_to_160(img)
        img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float() / 255.0  # shape: (1, 1, 160, 160)
        img_tensor = img_tensor.to(self.device)
        img_tensor = img_tensor.repeat(1, 3, 1, 1)  # Repeat channel for FaceNet compatibility
        with torch.no_grad():
            emb = self.model(img_tensor).squeeze().cpu().numpy()
        return self._recognize_embedding(emb)

    def recognize_batch(self, imgs):
        """
        Recognize a batch of face images. imgs: list of np.ndarray (BGR or grayscale)
        Returns: list of labels (str)
        """
        if not imgs:
            return []
        # Convert all to grayscale and pad in one go (optimized)
        processed = [
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            for img in imgs
        ]
        processed = [
            self._pad_to_160(img) if img.shape[0] != 160 or img.shape[1] != 160 else img
            for img in processed
        ]
        batch = np.stack(processed, axis=0)  # (N, 160, 160)
        batch_tensor = torch.from_numpy(batch).unsqueeze(1).float() / 255.0  # (N, 1, 160, 160)
        batch_tensor = batch_tensor.to(self.device)
        batch_tensor = batch_tensor.repeat(1, 3, 1, 1)  # (N, 3, 160, 160)
        with torch.no_grad():
            embs = self.model(batch_tensor).cpu().numpy()  # (N, 512)
        labels = [self._recognize_embedding(emb) for emb in embs]
        return labels
