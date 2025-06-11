"""
FaceNet-based face recognition module.
"""

from facenet_pytorch import InceptionResnetV1
import torch
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import random

class FaceRecognizer:
    def __init__(self, config):
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.face_db = {}  # {person_name: [embedding1, embedding2, ...]}
        self.threshold = 0.65
        self.db_path = config.FACE_DB_PATH
        self._load_face_db_and_validate()

    def _augment(self, img):
        # Simple augmentation: random horizontal flip, brightness
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        if random.random() > 0.5:
            factor = 0.7 + 0.6 * random.random()  # 0.7 to 1.3
            img = np.clip(img * factor, 0, 255).astype(np.uint8)
        return img

    def _load_face_db_and_validate(self):
        print("Loading face database and validating with threshold tuning...")
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
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (160, 160))
                img_tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    emb = self.model(img_tensor).squeeze().numpy()
                embs.append(emb)
            if embs:
                all_embs[person_name] = embs
            # Validation embeddings
            val_embs[person_name] = []
            for img_path in val_imgs:
                img = cv2.imread(img_path)
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (160, 160))
                img_tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    emb = self.model(img_tensor).squeeze().numpy()
                val_embs[person_name].append(emb)
        # Threshold tuning
        best_acc = 0
        best_thresh = self.threshold
        for thresh in np.arange(0.5, 0.85, 0.01):
            correct = 0
            total = 0
            for person, emb_list in val_embs.items():
                for emb in emb_list:
                    pred = self._recognize_embedding(emb, thresh)
                    if pred == person:
                        correct += 1
                    total += 1
            acc = correct / total if total > 0 else 0
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        self.threshold = best_thresh
        print(f"Best threshold: {self.threshold:.2f} | Validation accuracy: {best_acc*100:.2f}%")
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
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (160, 160))
                img_tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    emb = self.model(img_tensor).squeeze().numpy()
                embs.append(emb)
            if embs:
                all_data_embs[person_name] = embs
        self.face_db = all_data_embs

    def _recognize_embedding(self, embedding, threshold=None):
        """
        Recognize a face embedding against the face database.
        """
        if threshold is None:
            threshold = self.threshold
        min_dist = float('inf')
        best_match = None
        for person_name, embs in self.face_db.items():
            for db_emb in embs:
                dist = np.linalg.norm(embedding - db_emb)
                if dist < min_dist:
                    min_dist = dist
                    best_match = person_name
        return best_match if min_dist <= threshold else "Unknown"

    def recognize(self, img):
        """
        Recognize a face in the given image.
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (160, 160))
        img_tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            emb = self.model(img_tensor).squeeze().numpy()
        return self._recognize_embedding(emb)
