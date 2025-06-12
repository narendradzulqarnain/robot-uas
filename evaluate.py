import os
import torch
import numpy as np
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms

def to_grayscale_repeat3_np(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray3 = cv2.merge([gray, gray, gray])
    return gray3

def find_best_match_vectorized(embedding, known_embeddings, known_labels, threshold):
    if len(known_embeddings) == 0:
        return "Unknown", 0.0
    embedding_norm = np.linalg.norm(embedding)
    known_embeddings_norm = np.linalg.norm(known_embeddings, axis=1)
    similarities = np.dot(known_embeddings, embedding) / (known_embeddings_norm * embedding_norm + 1e-8)
    best_idx = np.argmax(similarities)
    best_sim = similarities[best_idx]
    if best_sim > threshold:
        return known_labels[best_idx], best_sim
    return "Unknown", best_sim

def evaluate_realtime(video_path, model_path, threshold=0.65):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Menggunakan device: {device}")

    # Load model
    try:
        model = InceptionResnetV1(pretrained=None).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model.eval()
        print("Model fine-tuned berhasil dimuat.")
    except Exception as e:
        print("Gagal memuat model:", e)
        return

    # Load embeddings
    try:
        data = np.load('face_embeddings.npz')
        known_labels = []
        known_embeddings = []
        for person in data.files:
            embs = data[person]
            known_labels.extend([person] * len(embs))
            known_embeddings.extend(embs)
        known_embeddings = np.stack(known_embeddings)
        print(f"Berhasil memuat {len(set(known_labels))} wajah yang dikenali.")
    except FileNotFoundError:
        print("File face_embeddings.npz tidak ditemukan.")
        return

    mtcnn = MTCNN(
        keep_all=True, device=device, min_face_size=20,
        thresholds=[0.7, 0.7, 0.8]
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Gagal membuka sumber video.")
        return

    PROCESS_EVERY_N_FRAMES = 1
    RESIZE_FACTOR = 0.75
    frame_count = 0

    print("Memulai pengenalan wajah... Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(rgb_small)
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    face_img = rgb_small[y1:y2, x1:x2]
                    if face_img.size == 0:
                        continue
                    face_img = cv2.resize(face_img, (160, 160))
                    face_gray3 = to_grayscale_repeat3_np(face_img)
                    face_tensor = torch.tensor(face_gray3).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    face_tensor = (face_tensor - 0.5) / 0.5
                    face_tensor = face_tensor.to(device)
                    with torch.no_grad():
                        emb = model(face_tensor).cpu().numpy().flatten()
                    label, sim = find_best_match_vectorized(emb, known_embeddings, known_labels, threshold)
                    color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (int(x1/RESIZE_FACTOR), int(y1/RESIZE_FACTOR)),
                                  (int(x2/RESIZE_FACTOR), int(y2/RESIZE_FACTOR)), color, 2)
                    cv2.putText(frame, f"{label} ({sim:.2f})", (int(x1/RESIZE_FACTOR), int(y1/RESIZE_FACTOR)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ganti '0' dengan path video jika ingin dari file
    evaluate_realtime('test.mp4', './facenet_finetuned_best.pth')