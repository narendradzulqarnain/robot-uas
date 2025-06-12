import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1 # <-- Import yang diperlukan
from PIL import Image
from collections import deque, defaultdict
import time

def find_best_match_vectorized(embedding, known_embeddings, known_labels, threshold):
    """Kalkulasi kemiripan secara vectorized untuk kecepatan."""
    if len(known_embeddings) == 0:
        return "Unknown", 0.0
    
    # Normalisasi L2 untuk kalkulasi cosine similarity yang benar
    embedding_norm = np.linalg.norm(embedding)
    known_embeddings_norm = np.linalg.norm(known_embeddings, axis=1)
    
    # Kalkulasi cosine similarity untuk semua embedding yang diketahui
    similarities = np.dot(known_embeddings, embedding) / (known_embeddings_norm * embedding_norm + 1e-8)
    
    best_idx = np.argmax(similarities)
    best_sim = similarities[best_idx]
    
    if best_sim > threshold:
        return known_labels[best_idx], best_sim
    return "Unknown", best_sim

def evaluate_realtime(video_path, model_path, threshold=0.65):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Menggunakan device: {device}")
    
    # --- Memuat Model Fine-Tuned ---
    try:
        print(f"Memuat model fine-tuned dari: {model_path}")
        model = InceptionResnetV1(pretrained=None).to(device)
        
        # --- PERBAIKAN DI SINI ---
        # Tambahkan strict=False untuk mengabaikan lapisan `logits` yang tidak cocok
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        
        model.eval()
        print("Model fine-tuned berhasil dimuat.")
    except Exception as e:
        print(f"Error saat memuat model: {e}")
        return
    
    # --- Memuat Database Wajah yang Dikenali ---
    try:
        known_embeddings = np.load('known_embeddings.npy')
        known_labels = np.load('known_labels.npy')
        print(f"Berhasil memuat {len(known_embeddings)} wajah yang dikenali.")
    except FileNotFoundError:
        print("Error: File embedding tidak ditemukan. Jalankan generate_embeddings.py terlebih dahulu.")
        return
    
    # --- Setup MTCNN ---
    mtcnn = MTCNN(
        keep_all=True, device=device, min_face_size=20,
        thresholds=[0.7, 0.7, 0.8]
    )
    
    # --- Buka Sumber Video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka sumber video.")
        return

    # --- Variabel untuk Optimisasi & Smoothing ---
    PROCESS_EVERY_N_FRAMES = 1  # Hanya proses setiap 2 frame untuk kecepatan
    RESIZE_FACTOR = 0.75        # Ubah ukuran frame untuk deteksi lebih cepat
    face_buffer = defaultdict(lambda: deque(maxlen=10)) # Buffer untuk temporal smoothing
    
    frame_count = 0
    last_detections = []
    
    print("Memulai pengenalan wajah... Tekan 'q' untuk keluar.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        display_frame = frame.copy()
        
        # Hanya proses deteksi pada frame tertentu
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
            if len(small_frame.shape) == 2 or small_frame.shape[2] == 1:
                small_frame = cv2.cvtColor(small_frame, cv2.COLOR_GRAY2BGR)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Deteksi wajah
            boxes, _ = mtcnn.detect(rgb_small)
            
            # Ekstrak embeddings untuk wajah yang terdeteksi
            if boxes is not None:
                # MTCNN dapat langsung mengekstrak tensor wajah dari PIL image
                pil_img = Image.fromarray(rgb_small)
                face_tensors = mtcnn(pil_img)
                
                if face_tensors is not None:
                    with torch.no_grad():
                        embeddings = model(face_tensors.to(device)).cpu().numpy()
                    
                    # Kembalikan ukuran box ke frame asli
                    boxes = boxes / RESIZE_FACTOR
                    
                    current_detections = []
                    for i, box in enumerate(boxes):
                        label, conf = find_best_match_vectorized(
                            embeddings[i], known_embeddings, known_labels, threshold
                        )
                        current_detections.append({'box': box, 'label': label, 'conf': conf})
                    last_detections = current_detections

        # Gambar hasil deteksi terakhir pada setiap frame
        for detection in last_detections:
            x1, y1, x2, y2 = [int(coord) for coord in detection['box']]
            label, confidence = detection['label'], detection['conf']
            
            # Temporal smoothing untuk label yang lebih stabil
            face_key = f"{(x1+x2)//2}_{(y1+y2)//2}" # Kunci sederhana berdasarkan pusat box
            face_buffer[face_key].append(label)
            smoothed_label = max(set(face_buffer[face_key]), key=list(face_buffer[face_key]).count)

            color = (0, 255, 0) if smoothed_label != "Unknown" else (0, 0, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            label_text = f"{smoothed_label} ({confidence:.2f})"
            cv2.putText(display_frame, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow('Face Recognition', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Gunakan 0 untuk webcam, atau berikan path ke file video
    video_source = './test.mp4'  # Ganti dengan '0' untuk webcam
    model_file = './facenet_finetuned_best.pth'
    
    evaluate_realtime(video_source, model_file)