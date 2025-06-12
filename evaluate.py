import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1 # <-- Import yang diperlukan
from PIL import Image
from collections import deque, defaultdict
import time

def find_best_match_adaptive(embedding, known_embeddings, known_labels):
    """Menggunakan threshold yang dinamis berdasarkan skor kecocokan."""
    similarities = np.dot(known_embeddings, embedding) / (
        np.linalg.norm(known_embeddings, axis=1) * np.linalg.norm(embedding) + 1e-8)
    
    best_idx = np.argmax(similarities)
    best_sim = similarities[best_idx]
    
    # Threshold dinamis: lebih tinggi jika ada skor similarity yang hampir sama
    sorted_sim = np.sort(similarities)[::-1]
    if len(sorted_sim) > 1:
        gap = sorted_sim[0] - sorted_sim[1]  # Selisih dengan runner-up
        # Jika gap kecil, kita butuh keyakinan lebih tinggi
        threshold = 0.60 if gap > 0.15 else 0.70
    else:
        threshold = 0.65
    
    if best_sim > threshold:
        return known_labels[best_idx], best_sim
    return "Unknown", best_sim

def evaluate_realtime(video_path, model_path, blur_intensity=25):
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
        keep_all=True, device=device, min_face_size=40,
        thresholds=[0.6, 0.7, 0.7]
    )
    
    # --- Buka Sumber Video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka sumber video.")
        return

    # --- Variabel untuk Optimisasi & Smoothing ---
    PROCESS_EVERY_N_FRAMES = 1  # Hanya proses setiap 2 frame untuk kecepatan
    RESIZE_FACTOR = 1        # Ubah ukuran frame untuk deteksi lebih cepat
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
                        label, conf = find_best_match_adaptive(
                            embeddings[i], known_embeddings, known_labels
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

            # Jika wajah tidak dikenal, blur wajahnya
            if smoothed_label == "Unknown":
                display_frame = blur_face(display_frame, (x1, y1, x2, y2), blur_factor=blur_intensity)
                color = (0, 0, 255)  # Merah untuk tidak dikenal
            else:
                color = (0, 255, 0)  # Hijau untuk dikenal
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            label_text = f"{smoothed_label} ({confidence:.2f})"
            cv2.putText(display_frame, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow('Face Recognition', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def blur_face(image, box, blur_factor=25):
    """
    Menerapkan Gaussian blur pada area wajah tertentu dalam gambar
    
    Parameters:
    - image: Frame/gambar yang akan dimodifikasi
    - box: Bounding box wajah (x1, y1, x2, y2)
    - blur_factor: Intensitas blur (makin besar makin blur)
    
    Returns:
    - Image dengan area wajah yang sudah di-blur
    """
    x1, y1, x2, y2 = [int(coord) for coord in box]
    
    # Pastikan koordinat dalam batas frame
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    
    # Crop bagian wajah
    face_region = image[y1:y2, x1:x2]
    
    # Terapkan Gaussian blur
    blurred_face = cv2.GaussianBlur(face_region, (blur_factor, blur_factor), 0)
    
    # Masukkan kembali ke gambar asli
    image[y1:y2, x1:x2] = blurred_face
    
    return image

if __name__ == "__main__":
    # Gunakan 0 untuk webcam, atau berikan path ke file video
    video_source = './test.mp4'  # Ganti dengan '0' untuk webcam
    model_file = './facenet_finetuned_best.pth'
    
    evaluate_realtime(video_source, model_file)