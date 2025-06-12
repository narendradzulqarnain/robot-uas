import os
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1 # <-- Import yang diperlukan
from PIL import Image
from torchvision import datasets
from torch.utils.data import DataLoader

def generate_embeddings():
    # --- Konfigurasi ---
    # Gunakan model terbaik dari hasil fine-tuning
    model_path = './facenet_finetuned_best.pth'
    # Direktori data training Anda
    data_dir = './img_train'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Menggunakan device: {device}")
    
    # --- Memuat Model Fine-Tuned ---
    print(f"Memuat model fine-tuned dari: {model_path}")
    # Inisialisasi arsitektur model (tanpa bobot pretrained karena akan kita timpa)
    model = InceptionResnetV1(pretrained=None).to(device)
    
    # Muat bobot yang sudah Anda fine-tune
    # --- PERBAIKAN DI SINI ---
    # Tambahkan strict=False untuk mengabaikan lapisan yang tidak cocok (seperti logits)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    
    model.eval() # Set ke mode evaluasi
    print("Model fine-tuned berhasil dimuat.")

    # --- Setup MTCNN untuk Deteksi & Crop Wajah ---
    mtcnn = MTCNN(
        image_size=160, margin=14, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709,
        post_process=True, device=device, keep_all=False
    )

    # --- Proses Pembuatan Embeddings ---
    person_prototypes = {}
    failed_images = []

    print("Membuat prototype embeddings untuk wajah yang dikenali...")
    
    dataset = datasets.ImageFolder(data_dir)
    # Mendapatkan nama kelas (nama orang) dari dataset
    class_names = dataset.classes
    
    for person_name in class_names:
        person_dir = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        print(f"Memproses {person_name}...")
        individual_embeddings = []
        
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                
                # Deteksi dan crop wajah dari gambar
                face_tensor = mtcnn(img, save_path=None)
                
                if face_tensor is not None:
                    # Dapatkan embedding dari wajah yang terdeteksi
                    with torch.no_grad():
                        face_tensor = face_tensor.unsqueeze(0).to(device)
                        emb = model(face_tensor).cpu().numpy().flatten()
                    individual_embeddings.append(emb)
                else:
                    raise ValueError("MTCNN gagal mendeteksi wajah.")
                    
            except Exception as e:
                failed_images.append(f"{person_name}/{img_name}: {str(e)}")
                continue
        
        if individual_embeddings:
            # --- Kalkulasi Medoid untuk Prototipe yang Kuat ---
            embeddings_array = np.array(individual_embeddings)
            # 1. Hitung embedding rata-rata
            mean_embedding = np.mean(embeddings_array, axis=0)
            # 2. Cari jarak setiap embedding dari rata-rata
            distances = np.linalg.norm(embeddings_array - mean_embedding, axis=1)
            # 3. Medoid adalah embedding dengan jarak terkecil ke rata-rata
            medoid_index = np.argmin(distances)
            medoid_embedding = embeddings_array[medoid_index]
            
            # Normalisasi L2 pada prototipe final
            medoid_embedding /= np.linalg.norm(medoid_embedding)
            
            person_prototypes[person_name] = medoid_embedding
            print(f"  -> Prototipe untuk {person_name} dibuat dari {len(individual_embeddings)} sampel.")
        else:
            print(f"  PERINGATAN: Tidak ada embedding yang valid untuk {person_name}.")

    if failed_images:
        print(f"\nGagal memproses {len(failed_images)} gambar:")
        for failure in failed_images[:5]:
            print(f"  - {failure}")

    # --- Simpan Embeddings dan Label ---
    labels = list(person_prototypes.keys())
    embeddings = np.array(list(person_prototypes.values()))
    
    if len(labels) > 0:
        np.save('known_embeddings.npy', embeddings)
        np.save('known_labels.npy', np.array(labels))
        print(f"\nBerhasil menyimpan {len(embeddings)} prototype embeddings.")
        print(f"Bentuk embedding: {embeddings.shape}")
        print("\nEmbeddings dan label berhasil disimpan!")
    else:
        print("\nTidak ada embedding yang dibuat. Penyimpanan dibatalkan.")

if __name__ == "__main__":
    generate_embeddings()