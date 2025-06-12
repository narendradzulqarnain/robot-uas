import os
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torchvision import datasets, transforms  # <- tambahkan 'transforms' di sini

def to_grayscale_repeat3(img):
    img = img.convert('L')
    img = np.array(img)
    img = np.stack([img]*3, axis=-1)
    return Image.fromarray(img)

def generate_embeddings():
    model_path = './facenet_finetuned_best.pth'
    data_dir = './img_train'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Menggunakan device: {device}")

    print(f"Memuat model fine-tuned dari: {model_path}")
    model = InceptionResnetV1(pretrained=None).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    print("Model fine-tuned berhasil dimuat.")

    mtcnn = MTCNN(
        image_size=160, margin=14, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709,
        post_process=True, device=device, keep_all=False
    )

    person_prototypes = {}
    failed_images = []

    print("Membuat prototype embeddings untuk wajah yang dikenali...")

    dataset = datasets.ImageFolder(data_dir)
    class_names = dataset.classes

    for class_idx, person_name in enumerate(class_names):
        person_dir = os.path.join(data_dir, person_name)
        embeddings = []
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                face_tensor = mtcnn(img)
                if face_tensor is None:
                    failed_images.append(img_path)
                    continue
                # Convert crop to grayscale, repeat 3 channel, normalize
                face_np = face_tensor.permute(1,2,0).cpu().numpy() * 255
                face_pil = Image.fromarray(face_np.astype(np.uint8))
                face_gray = to_grayscale_repeat3(face_pil)
                face_tensor_gray = transforms.functional.to_tensor(face_gray)
                face_tensor_gray = (face_tensor_gray - 0.5) / 0.5
                face_tensor_gray = face_tensor_gray.unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(face_tensor_gray).cpu().numpy().flatten()
                embeddings.append(emb)
            except Exception as e:
                print(f"Gagal memproses {img_path}: {e}")
                failed_images.append(img_path)
        if embeddings:
            person_prototypes[person_name] = np.stack(embeddings)
            print(f"{person_name}: {len(embeddings)} embeddings")
        else:
            print(f"{person_name}: Tidak ada embedding yang berhasil dibuat.")

    np.savez('face_embeddings.npz', **person_prototypes)
    print("Embeddings berhasil disimpan ke face_embeddings.npz")

    if failed_images:
        print("Gagal memproses beberapa gambar:")
        for img_path in failed_images:
            print("  ", img_path)

if __name__ == "__main__":
    generate_embeddings()