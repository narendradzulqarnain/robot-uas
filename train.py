import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from facenet_pytorch import InceptionResnetV1 # <-- Import model dari library
import numpy as np

def train_finetune_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs):
    """
    Fungsi training loop untuk fine-tuning model FaceNet.
    """
    model.train() # Set model ke mode training
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            # Dapatkan embeddings dari model
            embeddings = model(images)
            
            # Online hard triplet mining (sama seperti kode Anda sebelumnya)
            pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
            
            hard_triplets_loss = []
            for i in range(images.size(0)):
                anchor_label = labels[i]
                positive_mask = (labels == anchor_label) & (torch.arange(images.size(0)).to(device) != i)
                negative_mask = (labels != anchor_label)
                
                if not torch.any(positive_mask) or not torch.any(negative_mask):
                    continue
                
                hardest_positive_dist = pairwise_dist[i][positive_mask].max()
                hardest_negative_dist = pairwise_dist[i][negative_mask].min()
                
                triplet_loss = torch.relu(hardest_positive_dist - hardest_negative_dist + criterion.margin)
                hard_triplets_loss.append(triplet_loss)

            if not hard_triplets_loss:
                continue
            
            loss = torch.mean(torch.stack(hard_triplets_loss))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'  Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        avg_loss = running_loss / num_batches if num_batches > 0 else float('inf')
        scheduler.step(avg_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}')
        
        # Simpan model terbaik berdasarkan loss
        if avg_loss < best_loss and avg_loss > 0:
            best_loss = avg_loss
            # Simpan hanya bobot model (state_dict)
            torch.save(model.state_dict(), 'facenet_finetuned_best.pth')
            print(f'  -> Model fine-tuned terbaik disimpan (loss: {best_loss:.4f})')

def main():
    # --- Konfigurasi ---
    data_dir = './img_train'
    num_epochs = 40  # Jumlah epoch bisa lebih sedikit untuk fine-tuning
    
    # PENTING: Gunakan learning rate yang SANGAT KECIL untuk fine-tuning
    learning_rate = 1e-5 
    
    batch_size = 32
    margin = 0.6
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Menggunakan device: {device}")

    # --- Memuat Model Pretrained ---
    # 'vggface2' atau 'casia-webface' adalah pilihan dataset pretrained
    print("Memuat model InceptionResnetV1 pretrained (vggface2)...")
    model = InceptionResnetV1(pretrained='vggface2').to(device)
    
    # --- Data Augmentation dan Loader ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # Filter kelas yang memiliki kurang dari 2 sampel
    class_counts = torch.bincount(torch.tensor(dataset.targets))
    valid_classes = (class_counts >= 2).nonzero(as_tuple=True)[0]
    valid_indices = [i for i, target in enumerate(dataset.targets) if target in valid_classes]
    sampler = torch.utils.data.SubsetRandomSampler(valid_indices)
    
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    print(f"Akan melakukan fine-tuning pada {len(valid_indices)} sampel dari {len(valid_classes)} kelas.")

    # --- Setup untuk Training ---
    criterion = nn.TripletMarginLoss(margin=margin, p=2, reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-7)

    print("Memulai proses fine-tuning...")
    train_finetune_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs)

    # Simpan model final
    torch.save(model.state_dict(), 'facenet_finetuned_final.pth')
    print("Model fine-tuned final disimpan.")

if __name__ == '__main__':
    main()