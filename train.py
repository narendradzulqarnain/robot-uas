import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from facenet_pytorch import InceptionResnetV1
from PIL import Image

class GrayscaleRepeat3:
    def __call__(self, img):
        img = img.convert('L')  # Grayscale
        img = transforms.functional.to_tensor(img)  # (1,H,W)
        img = img.repeat(3, 1, 1)  # (3,H,W)
        return img

def train_finetune_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs):
    model.train()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            embeddings = model(images)

            # Online hard triplet mining
            pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
            hard_triplets_loss = []
            for i in range(images.size(0)):
                anchor_label = labels[i]
                anchor_emb = embeddings[i]
                pos_mask = (labels == anchor_label)
                pos_mask[i] = False
                if pos_mask.sum() == 0:
                    continue
                pos_dists = pairwise_dist[i][pos_mask]
                hardest_pos = pos_dists.max()
                neg_mask = (labels != anchor_label)
                if neg_mask.sum() == 0:
                    continue
                neg_dists = pairwise_dist[i][neg_mask]
                hardest_neg = neg_dists.min()
                loss = criterion(anchor_emb.unsqueeze(0), 
                                 anchor_emb.unsqueeze(0), 
                                 anchor_emb.unsqueeze(0) + (hardest_neg - hardest_pos).unsqueeze(0))
                hard_triplets_loss.append(loss.mean())

            if not hard_triplets_loss:
                continue

            loss = torch.mean(torch.stack(hard_triplets_loss))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}")

        avg_loss = running_loss / num_batches if num_batches > 0 else float('inf')
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}')

        if avg_loss < best_loss and avg_loss > 0:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'facenet_finetuned_best.pth')
            print("Model terbaik disimpan (facenet_finetuned_best.pth)")

def main():
    data_dir = './img_train'
    num_epochs = 40
    learning_rate = 1e-5
    batch_size = 32
    margin = 0.6

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Menggunakan device: {device}")

    print("Memuat model InceptionResnetV1 pretrained (vggface2)...")
    model = InceptionResnetV1(pretrained='vggface2').to(device)

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        GrayscaleRepeat3(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
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

    criterion = nn.TripletMarginLoss(margin=margin, p=2, reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-7)

    print("Memulai proses fine-tuning...")
    train_finetune_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs)

    torch.save(model.state_dict(), 'facenet_finetuned_final.pth')
    print("Model fine-tuned final disimpan.")

if __name__ == '__main__':
    main()