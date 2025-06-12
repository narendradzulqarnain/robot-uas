import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms
from facenet_model import FaceNet

def load_model(model_path, embedding_size=None):
    """Load model with automatic embedding size detection"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint to detect embedding size
    checkpoint = torch.load(model_path, map_location=device)
    
    if embedding_size is None:
        # Auto-detect embedding size from model weights
        embedding_key = 'embedding_head.9.weight'  # Updated key for new architecture
        if embedding_key in checkpoint:
            embedding_size = checkpoint[embedding_key].shape[0]
        else:
            embedding_size = 256  # Default fallback
    
    model = FaceNet(embedding_size).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def load_data(data_dir, batch_size=16, img_size=160):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                img = Image.open(img_path).convert('RGB')
                images.append(np.array(img))
                labels.append(label)
    return images, labels

def preprocess_image(image, target_size=(160, 160)):
    if isinstance(image, np.ndarray):
        from cv2 import cvtColor, COLOR_BGR2RGB
        image = cvtColor(image, COLOR_BGR2RGB)
        image = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def extract_frames_from_video(video_path, frame_rate=1):
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frames.append(frame)
        count += 1

    cap.release()
    return frames

def draw_label_on_frame(frame, label, position=(50, 50), color=(255, 0, 0), font_scale=1, thickness=2):
    cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return frame

def check_model_info(model_path):
    """Check model information"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Find embedding size with updated key
        embedding_key = 'embedding_head.9.weight'
        if embedding_key in checkpoint:
            embedding_size = checkpoint[embedding_key].shape[0]
            print(f"Model embedding size: {embedding_size}")
        else:
            print("Could not determine embedding size")
            embedding_size = None
        
        # Show model structure info
        print("Model structure:")
        for key in checkpoint.keys():
            if 'embedding_head' in key:
                print(f"  {key}: {checkpoint[key].shape}")
                
        return embedding_size
    except Exception as e:
        print(f"Error reading model: {e}")
        return None

def validate_training_data(data_dir='./img_train'):
    """Validate training data quality"""
    print("Validating training data...")
    
    total_images = 0
    valid_images = 0
    issues = []
    
    for person in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person)
        if not os.path.isdir(person_path):
            continue
            
        person_images = 0
        person_valid = 0
        
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            total_images += 1
            person_images += 1
            
            try:
                img = Image.open(img_path)
                img = img.convert('RGB')
                
                # Check image size
                width, height = img.size
                if width < 100 or height < 100:
                    issues.append(f"{person}/{img_file}: Too small ({width}x{height})")
                    continue
                
                # Check if image is corrupted
                img.verify()
                valid_images += 1
                person_valid += 1
                
            except Exception as e:
                issues.append(f"{person}/{img_file}: {str(e)}")
        
        print(f"{person}: {person_valid}/{person_images} valid images")
        if person_valid < 5:
            print(f"  WARNING: {person} has only {person_valid} valid images (recommended: 10+)")
    
    print(f"\nOverall: {valid_images}/{total_images} valid images")
    
    if issues:
        print(f"\nIssues found ({len(issues)}):")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
    
    return valid_images, total_images, issues