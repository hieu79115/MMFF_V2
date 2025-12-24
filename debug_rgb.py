import torch
import numpy as np
from PIL import Image
import os
import pickle
from torchvision import transforms
import matplotlib.pyplot as plt

def debug_rgb_data(data_dir, num_samples=5):
    """Debug RGB data để tìm vấn đề"""
    print("="*80)
    print("DEBUGGING RGB DATA")
    print("="*80)
    
    # Load labels
    with open(os.path.join(data_dir, 'train_label.pkl'), 'rb') as f:
        sample_names, labels = pickle.load(f)
    
    image_dir = os.path.join(data_dir, 'images')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print(f"\nChecking {num_samples} random samples...")
    
    issues = []
    valid_count = 0
    
    # Check random samples
    indices = np.random.choice(len(sample_names), min(num_samples, len(sample_names)), replace=False)
    
    for idx in indices:
        name = sample_names[idx]
        label = labels[idx]
        
        img_path_jpg = os.path.join(image_dir, f"{name}.jpg")
        img_path_png = os.path.join(image_dir, f"{name}.png")
        
        img_path = img_path_jpg if os.path.exists(img_path_jpg) else img_path_png
        
        if not os.path.exists(img_path):
            issues.append(f"Missing image: {name}")
            continue
        
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            print(f"\n[{idx}] {name}")
            print(f"  Label: {label}")
            print(f"  Image size: {img.size}")
            print(f"  Image mode: {img.mode}")
            
            # Check if image is blank/black
            img_array = np.array(img)
            mean_val = img_array.mean()
            std_val = img_array.std()
            
            print(f"  Pixel mean: {mean_val:.2f}")
            print(f"  Pixel std: {std_val:.2f}")
            
            if mean_val < 10 and std_val < 10:
                issues.append(f"Nearly blank image: {name} (mean={mean_val:.2f})")
            
            # Try transform
            img_tensor = transform(img)
            print(f"  Tensor shape: {img_tensor.shape}")
            print(f"  Tensor mean: {img_tensor.mean():.4f}")
            print(f"  Tensor std: {img_tensor.std():.4f}")
            
            valid_count += 1
            
        except Exception as e:
            issues.append(f"Error loading {name}: {str(e)}")
    
    print(f"\n" + "="*80)
    print(f"Valid images: {valid_count}/{len(indices)}")
    
    if issues:
        print(f"\nISSUES FOUND ({len(issues)}):")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo issues found!")
    
    # Check all images statistics
    print(f"\n" + "="*80)
    print("CHECKING ALL IMAGES...")
    
    all_exists = 0
    all_missing = 0
    
    for name in sample_names:
        img_path_jpg = os.path.join(image_dir, f"{name}.jpg")
        img_path_png = os.path.join(image_dir, f"{name}.png")
        
        if os.path.exists(img_path_jpg) or os.path.exists(img_path_png):
            all_exists += 1
        else:
            all_missing += 1
    
    print(f"Total samples: {len(sample_names)}")
    print(f"Images found: {all_exists}")
    print(f"Images missing: {all_missing}")
    print(f"Coverage: {100*all_exists/len(sample_names):.2f}%")
    
    if all_missing > 0:
        print("\n WARNING: Some images are missing!")
        print("   This will cause RGB stream to train on black placeholders")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--num-samples', type=int, default=10)
    args = parser.parse_args()
    
    debug_rgb_data(args.data_dir, args.num_samples)