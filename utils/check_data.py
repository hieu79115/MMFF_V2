import numpy as np
import pickle
import argparse
import os

def check_data(data_dir):
    """Check data format and labels"""
    print("="*80)
    print("CHECKING DATA")
    print("="*80)
    
    # Check train data
    print("\n[Train Data]")
    train_data = np.load(os.path.join(data_dir, 'train_data.npy'), mmap_mode='r')
    print(f"Skeleton shape: {train_data.shape}")
    
    with open(os.path.join(data_dir, 'train_label.pkl'), 'rb') as f:
        train_names, train_labels = pickle.load(f)
    
    train_labels = np.array(train_labels)
    print(f"Number of samples: {len(train_labels)}")
    print(f"Label range: {train_labels.min()} to {train_labels.max()}")
    print(f"Unique labels: {sorted(np.unique(train_labels).tolist())}")
    print(f"Number of unique labels: {len(np.unique(train_labels))}")
    
    # Check val data
    print("\n[Validation Data]")
    val_data = np.load(os.path.join(data_dir, 'val_data.npy'), mmap_mode='r')
    print(f"Skeleton shape: {val_data.shape}")
    
    with open(os.path.join(data_dir, 'val_label.pkl'), 'rb') as f:
        val_names, val_labels = pickle.load(f)
    
    val_labels = np.array(val_labels)
    print(f"Number of samples: {len(val_labels)}")
    print(f"Label range: {val_labels.min()} to {val_labels.max()}")
    print(f"Unique labels: {sorted(np.unique(val_labels).tolist())}")
    print(f"Number of unique labels: {len(np.unique(val_labels))}")
    
    # Check images
    print("\n[Images]")
    image_dir = os.path.join(data_dir, 'images')
    if os.path.exists(image_dir):
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        print(f"Number of images: {len(image_files)}")
        
        # Check if all samples have images
        missing_images = []
        for name in train_names[:10]:  # Check first 10
            if not os.path.exists(os.path.join(image_dir, f"{name}.jpg")) and \
               not os.path.exists(os.path.join(image_dir, f"{name}.png")):
                missing_images.append(name)
        
        if missing_images:
            print(f"Warning: {len(missing_images)} images missing (showing first 10)")
            print(f"Examples: {missing_images[:5]}")
        else:
            print("All sample images found!")
    else:
        print(f"Warning: Image directory not found: {image_dir}")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    all_labels = np.concatenate([train_labels, val_labels])
    unique_labels = sorted(np.unique(all_labels).tolist())
    num_classes = len(unique_labels)
    
    print(f"\nDetected {num_classes} classes")
    print(f"Labels: {unique_labels}")
    
    if min(unique_labels) == 0:
        print(f"\n✓ Labels start from 0 (correct)")
        print(f"  Use: --num-classes {num_classes}")
    elif min(unique_labels) == 1:
        print(f"\n✗ Labels start from 1 (will be remapped to 0)")
        print(f"  Use: --num-classes {num_classes}")
    else:
        print(f"\n✗ Labels start from {min(unique_labels)} (will be remapped to 0)")
        print(f"  Use: --num-classes {num_classes}")
    
    # Check skeleton dimensions
    C, T, V, M = train_data.shape[1:]
    print(f"\nSkeleton format: C={C}, T={T}, V={V}, M={M}")
    print(f"  Use: --num-joints {V}")
    print(f"  Recommended: --num-frames {min(T, 300)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    args = parser.parse_args()
    
    check_data(args.data_dir)