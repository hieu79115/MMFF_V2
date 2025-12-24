import torch
import torch.nn as nn
from models.skeleton import SkeletonStream_STGCN
from models.rgb_simple import RGBStream_ResNet18
from utils.dataset import get_dataloaders
import argparse

def test_single_batch(data_dir):
    """Test if model can overfit a single batch"""
    print("="*80)
    print("TESTING SINGLE BATCH OVERFITTING")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load one batch
    train_loader, _ = get_dataloaders(
        data_dir=data_dir,
        batch_size=8,
        num_workers=0,
        num_frames=150,
        num_joints=20
    )
    
    # Get single batch
    skeleton_batch, image_batch, label_batch = next(iter(train_loader))
    skeleton_batch = skeleton_batch.to(device)
    image_batch = image_batch.to(device)
    label_batch = label_batch.to(device)
    
    print(f"\nBatch size: {skeleton_batch.shape[0]}")
    print(f"Skeleton shape: {skeleton_batch.shape}")
    print(f"Image shape: {image_batch.shape}")
    print(f"Labels: {label_batch.tolist()}")
    
    # Test RGB stream
    print("\n" + "-"*80)
    print("Testing RGB Stream (ResNet18)")
    print("-"*80)
    
    rgb_stream = RGBStream_ResNet18(pretrained=True).to(device)
    rgb_classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(512, 27)
    ).to(device)
    
    optimizer = torch.optim.SGD(
        list(rgb_stream.parameters()) + list(rgb_classifier.parameters()),
        lr=0.1,  # High LR for overfitting test
        momentum=0.9
    )
    criterion = nn.CrossEntropyLoss()
    
    print("\nTraining RGB on single batch (should reach ~100% if working)...")
    for epoch in range(50):
        optimizer.zero_grad()
        
        feat = rgb_stream(image_batch)
        output = rgb_classifier(feat)
        loss = criterion(output, label_batch)
        
        loss.backward()
        optimizer.step()
        
        _, predicted = output.max(1)
        acc = 100. * predicted.eq(label_batch).sum().item() / label_batch.size(0)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc:.2f}%")
    
    print(f"\nFinal RGB Accuracy: {acc:.2f}%")
    
    if acc < 80:
        print("⚠️  WARNING: RGB cannot overfit single batch!")
        print("   Possible issues:")
        print("   - Images are corrupted/blank")
        print("   - Transform is wrong")
        print("   - Model architecture issue")
    else:
        print("✓ RGB stream working correctly!")
    
    # Test Skeleton stream
    print("\n" + "-"*80)
    print("Testing Skeleton Stream")
    print("-"*80)
    
    graph_args = {'layout': 'ut-mhad'}
    skeleton_stream = SkeletonStream_STGCN(
        in_channels=3,
        num_class=27,
        graph_args=graph_args
    ).to(device)
    skeleton_classifier = nn.Linear(256, 27).to(device)
    
    optimizer = torch.optim.SGD(
        list(skeleton_stream.parameters()) + list(skeleton_classifier.parameters()),
        lr=0.1,
        momentum=0.9
    )
    
    print("\nTraining Skeleton on single batch...")
    for epoch in range(50):
        optimizer.zero_grad()
        
        feat = skeleton_stream(skeleton_batch)
        output = skeleton_classifier(feat)
        loss = criterion(output, label_batch)
        
        loss.backward()
        optimizer.step()
        
        _, predicted = output.max(1)
        acc = 100. * predicted.eq(label_batch).sum().item() / label_batch.size(0)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc:.2f}%")
    
    print(f"\nFinal Skeleton Accuracy: {acc:.2f}%")
    
    if acc < 80:
        print("WARNING: Skeleton cannot overfit single batch!")
    else:
        print("✓ Skeleton stream working correctly!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data')
    args = parser.parse_args()
    
    test_single_batch(args.data_dir)