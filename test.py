import torch
import torch.nn as nn
import argparse
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from models.mmff_net import MMFF_Net
from utils.dataset import get_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description='Test MMFF Network')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--dataset', type=str, default='ntu', choices=['ntu', 'ut-mhad'], help='Dataset name')
    
    # Model
    parser.add_argument('--num-classes', type=int, default=60, help='Number of classes')
    parser.add_argument('--num-joints', type=int, default=25, help='Number of joints')
    parser.add_argument('--num-frames', type=int, default=300, help='Number of frames to sample')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    
    # Testing
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    
    # Output
    parser.add_argument('--save-dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--save-confusion-matrix', action='store_true', help='Save confusion matrix')
    
    return parser.parse_args()

def test(model, test_loader, device):
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for skeleton, image, label in pbar:
            skeleton = skeleton.to(device)
            image = image.to(device)
            
            # Forward
            output = model(skeleton, image)
            _, predicted = output.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.numpy())
    
    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(cm, save_path, class_names=None):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(20, 20))
    
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f'Confusion matrix saved to {save_path}')

def main():
    args = parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data loader (use validation set for testing)
    print('Loading data...')
    _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_frames=args.num_frames,
        num_joints=args.num_joints
    )
    print(f'Test samples: {len(test_loader.dataset)}')
    
    # Model
    graph_args = {'layout': 'ntu-rgb+d' if args.dataset == 'ntu' else 'ut-mhad'}
    model = MMFF_Net(
        num_classes=args.num_classes,
        num_joints=args.num_joints,
        graph_args=graph_args
    ).to(device)
    
    # Load checkpoint
    print(f'Loading checkpoint: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    print('Testing...')
    predictions, labels = test(model, test_loader, device)
    
    # Calculate accuracy
    accuracy = 100.0 * np.sum(predictions == labels) / len(labels)
    print(f'\nTest Accuracy: {accuracy:.2f}%')
    
    # Classification report
    print('\nClassification Report:')
    print(classification_report(labels, predictions, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Save results
    results_file = os.path.join(args.save_dir, 'test_results.txt')
    with open(results_file, 'w') as f:
        f.write(f'Test Accuracy: {accuracy:.2f}%\n\n')
        f.write('Classification Report:\n')
        f.write(classification_report(labels, predictions, digits=4))
    print(f'\nResults saved to {results_file}')
    
    # Plot and save confusion matrix
    if args.save_confusion_matrix:
        cm_path = os.path.join(args.save_dir, 'confusion_matrix.png')
        plot_confusion_matrix(cm, cm_path)
        
        # Save normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm_path = os.path.join(args.save_dir, 'confusion_matrix_normalized.png')
        plot_confusion_matrix(cm_normalized, cm_norm_path)
    
    print('\nTesting completed!')

if __name__ == '__main__':
    main()