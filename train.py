import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import time
from tqdm import tqdm
import numpy as np

from models.mmff_net import MMFF_Net
from utils.dataset import get_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description='Train MMFF Network')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--dataset', type=str, default='ntu', choices=['ntu', 'ut-mhad'], help='Dataset name')
    
    # Model
    parser.add_argument('--num-classes', type=int, default=60, help='Number of classes')
    parser.add_argument('--num-joints', type=int, default=25, help='Number of joints (25 for NTU, 20 for UT-MHAD)')
    parser.add_argument('--num-frames', type=int, default=300, help='Number of frames to sample')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--lr-decay-epoch', type=int, nargs='+', default=[40, 60], help='Epochs to decay learning rate')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='Learning rate decay rate')
    
    # Others
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Directory to save logs')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--save-interval', type=int, default=5, help='Save checkpoint every N epochs')
    
    return parser.parse_args()

def adjust_learning_rate(optimizer, epoch, args):
    """Decay learning rate by a factor every N epochs"""
    lr = args.lr
    for milestone in args.lr_decay_epoch:
        if epoch >= milestone:
            lr *= args.lr_decay_rate
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer):
    model.train()
    
    losses = []
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (skeleton, image, label) in enumerate(pbar):
        skeleton = skeleton.to(device)
        image = image.to(device)
        label = label.to(device)
        
        # Forward
        optimizer.zero_grad()
        output = model(skeleton, image)
        loss = criterion(output, label)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Statistics
        losses.append(loss.item())
        _, predicted = output.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{np.mean(losses):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
        
        # Log to tensorboard
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar('train/loss', loss.item(), global_step)
        writer.add_scalar('train/acc', 100. * correct / total, global_step)
    
    return np.mean(losses), 100. * correct / total

def validate(model, val_loader, criterion, device, epoch, writer):
    model.eval()
    
    losses = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for skeleton, image, label in pbar:
            skeleton = skeleton.to(device)
            image = image.to(device)
            label = label.to(device)
            
            # Forward
            output = model(skeleton, image)
            loss = criterion(output, label)
            
            # Statistics
            losses.append(loss.item())
            _, predicted = output.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{np.mean(losses):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
    
    avg_loss = np.mean(losses)
    accuracy = 100. * correct / total
    
    writer.add_scalar('val/loss', avg_loss, epoch)
    writer.add_scalar('val/acc', accuracy, epoch)
    
    return avg_loss, accuracy

def main():
    args = parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data loaders
    print('Loading data...')
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_frames=args.num_frames,
        num_joints=args.num_joints
    )
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Val samples: {len(val_loader.dataset)}')
    
    # Model
    graph_args = {'layout': 'ntu-rgb+d' if args.dataset == 'ntu' else 'ut-mhad'}
    model = MMFF_Net(
        num_classes=args.num_classes,
        num_joints=args.num_joints,
        graph_args=graph_args
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0
    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
    
    # Tensorboard writer
    writer = SummaryWriter(args.log_dir)
    
    # Training loop
    print('Starting training...')
    for epoch in range(start_epoch, args.epochs):
        # Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, args)
        print(f'\nEpoch {epoch}/{args.epochs} - LR: {lr}')
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        if (epoch + 1) % args.save_interval == 0 or is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'args': args
            }
            
            save_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, save_path)
            print(f'Checkpoint saved: {save_path}')
            
            if is_best:
                best_path = os.path.join(args.save_dir, 'best_model.pth')
                torch.save(checkpoint, best_path)
                print(f'Best model saved: {best_path}')
    
    print(f'\nTraining completed! Best accuracy: {best_acc:.2f}%')
    writer.close()

if __name__ == '__main__':
    main()