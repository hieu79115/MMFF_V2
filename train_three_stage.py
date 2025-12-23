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
from models.skeleton import SkeletonStream_STGCN
from models.rgb import RGBStream_Xception
from utils.dataset import get_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description='Three-Stage Training for MMFF Network')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--dataset', type=str, default='ut-mhad', choices=['ntu', 'ut-mhad'])
    
    # Model
    parser.add_argument('--num-classes', type=int, default=11, help='Number of classes')
    parser.add_argument('--num-joints', type=int, default=20, help='Number of joints')
    parser.add_argument('--num-frames', type=int, default=150, help='Number of frames')
    
    # Training stages
    parser.add_argument('--epochs-stage1', type=int, default=40, help='Epochs for stage 1')
    parser.add_argument('--epochs-stage2', type=int, default=20, help='Epochs for stage 2')
    parser.add_argument('--epochs-stage3', type=int, default=40, help='Epochs for stage 3')
    
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr-stage1', type=float, default=0.001, help='LR for stage 1')
    parser.add_argument('--lr-stage2', type=float, default=0.0001, help='LR for stage 2')
    parser.add_argument('--lr-stage3', type=float, default=0.0001, help='LR for stage 3')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay')
    
    # Others
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--resume-stage', type=int, default=0, help='Resume from stage (0=no resume)')
    
    return parser.parse_args()

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, stage_name, writer, global_step):
    model.train()
    
    losses = []
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'{stage_name} - Epoch {epoch}')
    for batch_idx, (skeleton, image, label) in enumerate(pbar):
        skeleton = skeleton.to(device)
        image = image.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        output = model(skeleton, image)
        loss = criterion(output, label)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        _, predicted = output.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()
        
        pbar.set_postfix({
            'loss': f'{np.mean(losses):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
        
        writer.add_scalar(f'{stage_name}/train_loss', loss.item(), global_step[0])
        writer.add_scalar(f'{stage_name}/train_acc', 100. * correct / total, global_step[0])
        global_step[0] += 1
    
    return np.mean(losses), 100. * correct / total

def validate(model, val_loader, criterion, device, epoch, stage_name, writer):
    model.eval()
    
    losses = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'{stage_name} - Validation')
        for skeleton, image, label in pbar:
            skeleton = skeleton.to(device)
            image = image.to(device)
            label = label.to(device)
            
            output = model(skeleton, image)
            loss = criterion(output, label)
            
            losses.append(loss.item())
            _, predicted = output.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            
            pbar.set_postfix({
                'loss': f'{np.mean(losses):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
    
    avg_loss = np.mean(losses)
    accuracy = 100. * correct / total
    
    writer.add_scalar(f'{stage_name}/val_loss', avg_loss, epoch)
    writer.add_scalar(f'{stage_name}/val_acc', accuracy, epoch)
    
    return avg_loss, accuracy

def stage1_train_separate_streams(args, device, train_loader, val_loader, writer):
    """Stage 1: Train Skeleton and RGB streams separately"""
    print('\n' + '='*80)
    print('STAGE 1: Training Skeleton and RGB Streams Separately')
    print('='*80)
    
    graph_args = {'layout': 'ntu-rgb+d' if args.dataset == 'ntu' else 'ut-mhad'}
    
    # Initialize full model
    full_model = MMFF_Net(
        num_classes=args.num_classes,
        num_joints=args.num_joints,
        graph_args=graph_args
    ).to(device)
    
    # Extract streams
    skeleton_stream = full_model.skeleton_stream
    rgb_stream = full_model.rgb_stream
    
    # Add temporary classifiers
    skeleton_classifier = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(128, args.num_classes)
    ).to(device)
    
    rgb_classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(2048, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, args.num_classes)
    ).to(device)
    
    # Optimizers
    skeleton_optimizer = optim.Adam(
        list(skeleton_stream.parameters()) + list(skeleton_classifier.parameters()),
        lr=args.lr_stage1,
        weight_decay=args.weight_decay
    )
    
    rgb_optimizer = optim.Adam(
        list(rgb_stream.parameters()) + list(rgb_classifier.parameters()),
        lr=args.lr_stage1,
        weight_decay=args.weight_decay
    )
    
    criterion = nn.CrossEntropyLoss()
    
    best_skeleton_acc = 0
    best_rgb_acc = 0
    global_step = [0]
    
    for epoch in range(args.epochs_stage1):
        print(f'\nStage 1 - Epoch {epoch+1}/{args.epochs_stage1}')
        
        # Train Skeleton Stream
        skeleton_stream.train()
        skeleton_classifier.train()
        
        skel_losses = []
        skel_correct = 0
        skel_total = 0
        
        pbar = tqdm(train_loader, desc='Training Skeleton Stream')
        for skeleton, _, label in pbar:
            skeleton = skeleton.to(device)
            label = label.to(device)
            
            skeleton_optimizer.zero_grad()
            skel_feat = skeleton_stream(skeleton)
            skel_out = skeleton_classifier(skel_feat)
            skel_loss = criterion(skel_out, label)
            
            skel_loss.backward()
            skeleton_optimizer.step()
            
            skel_losses.append(skel_loss.item())
            _, predicted = skel_out.max(1)
            skel_total += label.size(0)
            skel_correct += predicted.eq(label).sum().item()
            
            pbar.set_postfix({'loss': f'{np.mean(skel_losses):.4f}', 
                            'acc': f'{100.*skel_correct/skel_total:.2f}%'})
        
        skel_train_acc = 100. * skel_correct / skel_total
        
        # Train RGB Stream
        rgb_stream.train()
        rgb_classifier.train()
        
        rgb_losses = []
        rgb_correct = 0
        rgb_total = 0
        
        pbar = tqdm(train_loader, desc='Training RGB Stream')
        for _, image, label in pbar:
            image = image.to(device)
            label = label.to(device)
            
            rgb_optimizer.zero_grad()
            rgb_feat = rgb_stream(image)
            rgb_out = rgb_classifier(rgb_feat)
            rgb_loss = criterion(rgb_out, label)
            
            rgb_loss.backward()
            rgb_optimizer.step()
            
            rgb_losses.append(rgb_loss.item())
            _, predicted = rgb_out.max(1)
            rgb_total += label.size(0)
            rgb_correct += predicted.eq(label).sum().item()
            
            pbar.set_postfix({'loss': f'{np.mean(rgb_losses):.4f}',
                            'acc': f'{100.*rgb_correct/rgb_total:.2f}%'})
        
        rgb_train_acc = 100. * rgb_correct / rgb_total
        
        # Validation
        skeleton_stream.eval()
        skeleton_classifier.eval()
        rgb_stream.eval()
        rgb_classifier.eval()
        
        skel_val_correct = 0
        rgb_val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for skeleton, image, label in val_loader:
                skeleton = skeleton.to(device)
                image = image.to(device)
                label = label.to(device)
                
                # Skeleton
                skel_feat = skeleton_stream(skeleton)
                skel_out = skeleton_classifier(skel_feat)
                _, predicted = skel_out.max(1)
                skel_val_correct += predicted.eq(label).sum().item()
                
                # RGB
                rgb_feat = rgb_stream(image)
                rgb_out = rgb_classifier(rgb_feat)
                _, predicted = rgb_out.max(1)
                rgb_val_correct += predicted.eq(label).sum().item()
                
                val_total += label.size(0)
        
        skel_val_acc = 100. * skel_val_correct / val_total
        rgb_val_acc = 100. * rgb_val_correct / val_total
        
        print(f'Skeleton - Train Acc: {skel_train_acc:.2f}%, Val Acc: {skel_val_acc:.2f}%')
        print(f'RGB - Train Acc: {rgb_train_acc:.2f}%, Val Acc: {rgb_val_acc:.2f}%')
        
        # Log to tensorboard
        writer.add_scalar('Stage1/skeleton_train_acc', skel_train_acc, epoch)
        writer.add_scalar('Stage1/skeleton_val_acc', skel_val_acc, epoch)
        writer.add_scalar('Stage1/rgb_train_acc', rgb_train_acc, epoch)
        writer.add_scalar('Stage1/rgb_val_acc', rgb_val_acc, epoch)
        
        # Save best models
        if skel_val_acc > best_skeleton_acc:
            best_skeleton_acc = skel_val_acc
            torch.save(skeleton_stream.state_dict(), 
                      os.path.join(args.save_dir, 'skeleton_stream_best.pth'))
        
        if rgb_val_acc > best_rgb_acc:
            best_rgb_acc = rgb_val_acc
            torch.save(rgb_stream.state_dict(),
                      os.path.join(args.save_dir, 'rgb_stream_best.pth'))
    
    print(f'\nStage 1 Complete!')
    print(f'Best Skeleton Acc: {best_skeleton_acc:.2f}%')
    print(f'Best RGB Acc: {best_rgb_acc:.2f}%')
    
    return full_model

def stage2_train_fusion(args, device, model, train_loader, val_loader, writer):
    """Stage 2: Freeze backbones, train fusion modules"""
    print('\n' + '='*80)
    print('STAGE 2: Training Fusion Modules (Backbones Frozen)')
    print('='*80)
    
    # Load best weights from stage 1
    model.skeleton_stream.load_state_dict(
        torch.load(os.path.join(args.save_dir, 'skeleton_stream_best.pth'))
    )
    model.rgb_stream.load_state_dict(
        torch.load(os.path.join(args.save_dir, 'rgb_stream_best.pth'))
    )
    
    # Freeze backbones
    for param in model.skeleton_stream.parameters():
        param.requires_grad = False
    for param in model.rgb_stream.parameters():
        param.requires_grad = False
    
    # Only train fusion modules and classifier
    trainable_params = list(model.early_fusion.parameters()) + \
                      list(model.late_fusion.parameters()) + \
                      list(model.classifier.parameters())
    
    optimizer = optim.Adam(trainable_params, lr=args.lr_stage2, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    global_step = [0]
    
    for epoch in range(args.epochs_stage2):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            epoch, 'Stage2', writer, global_step
        )
        
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, 'Stage2', writer
        )
        
        print(f'Epoch {epoch+1}/{args.epochs_stage2} - '
              f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),
                      os.path.join(args.save_dir, 'stage2_best.pth'))
    
    print(f'\nStage 2 Complete! Best Val Acc: {best_acc:.2f}%')
    return model

def stage3_finetune_all(args, device, model, train_loader, val_loader, writer):
    """Stage 3: Fine-tune entire network"""
    print('\n' + '='*80)
    print('STAGE 3: Fine-tuning Entire Network')
    print('='*80)
    
    # Load best model from stage 2
    model.load_state_dict(
        torch.load(os.path.join(args.save_dir, 'stage2_best.pth'))
    )
    
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr_stage3, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[args.epochs_stage3//2, args.epochs_stage3*3//4],
        gamma=0.1
    )
    
    best_acc = 0
    global_step = [0]
    
    for epoch in range(args.epochs_stage3):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, 'Stage3', writer, global_step
        )
        
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, 'Stage3', writer
        )
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{args.epochs_stage3} - '
              f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'args': args
            }, os.path.join(args.save_dir, 'final_best_model.pth'))
    
    print(f'\nStage 3 Complete! Best Val Acc: {best_acc:.2f}%')
    return model

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
    
    # Tensorboard
    writer = SummaryWriter(args.log_dir)
    
    # Three-stage training
    if args.resume_stage <= 1:
        model = stage1_train_separate_streams(args, device, train_loader, val_loader, writer)
    
    if args.resume_stage <= 2:
        graph_args = {'layout': 'ntu-rgb+d' if args.dataset == 'ntu' else 'ut-mhad'}
        if args.resume_stage == 2:
            model = MMFF_Net(args.num_classes, args.num_joints, graph_args).to(device)
        model = stage2_train_fusion(args, device, model, train_loader, val_loader, writer)
    
    if args.resume_stage <= 3:
        graph_args = {'layout': 'ntu-rgb+d' if args.dataset == 'ntu' else 'ut-mhad'}
        if args.resume_stage == 3:
            model = MMFF_Net(args.num_classes, args.num_joints, graph_args).to(device)
        model = stage3_finetune_all(args, device, model, train_loader, val_loader, writer)
    
    writer.close()
    print('\n' + '='*80)
    print('THREE-STAGE TRAINING COMPLETED!')
    print('='*80)

if __name__ == '__main__':
    main()