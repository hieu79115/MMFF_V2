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
# from models.rgb import RGBStream_Xception
from models.rgb_simple import RGBStream_ResNet18
from utils.dataset import get_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description='Three-Stage Training for MMFF Network (FIXED)')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='ut-mhad', choices=['ntu', 'ut-mhad'])
    
    # Model
    parser.add_argument('--num-classes', type=int, default=27)  # Auto-detected
    parser.add_argument('--num-joints', type=int, default=20)
    parser.add_argument('--num-frames', type=int, default=150)
    
    # Training stages
    parser.add_argument('--epochs-stage1', type=int, default=60)
    parser.add_argument('--epochs-stage2', type=int, default=40)
    parser.add_argument('--epochs-stage3', type=int, default=60)
    
    parser.add_argument('--batch-size', type=int, default=16)
    
    # FIXED: Higher learning rates
    parser.add_argument('--lr-stage1-skeleton', type=float, default=0.01, help='LR for skeleton in stage 1')
    parser.add_argument('--lr-stage1-rgb', type=float, default=0.01, help='LR for RGB in stage 1')
    parser.add_argument('--lr-stage2', type=float, default=0.001, help='LR for stage 2')
    parser.add_argument('--lr-stage3', type=float, default=0.0001, help='LR for stage 3')
    parser.add_argument('--weight-decay', type=float, default=0.0004)
    
    # Others
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--save-dir', type=str, default='./checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--resume-stage', type=int, default=0)
    parser.add_argument('--print-freq', type=int, default=10)
    
    return parser.parse_args()

def train_epoch_simple(model, classifier, train_loader, criterion, optimizer, device, desc):
    """Simple training for single stream"""
    model.train()
    classifier.train()
    
    losses = []
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=desc)
    for data, label in pbar:
        data = data.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        feat = model(data)
        output = classifier(feat)
        loss = criterion(output, label)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        _, predicted = output.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()
        
        if len(losses) % 10 == 0:
            pbar.set_postfix({
                'loss': f'{np.mean(losses[-10:]):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    return np.mean(losses), 100. * correct / total

def validate_simple(model, classifier, val_loader, criterion, device):
    """Simple validation for single stream"""
    model.eval()
    classifier.eval()
    
    losses = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)
            
            feat = model(data)
            output = classifier(feat)
            loss = criterion(output, label)
            
            losses.append(loss.item())
            _, predicted = output.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
    
    return np.mean(losses), 100. * correct / total

def stage1_train_separate_streams(args, device, train_loader, val_loader, writer):
    """Stage 1: Train Skeleton and RGB streams separately with FIXED settings"""
    print('\n' + '='*80)
    print('STAGE 1: Training Skeleton and RGB Streams Separately (FIXED)')
    print('='*80)
    
    graph_args = {'layout': 'ntu-rgb+d' if args.dataset == 'ntu' else 'ut-mhad'}
    
    # Initialize streams
    skeleton_stream = SkeletonStream_STGCN(
        in_channels=3,
        num_class=args.num_classes,
        graph_args=graph_args
    ).to(device)
    
    # rgb_stream = RGBStream_Xception().to(device)
    rgb_stream = RGBStream_ResNet18(pretrained=True).to(device)
    
    # Simpler classifiers with BatchNorm
    # skeleton_classifier = nn.Sequential(
    #     nn.BatchNorm1d(256),
    #     nn.Dropout(0.3),
    #     nn.Linear(256, args.num_classes)
    # ).to(device)

    skeleton_classifier = nn.Sequential(
        nn.BatchNorm1d(256),
        nn.Dropout(0.5),  # Tăng từ 0.3 lên 0.5
        nn.Linear(256, args.num_classes)
    ).to(device)
    
    # rgb_classifier = nn.Sequential(
    #     nn.AdaptiveAvgPool2d(1),
    #     nn.Flatten(),
    #     nn.BatchNorm1d(2048),
    #     nn.Dropout(0.5),
    #     nn.Linear(2048, 512),
    #     nn.BatchNorm1d(512),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(0.5),
    #     nn.Linear(512, args.num_classes)
    # ).to(device)
    
    rgb_classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.6),  # Tăng từ 0.5 lên 0.6
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),  # Tăng từ 0.3 lên 0.5
        nn.Linear(256, args.num_classes)
    ).to(device)
    
    # FIXED: Higher learning rates with SGD
    skeleton_optimizer = optim.SGD(
        list(skeleton_stream.parameters()) + list(skeleton_classifier.parameters()),
        lr=args.lr_stage1_skeleton,
        momentum=0.9,
        weight_decay=0.001,  # Tăng từ 0.0004 lên 0.001
        nesterov=True
    )

    rgb_optimizer = optim.SGD(
        list(rgb_stream.parameters()) + list(rgb_classifier.parameters()),
        lr=args.lr_stage1_rgb,
        momentum=0.9,
        weight_decay=0.001,  # Tăng từ 0.0004 lên 0.001
        nesterov=True
    )
    
    # Learning rate schedulers
    # skeleton_scheduler = optim.lr_scheduler.MultiStepLR(
    #     skeleton_optimizer, 
    #     milestones=[30, 45], 
    #     gamma=0.1
    # )
    
    # rgb_scheduler = optim.lr_scheduler.MultiStepLR(
    #     rgb_optimizer,
    #     milestones=[30, 45],
    #     gamma=0.1
    # )

    skeleton_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        skeleton_optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )

    rgb_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        rgb_optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_skeleton_acc = 0
    best_rgb_acc = 0
    patience = 10
    skeleton_no_improve = 0
    rgb_no_improve = 0
    
    for epoch in range(args.epochs_stage1):
        print(f'\n[Stage 1] Epoch {epoch+1}/{args.epochs_stage1}')
        print(f'LR - Skeleton: {skeleton_optimizer.param_groups[0]["lr"]:.6f}, RGB: {rgb_optimizer.param_groups[0]["lr"]:.6f}')
        
        # Train Skeleton
        def get_skeleton_data(loader):
            for skeleton, _, label in loader:
                yield skeleton, label
        
        skeleton_loader = list(get_skeleton_data(train_loader))
        skel_train_loss, skel_train_acc = train_epoch_simple(
            skeleton_stream, skeleton_classifier,
            skeleton_loader, criterion, skeleton_optimizer, device,
            f'Training Skeleton [{epoch+1}/{args.epochs_stage1}]'
        )
        
        # Train RGB
        def get_rgb_data(loader):
            for _, image, label in loader:
                yield image, label
        
        rgb_loader = list(get_rgb_data(train_loader))
        rgb_train_loss, rgb_train_acc = train_epoch_simple(
            rgb_stream, rgb_classifier,
            rgb_loader, criterion, rgb_optimizer, device,
            f'Training RGB [{epoch+1}/{args.epochs_stage1}]'
        )
        
        # Validate
        skeleton_val_loader = list(get_skeleton_data(val_loader))
        skel_val_loss, skel_val_acc = validate_simple(
            skeleton_stream, skeleton_classifier,
            skeleton_val_loader, criterion, device
        )
        
        rgb_val_loader = list(get_rgb_data(val_loader))
        rgb_val_loss, rgb_val_acc = validate_simple(
            rgb_stream, rgb_classifier,
            rgb_val_loader, criterion, device
        )
        
        print(f'Skeleton - Train: {skel_train_acc:.2f}%, Val: {skel_val_acc:.2f}%')
        print(f'RGB      - Train: {rgb_train_acc:.2f}%, Val: {rgb_val_acc:.2f}%')
        
        # Step schedulers
        # skeleton_scheduler.step()
        # rgb_scheduler.step()
        skeleton_scheduler.step(skel_val_acc)
        rgb_scheduler.step(rgb_val_acc)
        
        # Log
        writer.add_scalar('Stage1/skeleton_train_acc', skel_train_acc, epoch)
        writer.add_scalar('Stage1/skeleton_val_acc', skel_val_acc, epoch)
        writer.add_scalar('Stage1/rgb_train_acc', rgb_train_acc, epoch)
        writer.add_scalar('Stage1/rgb_val_acc', rgb_val_acc, epoch)
        
        # Save best
        if skel_val_acc > best_skeleton_acc:
            best_skeleton_acc = skel_val_acc
            torch.save(skeleton_stream.state_dict(), 
                      os.path.join(args.save_dir, 'skeleton_stream_best.pth'))
            print(f'  → Best skeleton model saved: {best_skeleton_acc:.2f}%')
        
        if rgb_val_acc > best_rgb_acc:
            best_rgb_acc = rgb_val_acc
            torch.save(rgb_stream.state_dict(),
                      os.path.join(args.save_dir, 'rgb_stream_best.pth'))
            print(f'  → Best RGB model saved: {best_rgb_acc:.2f}%')

        # Early stopping for skeleton
        if skel_val_acc > best_skeleton_acc:
            best_skeleton_acc = skel_val_acc
            skeleton_no_improve = 0
            torch.save(skeleton_stream.state_dict(), 
                    os.path.join(args.save_dir, 'skeleton_stream_best.pth'))
            print(f'  → Best skeleton model saved: {best_skeleton_acc:.2f}%')
        else:
            skeleton_no_improve += 1
        
        # Early stopping for RGB
        if rgb_val_acc > best_rgb_acc:
            best_rgb_acc = rgb_val_acc
            rgb_no_improve = 0
            torch.save(rgb_stream.state_dict(),
                    os.path.join(args.save_dir, 'rgb_stream_best.pth'))
            print(f'  → Best RGB model saved: {best_rgb_acc:.2f}%')
        else:
            rgb_no_improve += 1
        
        # Stop if both haven't improved
        if skeleton_no_improve >= patience and rgb_no_improve >= patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break
    
    print(f'\nStage 1 Complete!')
    print(f'Best Skeleton Acc: {best_skeleton_acc:.2f}%')
    print(f'Best RGB Acc: {best_rgb_acc:.2f}%')
    
    # Build full model
    full_model = MMFF_Net(
        num_classes=args.num_classes,
        num_joints=args.num_joints,
        graph_args=graph_args
    ).to(device)
    
    return full_model

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
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        losses.append(loss.item())
        _, predicted = output.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()
        
        if batch_idx % 5 == 0:
            pbar.set_postfix({
                'loss': f'{np.mean(losses[-10:]) if len(losses) >= 10 else np.mean(losses):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        if batch_idx % 10 == 0:
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

def stage2_train_fusion(args, device, model, train_loader, val_loader, writer):
    """Stage 2: Train fusion with backbones frozen"""
    print('\n' + '='*80)
    print('STAGE 2: Training Fusion Modules')
    print('='*80)
    
    # Load best weights
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
    
    trainable_params = list(model.early_fusion.parameters()) + \
                      list(model.late_fusion.parameters()) + \
                      list(model.classifier.parameters())
    
    optimizer = optim.Adam(trainable_params, lr=args.lr_stage2, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_stage2)
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
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{args.epochs_stage2} - '
              f'Train: {train_acc:.2f}%, Val: {val_acc:.2f}%, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc
            }, os.path.join(args.save_dir, 'stage2_best.pth'))
            print(f'  → Best model saved: {best_acc:.2f}%')
    
    print(f'\nStage 2 Complete! Best Val Acc: {best_acc:.2f}%')
    return model

def stage3_finetune_all(args, device, model, train_loader, val_loader, writer):
    """Stage 3: Fine-tune entire network"""
    print('\n' + '='*80)
    print('STAGE 3: Fine-tuning Entire Network')
    print('='*80)
    
    # Load best from stage 2
    checkpoint = torch.load(os.path.join(args.save_dir, 'stage2_best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr_stage3, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[args.epochs_stage3//2, args.epochs_stage3*3//4],
        gamma=0.1
    )
    criterion = nn.CrossEntropyLoss()
    
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
              f'Train: {train_acc:.2f}%, Val: {val_acc:.2f}%, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc
            }, os.path.join(args.save_dir, 'final_best_model.pth'))
            print(f'  → Best model saved: {best_acc:.2f}%')
    
    print(f'\nStage 3 Complete! Best Val Acc: {best_acc:.2f}%')
    return model

def main():
    args = parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    print('Loading data...')
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_frames=args.num_frames,
        num_joints=args.num_joints
    )
    
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