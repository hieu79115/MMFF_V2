import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os
import random

class MMFFDataset_Augmented(Dataset):
    """Dataset với augmentation mạnh hơn để chống overfit"""
    
    def __init__(self, 
                 skeleton_data_path,
                 label_path,
                 image_dir,
                 num_frames=300,
                 num_joints=25,
                 transform=None,
                 phase='train'):
        
        self.skeleton_data = np.load(skeleton_data_path, mmap_mode='r')
        
        with open(label_path, 'rb') as f:
            self.sample_names, labels = pickle.load(f)
        
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        self.label_map = {old_label: new_label for new_label, old_label in enumerate(sorted(unique_labels))}
        self.labels = [self.label_map[label] for label in labels]
        
        self.image_dir = image_dir
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.phase = phase
        
        if transform is None:
            if phase == 'train':
                # STRONGER AUGMENTATION
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # Random crop
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(15),  # Random rotation
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, 
                                         saturation=0.3, hue=0.1),  # Stronger color jitter
                    transforms.RandomGrayscale(p=0.1),  # Occasionally convert to grayscale
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225]),
                    transforms.RandomErasing(p=0.2)  # Random erasing
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        skeleton = self.skeleton_data[idx]
        label = self.labels[idx]
        sample_name = self.sample_names[idx]
        
        # Skeleton augmentation for training
        if self.phase == 'train':
            skeleton = self._augment_skeleton(skeleton)
        
        skeleton = self._process_skeleton(skeleton)
        
        # Load image
        image_path = os.path.join(self.image_dir, f"{sample_name}.jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_dir, f"{sample_name}.png")
        
        if not os.path.exists(image_path):
            image = Image.new('RGB', (224, 224), color='black')
        else:
            image = Image.open(image_path).convert('RGB')
        
        image = self.transform(image)
        
        skeleton = torch.from_numpy(skeleton).float()
        label = torch.tensor(label, dtype=torch.long)
        
        return skeleton, image, label
    
    def _augment_skeleton(self, skeleton):
        """Augment skeleton data"""
        C, T, V, M = skeleton.shape
        
        # Random temporal shift
        if random.random() > 0.5:
            shift = random.randint(-5, 5)
            if shift > 0:
                skeleton = np.concatenate([skeleton[:, shift:, :, :], 
                                         np.zeros((C, shift, V, M))], axis=1)
            elif shift < 0:
                skeleton = np.concatenate([np.zeros((C, -shift, V, M)), 
                                         skeleton[:, :shift, :, :]], axis=1)
        
        # Random scale
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            skeleton = skeleton * scale
        
        # Add small noise
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.01, skeleton.shape)
            skeleton = skeleton + noise
        
        return skeleton
    
    def _process_skeleton(self, skeleton):
        C, T, V, M = skeleton.shape
        
        if T < self.num_frames:
            pad_size = self.num_frames - T
            pad = np.zeros((C, pad_size, V, M))
            skeleton = np.concatenate([skeleton, pad], axis=1)
        elif T > self.num_frames:
            indices = np.linspace(0, T - 1, self.num_frames, dtype=int)
            skeleton = skeleton[:, indices, :, :]
        
        return skeleton.copy()


def get_dataloaders_augmented(data_dir, 
                              batch_size=16, 
                              num_workers=4,
                              num_frames=300,
                              num_joints=25):
    """Get dataloaders with strong augmentation"""
    train_dataset = MMFFDataset_Augmented(
        skeleton_data_path=os.path.join(data_dir, 'train_data.npy'),
        label_path=os.path.join(data_dir, 'train_label.pkl'),
        image_dir=os.path.join(data_dir, 'images'),
        num_frames=num_frames,
        num_joints=num_joints,
        phase='train'
    )
    
    val_dataset = MMFFDataset_Augmented(
        skeleton_data_path=os.path.join(data_dir, 'val_data.npy'),
        label_path=os.path.join(data_dir, 'val_label.pkl'),
        image_dir=os.path.join(data_dir, 'images'),
        num_frames=num_frames,
        num_joints=num_joints,
        phase='val'
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader