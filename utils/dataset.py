import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os

class MMFFDataset(Dataset):
    """
    Dataset cho MMFF (Multi-Modality Feature Fusion)
    """
    def __init__(self, 
                 skeleton_data_path,
                 label_path,
                 image_dir,
                 num_frames=300,
                 num_joints=25,
                 transform=None,
                 phase='train'):
        """
        Args:
            skeleton_data_path: đường dẫn đến file .npy chứa skeleton data
            label_path: đường dẫn đến file .pkl chứa labels
            image_dir: thư mục chứa ảnh RGB
            num_frames: số frame cần pad/crop
            num_joints: số khớp xương (25 cho NTU, 20 cho UT-MHAD)
            transform: transform cho ảnh
            phase: 'train' hoặc 'val'
        """
        self.skeleton_data = np.load(skeleton_data_path, mmap_mode='r')
        
        with open(label_path, 'rb') as f:
            self.sample_names, self.labels = pickle.load(f)
        
        self.image_dir = image_dir
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.phase = phase
        
        if transform is None:
            if phase == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((299, 299)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((299, 299)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Load skeleton data
        skeleton = self.skeleton_data[idx]  # Shape: (C, T, V, M)
        label = self.labels[idx]
        sample_name = self.sample_names[idx]
        
        # Pad or crop temporal dimension
        skeleton = self._process_skeleton(skeleton)
        
        # Load RGB image
        image_path = os.path.join(self.image_dir, f"{sample_name}.jpg")
        if not os.path.exists(image_path):
            # Fallback to .png if .jpg not found
            image_path = os.path.join(self.image_dir, f"{sample_name}.png")
        
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Convert to tensors
        skeleton = torch.from_numpy(skeleton).float()
        label = torch.tensor(label, dtype=torch.long)
        
        return skeleton, image, label
    
    def _process_skeleton(self, skeleton):
        """
        Xử lý skeleton data: pad hoặc crop temporal dimension
        Input shape: (C, T, V, M)
        Output shape: (C, num_frames, V, M)
        """
        C, T, V, M = skeleton.shape
        
        if T < self.num_frames:
            # Pad
            pad_size = self.num_frames - T
            pad = np.zeros((C, pad_size, V, M))
            skeleton = np.concatenate([skeleton, pad], axis=1)
        elif T > self.num_frames:
            # Sample frames uniformly
            indices = np.linspace(0, T - 1, self.num_frames, dtype=int)
            skeleton = skeleton[:, indices, :, :]
        
        return skeleton


def get_dataloaders(data_dir, 
                   batch_size=16, 
                   num_workers=4,
                   num_frames=300,
                   num_joints=25):
    """
    Tạo DataLoader cho train và validation
    """
    train_dataset = MMFFDataset(
        skeleton_data_path=os.path.join(data_dir, 'train_data.npy'),
        label_path=os.path.join(data_dir, 'train_label.pkl'),
        image_dir=os.path.join(data_dir, 'images'),
        num_frames=num_frames,
        num_joints=num_joints,
        phase='train'
    )
    
    val_dataset = MMFFDataset(
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
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader