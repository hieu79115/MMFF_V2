import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionModule(nn.Module):
    """
    Self-Attention Module cho RGB stream
    Giúp network tập trung vào vùng chứa con người
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (N, C, H, W)
        attention_mask = self.conv(x)  # (N, 1, H, W)
        attention_mask = self.sigmoid(attention_mask)
        
        out = x * attention_mask
        return out


class SkeletonAttentionModule(nn.Module):
    """
    Skeleton Attention Module - Early Fusion
    Project skeleton sequence lên RGB frame để tạo attention mask
    """
    def __init__(self, in_channels, num_joints=25):
        super().__init__()
        self.num_joints = num_joints
        # Không cần learnable parameters, chỉ tính toán attention từ skeleton
    
    def forward(self, rgb_feature, skeleton_seq):
        """
        Args:
            rgb_feature: (N, C, H, W) - feature map từ RGB stream
            skeleton_seq: (N, 3, T, V, M) - skeleton sequence
        Returns:
            out: (N, C, H, W) - feature map sau khi apply attention
        """
        N, C, H, W = rgb_feature.shape
        N, _, T, V, M = skeleton_seq.shape
        
        # Tính moving distance của từng joint
        # Lấy frame đầu và frame giữa
        first_frame = skeleton_seq[:, :, 0, :, :]  # (N, 3, V, M)
        mid_frame = skeleton_seq[:, :, T//2, :, :]  # (N, 3, V, M)
        
        # Tính khoảng cách di chuyển
        distances = torch.norm(mid_frame - first_frame, dim=1)  # (N, V, M)
        distances = distances.mean(dim=2)  # Average across people: (N, V)
        
        # Tìm joint có khoảng cách lớn nhất
        max_dist_idx = torch.argmax(distances, dim=1)  # (N,)
        
        # Tạo attention mask dựa trên vị trí joint
        # Simplified: tạo một mask Gaussian centered tại vị trí joint
        attention_mask = torch.zeros(N, 1, H, W, device=rgb_feature.device)
        
        for i in range(N):
            # Lấy vị trí 2D của joint (giả sử đã được normalize về [0, 1])
            joint_idx = max_dist_idx[i]
            joint_pos = mid_frame[i, :2, joint_idx, 0]  # (2,) - x, y coordinates
            
            # Chuyển về pixel coordinates
            x_pos = int(joint_pos[0].item() * W)
            y_pos = int(joint_pos[1].item() * H)
            
            # Tạo Gaussian mask centered tại joint
            y_grid, x_grid = torch.meshgrid(
                torch.arange(H, device=rgb_feature.device),
                torch.arange(W, device=rgb_feature.device),
                indexing='ij'
            )
            
            sigma = min(H, W) * 0.15  # Adjust sigma based on feature map size
            gaussian = torch.exp(-((x_grid - x_pos)**2 + (y_grid - y_pos)**2) / (2 * sigma**2))
            attention_mask[i, 0] = gaussian
        
        # Normalize attention mask
        attention_mask = torch.sigmoid(attention_mask)
        
        # Apply attention
        out = rgb_feature * attention_mask
        
        return out


class EarlyFusionModule(nn.Module):
    """
    Early Fusion Module kết hợp Self-Attention và Skeleton-Attention
    """
    def __init__(self, in_channels, num_joints=25):
        super().__init__()
        self.self_attention = SelfAttentionModule(in_channels)
        self.skeleton_attention = SkeletonAttentionModule(in_channels, num_joints)
        
        # Combine features
        self.combine = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True)
    )

    def forward(self, rgb_feature, skeleton_seq):
        # Apply self-attention
        self_att_feature = self.self_attention(rgb_feature)
        
        # Apply skeleton-attention
        skeleton_att_feature = self.skeleton_attention(rgb_feature, skeleton_seq)
        
        # Concatenate and combine
        combined = torch.cat([self_att_feature, skeleton_att_feature], dim=1)
        out = self.combine(combined)
        
        return out