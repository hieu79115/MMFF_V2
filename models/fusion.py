import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion Module cho Late Fusion
    Theo paper: explore correlation giữa skeleton và RGB features
    """
    def __init__(self, skeleton_dim=256, rgb_dim=2048, fusion_dim=512):
        super().__init__()
        
        # Project features to same dimension
        self.skeleton_proj = nn.Linear(skeleton_dim, fusion_dim)
        self.rgb_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(rgb_dim, fusion_dim)
        )
        
        # Cross-attention
        self.query_conv = nn.Linear(fusion_dim, fusion_dim)
        self.key_conv = nn.Linear(fusion_dim, fusion_dim)
        self.value_conv = nn.Linear(fusion_dim, fusion_dim)
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
    def forward(self, skeleton_feat, rgb_feat):
        """
        Args:
            skeleton_feat: (N, 256) từ ST-GCN
            rgb_feat: (N, 2048, H, W) từ Xception
        Returns:
            fused_feat: (N, fusion_dim)
        """
        # Project to same dimension
        skeleton_feat = self.skeleton_proj(skeleton_feat)  # (N, fusion_dim)
        rgb_feat = self.rgb_proj(rgb_feat)  # (N, fusion_dim)
        
        # Compute attention
        # Skeleton attends to RGB
        q_skel = self.query_conv(skeleton_feat)  # (N, fusion_dim)
        k_rgb = self.key_conv(rgb_feat)  # (N, fusion_dim)
        v_rgb = self.value_conv(rgb_feat)  # (N, fusion_dim)
        
        attention_scores = torch.matmul(q_skel.unsqueeze(1), k_rgb.unsqueeze(2))  # (N, 1, 1)
        attention_scores = attention_scores.squeeze() / (skeleton_feat.size(1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        attended_rgb = attention_weights.unsqueeze(1) * v_rgb  # (N, fusion_dim)
        
        # RGB attends to Skeleton
        q_rgb = self.query_conv(rgb_feat)
        k_skel = self.key_conv(skeleton_feat)
        v_skel = self.value_conv(skeleton_feat)
        
        attention_scores = torch.matmul(q_rgb.unsqueeze(1), k_skel.unsqueeze(2))
        attention_scores = attention_scores.squeeze() / (rgb_feat.size(1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        attended_skel = attention_weights.unsqueeze(1) * v_skel
        
        # Concatenate and fuse
        fused = torch.cat([attended_rgb, attended_skel], dim=1)  # (N, fusion_dim*2)
        fused = self.out_proj(fused)  # (N, fusion_dim)
        
        return fused