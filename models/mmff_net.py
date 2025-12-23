import torch
import torch.nn as nn
from models.skeleton import SkeletonStream_STGCN
from models.rgb import RGBStream_Xception
from models.attention import EarlyFusionModule
from models.fusion import CrossAttentionFusion

class MMFF_Net(nn.Module):
    """
    Multi-Modality Feature Fusion Network
    Kết hợp Skeleton Stream (ST-GCN) và RGB Stream (Xception)
    """
    def __init__(self, num_classes, num_joints=25, graph_args={'layout': 'ntu-rgb+d'}):
        super().__init__()
        
        # Skeleton Stream
        self.skeleton_stream = SkeletonStream_STGCN(
            in_channels=3,
            num_class=num_classes,
            graph_args=graph_args
        )
        
        # RGB Stream
        self.rgb_stream = RGBStream_Xception()
        
        # Early Fusion (Attention)
        self.early_fusion = EarlyFusionModule(
            in_channels=2048,
            num_joints=num_joints
        )
        
        # Late Fusion
        self.late_fusion = CrossAttentionFusion(
            skeleton_dim=256,
            rgb_dim=2048,
            fusion_dim=512
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, skeleton, rgb_image):
        """
        Args:
            skeleton: (N, C, T, V, M) - skeleton sequence
            rgb_image: (N, 3, H, W) - RGB image
        Returns:
            logits: (N, num_classes)
        """
        # Extract RGB features
        rgb_feat = self.rgb_stream(rgb_image)  # (N, 2048, H/32, W/32)
        
        # Early Fusion: Apply attention on RGB features
        rgb_feat = self.early_fusion(rgb_feat, skeleton)  # (N, 2048, H/32, W/32)
        
        # Extract skeleton features
        skeleton_feat = self.skeleton_stream(skeleton)  # (N, 256)
        
        # Late Fusion
        fused_feat = self.late_fusion(skeleton_feat, rgb_feat)  # (N, 512)
        
        # Classification
        logits = self.classifier(fused_feat)  # (N, num_classes)
        
        return logits