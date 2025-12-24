import torch
import torch.nn as nn
import torchvision.models as models

class RGBStream_ResNet18(nn.Module):
    """
    ResNet18 backbone cho RGB Stream - Nhẹ hơn Xception
    Phù hợp cho dataset nhỏ
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove the final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Output: (N, 512, H/32, W/32)
        
    def forward(self, x):
        """
        Args:
            x: (N, 3, H, W)
        Returns:
            features: (N, 512, H/32, W/32)
        """
        return self.features(x)