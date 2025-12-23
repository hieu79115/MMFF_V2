import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.graph import Graph

class ConvTemporalGraphical(nn.Module):
    """Spatial temporal graph convolutional layer"""
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1,
                 t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias
        )
    
    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        
        return x.contiguous(), A


class STGCNBlock(nn.Module):
    """ST-GCN block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 dropout=0, residual=True):
        super().__init__()
        
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)
                ),
                nn.BatchNorm2d(out_channels),
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        
        return self.relu(x), A


class SkeletonStream_STGCN(nn.Module):
    """
    ST-GCN Backbone cho Skeleton Stream
    Theo paper: output shape (256, T/4, V) -> Global Average Pooling -> (256,)
    """
    def __init__(self, in_channels=3, num_class=60, graph_args={'layout': 'ntu-rgb+d'},
                 edge_importance_weighting=True, **kwargs):
        super().__init__()
        
        # Load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        
        # Build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        
        self.st_gcn_networks = nn.ModuleList((
            STGCNBlock(in_channels, 64, kernel_size, 1, residual=False, **kwargs),
            STGCNBlock(64, 64, kernel_size, 1, **kwargs),
            STGCNBlock(64, 64, kernel_size, 1, **kwargs),
            STGCNBlock(64, 64, kernel_size, 1, **kwargs),
            STGCNBlock(64, 128, kernel_size, 2, **kwargs),
            STGCNBlock(128, 128, kernel_size, 1, **kwargs),
            STGCNBlock(128, 128, kernel_size, 1, **kwargs),
            STGCNBlock(128, 256, kernel_size, 2, **kwargs),
            STGCNBlock(256, 256, kernel_size, 1, **kwargs),
            STGCNBlock(256, 256, kernel_size, 1, **kwargs),
        ))
        
        # Edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        # Input shape: (N, C, T, V, M)
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # (N, M, V, C, T)
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # (N, M, C, T, V)
        x = x.view(N * M, C, T, V)
        
        # Forward through ST-GCN blocks
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        
        # Global pooling: (N*M, C, T, V) -> (N*M, C, 1, 1) -> (N*M, C)
        x = self.global_pool(x)
        x = x.view(N, M, -1).mean(dim=1)  # Average across people
        
        return x  # Shape: (N, 256)