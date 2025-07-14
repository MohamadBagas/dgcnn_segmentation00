import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def knn(x, k):
    """Find k-nearest neighbors."""
    batch_size, dims, num_points = x.shape
    k = min(k, num_points)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   
    return idx

def get_graph_feature(x, k=20, idx=None):
    """
    Compute edge features for each point's k-NN graph.
    """
    batch_size, num_dims, num_points = x.size()
    k = min(k, num_points)

    if idx is None:
        idx = knn(x, k=k)
            
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
 
    x_t = x.transpose(2, 1).contiguous()
    feature = x_t.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x_t.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((x, feature - x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

class DGCNNSemSeg(nn.Module):
    """
    Deeper DGCNN segmentation model with 5 EdgeConv layers.
    """
    def __init__(self, num_classes, num_features=7, k=20):
        super(DGCNNSemSeg, self).__init__()
        self.k = k
        self.num_features = num_features
        
        self.conv1 = nn.Sequential(nn.Conv2d(num_features * 2, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        
        self.seg_head = nn.Sequential(
            nn.Conv1d(1024 + 512, 512, 1, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2), nn.Dropout(0.5),
            nn.Conv1d(512, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.5),
            nn.Conv1d(256, num_classes, 1, bias=False)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        batch_size, _, num_points = x.shape
        
        graph1 = get_graph_feature(x, k=self.k)
        x1 = self.conv1(graph1).max(dim=-1, keepdim=False)[0]

        graph2 = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(graph2).max(dim=-1, keepdim=False)[0]

        graph3 = get_graph_feature(x2, k=self.k)
        x3 = self.conv3(graph3).max(dim=-1, keepdim=False)[0]
        
        graph4 = get_graph_feature(x3, k=self.k)
        x4 = self.conv4(graph4).max(dim=-1, keepdim=False)[0]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)

        x_global_features = self.conv5(x_cat)
        x_global = F.adaptive_max_pool1d(x_global_features, 1)

        x_global_repeated = x_global.repeat(1, 1, num_points)

        x_combined = torch.cat((x_cat, x_global_repeated), dim=1)
        
        seg_pred = self.seg_head(x_combined)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        
        return seg_pred
