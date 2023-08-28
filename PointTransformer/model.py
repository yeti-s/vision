import torch
import torch.nn as nn
import torch.nn.functional as F


def get_points(points, idx):
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)

def farthest_point_sampling(points, n_points):
    device = points.device
    b, n, _ = points.size()
    distance = torch.ones((b, n)).to(device) * 1e10
    sampled = torch.ones((b, n_points)).long().to(device) * -1
    sample_idx = torch.randint(0, n, (b,)).to(device)
    
    for i in range(n_points):
        sampled[:,i] = sample_idx
        point = points[torch.arange(b).to(device), sample_idx,:].view(b, 1, -1)
        dist = torch.sum((points - point) ** 2, -1)
        distance = torch.min(dist, distance)
        sample_idx = torch.argmax(distance, -1)
        
    return sampled

def knn_point_sampling(points, k):
    distance = torch.cdist(points, points) # (b, n, d_points)
    return distance.argsort()[:,:,:k]

class SingleMLP(nn.Module):
    def __init__(self, d_feats, d_points = 3) -> None:
        super().__init__()
        
        self.fc = nn.Linear(d_points, d_feats)
        self.bn = nn.BatchNorm1d(d_feats)
    
    # x (b, n, d_points)
    def forward(self, x):
        x = self.fc(x)
        
        x.transpose_(1, 2)
        x = F.gelu(self.bn(x))
        
        x.transpose_(1, 2)
        return x

class PointTransformerLayer(nn.Module):
    def __init__(self, d_feats, k, d_points = 3) -> None:
        super(PointTransformerLayer, self).__init__()
        
        self.k = k
        self.query = nn.Linear(d_feats, d_feats)
        self.key = nn.Linear(d_feats, d_feats)
        self.value = nn.Linear(d_feats, d_feats)
        
        self.fc_delta1 = nn.Linear(d_points, d_points)
        self.fc_delta2 = nn.Linear(d_points, d_feats)
        self.bn_delta = nn.BatchNorm2d(d_points)
        
        self.fc_gamma1 = nn.Linear(d_feats, d_feats)
        self.fc_gamma2 = nn.Linear(d_feats, d_feats)
        self.bn_gamma1 = nn.BatchNorm2d(d_feats)
        self.bn_gamma2 = nn.BatchNorm2d(d_feats)

    
    # feats (b, n, d_feats), points (b, n, d_points)
    def forward(self, feats, points):

        knn_idx = knn_point_sampling(points, self.k) # (b, n, k)
        knn_feats = get_points(feats, knn_idx) # (b, n, k, d_feats)
        knn_points = get_points(points, knn_idx) # (b, n, k, d_points)
        
        # position encoding
        pos_encoding = points.unsqueeze(2) - knn_points # (b, n, k, d_points)
        pos_encoding = self.fc_delta1(pos_encoding) # (b, n, k, d_feats)
        # (b, d_feats, k, n)
        pos_encoding = pos_encoding.permute(0, 3, 2, 1) # 
        pos_encoding = F.gelu(self.bn_delta(pos_encoding))
        # (b, n, k, d_feats)
        pos_encoding = pos_encoding.permute(0, 3, 2, 1)
        pos_encoding = self.fc_delta2(pos_encoding)
        
        # self attention
        query = self.query(feats) # (b, n, d_feats)
        # (b, n, k, d_feats)
        key = self.key(knn_feats)
        
        gamma = query.unsqueeze(2) - key + pos_encoding # (b, n, k, d_feats)
        # (b, d_feats, k, n)
        gamma = gamma.permute(0, 3, 2, 1) 
        gamma = F.gelu(self.bn_gamma1(gamma))
        # (b, n, k, d_feats)
        gamma = gamma.permute(0, 3, 2, 1)
        gamma = self.fc_gamma1(gamma)
        # (b, d_feats, k, n)
        gamma = gamma.permute(0, 3, 2, 1) 
        gamma = F.gelu(self.bn_gamma2(gamma))
        # (b, n, k, d_feats)
        gamma = gamma.permute(0, 3, 2, 1)
        gamma = self.fc_gamma2(gamma)
        
        rho = F.softmax(gamma, dim=-2)
        value = self.value(knn_feats) + pos_encoding
        
        return torch.einsum('bnkd,bnkd->bnd', rho, value)
        
class PointTransformerBlock(nn.Module):
    def __init__(self, d_feats, k, d_points = 3) -> None:
        super(PointTransformerBlock, self).__init__()
        
        self.transformerLayer = PointTransformerLayer(d_feats, k, d_points)
        self.fc1 = nn.Linear(d_feats, d_feats)
        self.fc2 = nn.Linear(d_feats, d_feats)
        self.bn1 = nn.BatchNorm1d(d_feats)
        self.bn2 = nn.BatchNorm1d(d_feats)
        self.bn3 = nn.BatchNorm1d(d_feats)
    
    def forward(self, feats, points):
        # (b, n, d_feats)
        out = self.fc1(feats)
        # (b, d_feats, n)
        out.transpose_(1, 2)
        out = F.gelu(self.bn1(out))
        # (b, n, d_feats)
        out.transpose_(1, 2)
        out = self.transformerLayer(out, points)
        # (b, d_feats, n)
        out.transpose_(1, 2)
        out = F.gelu(self.bn2(out))
        # (b, n, d_feats)
        out.transpose_(1, 2)
        out = self.fc2(out)
        # (b, d_feats, n)
        out.transpose_(1, 2)
        out = F.gelu(self.bn3(out))
        # (b, n, d_feats)
        out.transpose_(1, 2)
        feats = feats + out
        
        return feats, points

class TransitionDown(nn.Module):
    def __init__(self, in_feats, out_feats, k, d_points=3) -> None:
        super(TransitionDown, self).__init__()
        
        self.k = k
        self.d_points = d_points
        
        self.fc = nn.Linear(in_feats + d_points, out_feats)
        self.bn = nn.BatchNorm2d(out_feats)
    
    # x (b, n, in_feats) , points (b, n, d_points)
    def forward(self, feats, points):
        b, n, in_feats = feats.size()
        
        knn_idx = knn_point_sampling(points, self.k) # (b, n, k)
        knn_points = get_points(points, knn_idx) # (b, n, k, d_points)
        knn_feats = get_points(feats, knn_idx) # (b, n, k, in_feats)
        
        n_points = n // 4
        p2_idx = farthest_point_sampling(points, n_points) # (b, n_points)
        p2_points = get_points(points, p2_idx)
        # (b, n, k, d_points) , (b, n, k, d_feats)
        knn_p2_points = torch.gather(knn_points, 1, p2_idx.view(b, n_points, 1, 1).expand(-1, -1, self.k, self.d_points))
        knn_p2_feats = torch.gather(knn_feats, 1, p2_idx.view(b, n_points, 1, 1).expand(-1, -1, self.k, in_feats))
        
        # why (d_points + in_feats) ?
        x = torch.concatenate([knn_p2_points, knn_p2_feats], dim=-1) # (b, n_points, k, d_points + in_feats)
        x = self.fc(x) # (b, n_points, k ,out_feats)
        x = x.permute(0, 3, 2, 1) # (b, out_feats, k, n_points)
        x = F.gelu(self.bn(x))
        
        # (b, n_points, out_feats)
        x = torch.max(x, dim=2).values
        x = x.permute(0, 2, 1)
        return x, p2_points
    
    
class PointTransformer(nn.Module):
    def __init__(self, d_points = 3, k = 16) -> None:
        super().__init__()
        
        self.d_points = d_points
        
        d_feats = 32
        
        self.mlp = SingleMLP(d_feats)
        blocks = [PointTransformerBlock(d_feats, k, d_points)]
        for _ in range(4):
            blocks.append(TransitionDown(d_feats, d_feats * 2, k, d_points))
            blocks.append(PointTransformerBlock(d_feats * 2, k, d_points))
            d_feats *= 2
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x):
        points = x[:,:,:self.d_points]
        feats = self.mlp(points)
        
        for block in self.blocks:
            feats, points = block(feats, points)
        return feats
        
        
points = torch.rand(4, 4096, 3).cuda()
model = PointTransformer().cuda()
feats = model(points)
print(feats.shape)