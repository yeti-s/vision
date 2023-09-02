import torch
import torch.nn as nn
import torch.nn.functional as F
import knn_cuda

DEFAULT_KNN_BUFFER_SIZE = 1000000000
EPS = 1e-8

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


def knn_point_sampling(ref, query, k, buffer_size = DEFAULT_KNN_BUFFER_SIZE):
    device = ref.device
    ref = ref.cuda()
    query = query.cuda()
    
    r_b, r_n, _ = ref.size()
    q_b, q_n, _ = query.size()
    batch_size = buffer_size // (r_b * r_n * q_b)

    s_idx = 0
    e_idx = batch_size
    knn = knn_cuda.KNN(k, True)
    knn_dist_list = []
    knn_idx_list = []
    while s_idx < q_n:
        e_idx = q_n if e_idx > q_n else e_idx
        
        knn_dist, knn_idx = knn(ref, query[:,s_idx:e_idx,:])
        knn_dist_list.append(knn_dist)
        knn_idx_list.append(knn_idx)
        torch.cuda.empty_cache()
        
        s_idx = e_idx
        e_idx += batch_size
         
    dist = torch.concatenate(knn_dist_list,dim=1).to(device)
    idx = torch.concatenate(knn_idx_list, dim=1).to(device)
    torch.cuda.empty_cache()
    
    return dist, idx

# 1 - high_resolution, 2 - low resolution
def trilinear_interpolate(points1, feats2, points2, k = 3):
    b, n, _ = points1.size()
    
    dist, idx = knn_point_sampling(points2, points1, k)
    dist_recip = 1.0 / (dist + EPS)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    new_feats = torch.sum(get_points(feats2, idx) * weight.view(b, n, k, 1), dim=2)
    
    return new_feats

class SingleMLP(nn.Module):
    def __init__(self, in_feats, out_feats, batch_norm=True) -> None:
        super().__init__()
        
        self.batch_norm = batch_norm
        
        self.fc = nn.Linear(in_feats, out_feats)
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feats)
    
    # x (b, n, d_points)
    def forward(self, x):
        x = self.fc(x)
        
        if self.batch_norm:
            x.transpose_(1, 2)
            x = F.gelu(self.bn(x))
            x.transpose_(1, 2)
        
        return x

class PointTransformerLayer(nn.Module):
    def __init__(self, d_points, d_feats, k) -> None:
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

        _, knn_idx = knn_point_sampling(points, points, self.k) # (b, n, k)
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
    def __init__(self, d_points, d_feats, k) -> None:
        super(PointTransformerBlock, self).__init__()
        
        self.transformerLayer = PointTransformerLayer(d_points, d_feats, k)
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
    def __init__(self, in_feats, out_feats, k, d_points) -> None:
        super(TransitionDown, self).__init__()
        
        self.k = k
        self.d_points = d_points
        
        self.fc = nn.Linear(in_feats + d_points, out_feats)
        self.bn = nn.BatchNorm2d(out_feats)
    
    # feats (b, n, in_feats) , points (b, n, d_points)
    def forward(self, feats, points):
        b, n, in_feats = feats.size()
        
        # farthest point sampling
        n_points = n // 4
        p2_idx = farthest_point_sampling(points, n_points) # (b, n_points)
        p2_points = get_points(points, p2_idx)
        # knn of sampled points
        _, knn_idx = knn_point_sampling(points, p2_points, self.k) # (b, n_points, k)
        p2_knn_points = get_points(points, knn_idx) # (b, n_points, k, d_points)
        p2_knn_feats = get_points(feats, knn_idx) # (b, n_points, k, in_feats)
        # why (d_points + in_feats) ?
        x = torch.concatenate([p2_knn_points, p2_knn_feats], dim=-1) # (b, n_points, k, d_points + in_feats)
        x = self.fc(x) # (b, n_points, k ,out_feats)
        x = x.permute(0, 3, 2, 1) # (b, out_feats, k, n_points)
        x = F.gelu(self.bn(x))
        # (b, n_points, out_feats)
        x = torch.max(x, dim=2).values
        x = x.permute(0, 2, 1)
        return x, p2_points

# feat2, point2 is high resolution
class TransitionUp(nn.Module):
    def __init__(self, in_feats, out_feats, d_points=3) -> None:
        super(TransitionUp, self).__init__()
        
        self.d_points = d_points
        
        self.f1_fc = nn.Linear(out_feats, out_feats)
        self.f1_bn = nn.BatchNorm1d(out_feats)
        
        self.f2_fc = nn.Linear(in_feats, out_feats)
        self.f2_bn = nn.BatchNorm1d(out_feats)
    
    # feats1 (b, 4n, out_feats), feats2 (b, n, in_feats)
    def forward(self, feats1, points1, feats2, points2):
        
        feats1 = self.f1_fc(feats1)
        feats1 = feats1.transpose(1, 2)
        feats1 = F.relu(self.f1_bn(feats1))
        feats1 = feats1.transpose(1, 2) # (b, 4n, out_feats)
        
        feats2 = self.f2_fc(feats2)
        feats2 = feats2.transpose(1, 2)
        feats2 = F.relu(self.f2_bn(feats2))
        feats2 = feats2.transpose(1, 2) # (b, n, out_feats)
        feats2 = trilinear_interpolate(points1, feats2, points2, self.d_points) # (b, 4n, out_feats)
        
        feats = feats1 + feats2
        return feats, points1 # (b, 4n, out_feats) , (b, 4n, d_points)
    
    

class TransitionDownBlock(nn.Module):
    def __init__(self, d_points, in_feats, k):
        super(TransitionDownBlock, self).__init__()

        self.transition = TransitionDown(in_feats, in_feats*2, k, d_points)
        self.transformer = PointTransformerBlock(d_points, in_feats*2, k)
    
    def forward(self, feats, points):
        feats, points = self.transition(feats, points)
        feats, points = self.transformer(feats, points)
        return feats, points

class TransitionUpBlock(nn.Module):
    def __init__(self, d_points, in_feats, k):
        super(TransitionUpBlock, self).__init__()

        self.transition = TransitionUp(in_feats, in_feats//2, d_points)
        self.transformer = PointTransformerBlock(d_points, in_feats//2, k)
    
    # feats1 (b, 4n, out_feats), feats2 (b, n, in_feats)
    def forward(self, feats1, points1, feats2, points2):
        feats, points = self.transition(feats1, points1, feats2, points2)
        feats, points = self.transformer(feats, points)
        return feats, points

class PointTransformer(nn.Module):
    def __init__(self, d_points = 3, k = 16) -> None:
        super().__init__()
        
        d_feats = 32
        blocks = [PointTransformerBlock(d_points, d_feats, k)]
        for _ in range(4):
            blocks.append(TransitionDownBlock(d_points, d_feats, k))
            d_feats *= 2
            
        self.d_points = d_points
        self.mlp = SingleMLP(d_points, 32)
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x):
        points = x[:,:,:self.d_points]
        feats = self.mlp(points)
        
        for block in self.blocks:
            feats, points = block(feats, points)
        return feats


class PointTransformerSeg(nn.Module):
    def __init__(self, d_in, d_out, d_points = 3, k = 16) -> None:
        super(PointTransformerSeg, self).__init__()
        
        d_feats = 32
        self.d_in = d_in
        self.d_points = d_points
        self.mlp1 = SingleMLP(d_in, d_feats)
        self.transformer1 = PointTransformerBlock(d_points, d_feats, k)
        
        down_blocks = []
        up_blocks = []
        for _ in range(4):
            down_blocks.append(TransitionDownBlock(d_points, d_feats, k))
            up_blocks.insert(0, TransitionUpBlock(d_points, d_feats*2, k))
            d_feats *= 2
            
        self.mlp2 = SingleMLP(d_feats, d_feats)
        self.transformer2 = PointTransformerBlock(d_points, d_feats, k)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks)
        self.mlp3 = SingleMLP(32, d_out, False)
    
    def forward(self, x):
        points = x[:,:,:self.d_points]
        feats = x[:,:,:self.d_in]
        feats = self.mlp1(feats)
        feats, points = self.transformer1(feats, points)
        
        out_feats, out_points = [feats], [points]
        
        for i in range(len(self.down_blocks)):
            block = self.down_blocks[i]
            feats, points = block(feats, points)
            if i < len(self.down_blocks) - 1:
                out_feats.insert(0, feats)
                out_points.insert(0, points)
        
        feats = self.mlp2(feats)
        feats2, points2 = self.transformer2(feats, points)
        
        for i in range(len(self.up_blocks)):
            block = self.up_blocks[i]
            feats1, points1 = out_feats[i], out_points[i]
            feats2, points2 = block(feats1, points1, feats2, points2)
        
        out = self.mlp3(feats2)
        return out

        
# points = torch.rand(8, 4096, 3).cuda()
# model = PointTransformer().cuda()
# feats = model(points)
# print(feats.shape, feats)

# points = torch.rand(1, 52584, 4).cuda()
# model = PointTransformerSeg(4, 2).cuda()
# feats = model(points)
# print(feats.shape)