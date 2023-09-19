import torch
import torch.nn as nn
import torch.nn.functional as F

# k - patch size, s - stride, p - padding
# def overlap_patch(x, k, s, p):
  
def img_to_attention(t):
    """(b, c, h, w) -> (b, n, dim)"""
    b, dim, _, _ = t.size()
    t = t.permute(0, 2, 3, 1).contiguous().view(b, -1, dim)
    return t

def attention_to_img(t, h, w):
    """(b, n, dim) -> (b, c, h, w)"""
    b, _, _ = t.size()
    t = t.permute(0, 2, 1).contiguous().view(b, -1, h, w)
    return t
  
# x : (b, dim, h, w), out : (b, dim, h, w)
def pad_tensor(x, h, w):
    d_y = h - x.size(2)
    d_x = w - x.size(3)
    return F.pad(x, [0, 0, 0, 0, d_x//2, d_x-d_x//2, d_y//2, d_y-d_y//2])
  
class OverlapPatch(nn.Module):
    """x : (b, c, h, w), return : (b, dim, h, w)"""
    def __init__(self, in_dim, out_dim, patch_size, stride, padding) -> None:
        super(OverlapPatch, self).__init__()
        
        self.conv = nn.Conv2d(in_dim, out_dim, patch_size, stride, padding)
        self.layer_norm = nn.LayerNorm(out_dim)
    
    def forward(self, x):
        x = self.conv(x)
        _, _, h, w = x.size()
        x = self.layer_norm(img_to_attention(x))
        
        return attention_to_img(x, h, w)

class ESALayer(nn.Module):
    """x : (b, n, dim), return : (b, n, dim)"""
    def __init__(self, dim, ratio) -> None:
        super(ESALayer, self).__init__()
        
        self.ratio = ratio
        self.layer_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim*ratio, dim)
        self.value = nn.Linear(dim*ratio, dim)
        self.out = nn.Linear(dim, dim)
    
    def forward(self, x):
        b, n, dim = x.size()
        dim_square = torch.sqrt(torch.Tensor([dim])).to(x.device)
        x = self.layer_norm(x)
        query = self.query(x)
        
        x = x.view(b, -1, dim * self.ratio)
        # (b, n/R, dim)
        key = self.key(x) 
        value = self.value(x)
        # (b, n, n/R)
        attn_score = torch.matmul(query, key.transpose(-2, -1)) / dim_square
        # (b, n, dim)
        attn_value = torch.matmul(F.softmax(attn_score), value)
        x = self.out(attn_value)
        
        return x
    

class MixFFN(nn.Module):
    """ x : (b, n, dim) and h : height, w : width, return : (b, n, dim) """
    def __init__(self, dim) -> None:
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(dim)
        self.mlp1 = nn.Linear(dim, dim)
        self.mlp2 = nn.Linear(dim, dim)
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim) # depth-wise
    
    def forward(self, x, h, w):
        x = self.layer_norm(x)
        skip = x
        
        x = self.mlp1(x)
        x = attention_to_img(x, h, w)
        x = self.conv(x)
        x = img_to_attention(x)
        x = self.mlp2(x)

        return x + skip
    
class DecoderBlock(nn.Module):
    """x : (b, dim, h, w), o_h : original height, o_w : original width , return : (b, dim, h, w)"""
    def __init__(self, in_dim, out_dim) -> None:
        super(DecoderBlock, self).__init__()
        
        self.fc = nn.Linear(in_dim, out_dim)
    
    def forward(self, x, o_h, o_w):
        _, _, h, w = x.size()
        x = self.fc(img_to_attention(x))
        x = attention_to_img(x, h, w)
        x = F.interpolate(x, (o_h, o_w), mode='bilinear', align_corners=True)
        
        return x
    
    
class EncoderBlock(nn.Module):
    """x : (b, dim, h, w), return : (b, dim, h, w)"""
    def __init__(self, n_layers, dim, ratio) -> None:
        super(EncoderBlock, self).__init__()
        
        layers = []
        for _ in range(n_layers):
            layers.append(ESALayer(dim, ratio))
            layers.append(MixFFN(dim))
        self.layers = nn.ModuleList(layers)
        self.n_layers = n_layers
        
    def forward(self, x):
        _, _, h, w = x.size()
        x = img_to_attention(x)
        
        for i in range(self.n_layers):
            esa_layer = self.layers[i*2]
            ffn_layer = self.layers[i*2+1]
            x = x + esa_layer(x)
            x = x + ffn_layer(x, h, w)
            
        return attention_to_img(x, h, w)

class MixTransformer(nn.Module):
    def __init__(self, in_channels, dims, n_layers, ratios) -> None:
        super(MixTransformer, self).__init__()
        
        overlap_patches = []
        encoder_blocks = []
        for i in range(len(dims)):
            in_dim = in_channels if i  == 0 else dims[i-1]
            patch_size = 3 if i > 0 else 7
            stride = 2 if i > 0 else 4
            padding = 1 if i > 0 else 3
            overlap_patches.append(OverlapPatch(in_dim, dims[i], patch_size, stride, padding))
            encoder_blocks.append(EncoderBlock(n_layers[i], dims[i], ratios[i]))
        
        self.overlap_patches = nn.ModuleList(overlap_patches)
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        
    def forward(self, x):
        outputs = []
        
        for i in range(len(self.overlap_patches)):
            x = self.overlap_patches[i](x)
            x = self.encoder_blocks[i](x)
            outputs.append(x)
        
        return outputs

class SegDecoder(nn.Module):
    def __init__(self, in_dims, n_classes) -> None:
        super(SegDecoder, self).__init__()
        out_dim = in_dims[0]
        
        self.decoder_blocks = nn.ModuleList([DecoderBlock(in_dims[i], out_dim) for i in range(len(in_dims))])
        self.mlp = nn.Linear(out_dim*4, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.conv = nn.Conv2d(out_dim, n_classes, 1)
        
        
    def forward(self, features, height, width):
        outputs = []
        for i in range(len(self.decoder_blocks)):
            outputs.append(self.decoder_blocks[i](features[i], features[0].size(2), features[0].size(3)))
            
        x = torch.cat(outputs, dim=1)
        _, _, h, w = x.size()
        
        x = self.mlp(img_to_attention(x))
        x = self.layer_norm(x)
        
        x = attention_to_img(x, h, w)
        x = F.interpolate(x, (height, width), mode='bilinear', align_corners=True)
        
        return self.conv(x)

class SegFormer(nn.Module):
    def __init__(self, in_channels, dims, n_layers, ratios, n_classes) -> None:
        super(SegFormer, self).__init__()
        
        self.encoder = MixTransformer(in_channels, dims, n_layers, ratios)
        self.decoder = SegDecoder(dims, n_classes)
        
    def forward(self, x):
        _, _, h, w = x.size()
        
        features = self.encoder(x)
        return self.decoder(features, h, w)
        


# # num_heads = [1, 2, 5, 8]
# n_layers = [3, 4, 6, 8]
# dims = [64, 128, 320, 512]
# # mlp_ratios=[4, 4, 4, 4]
# ratios=[8, 4, 2, 1]

# model = SegFormer(3, dims, n_layers, ratios, 2).cuda()
# t = torch.rand(2, 3, 1600, 256).cuda()
# print(model(t).shape)

