import torch
import torch.nn as nn
import torch.nn.functional as F

# code from https://github.com/NVlabs/SegFormer/tree/master

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape # N -> HxW
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

def resize(input,
           size=None,
           scale_factor=None,
           mode='bilinear',
           align_corners=None,
           warning=True):

    return F.interpolate(input, size, scale_factor, mode, align_corners)

def stochastic_depth(input: torch.Tensor, p: float,
                     mode: str, training: bool =  True):
    
    if not training or p == 0.0:
        # print(f'not adding stochastic depth of: {p}')
        return input
    
    survival_rate = 1.0 - p
    if mode == 'row':
        shape = [input.shape[0]] + [1] * (input.ndim - 1) # just converts BXCXHXW -> [B,1,1,1] list
    elif mode == 'batch':
        shape = [1] * input.ndim

    noise = torch.empty(shape, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    # print(f'added sDepth of: {p}')
    return input * noise

'''
Mix Transformer Encoder
'''
class Mlp(nn.Module):
    '''
    Implements Eq 3 in SegFormer Paper
    '''
    def __init__(self, inFeats, hidFeats=None, outFeats=None, act_layer = nn.GELU, drop=0.0):
        super(Mlp, self).__init__()
        outFeats = outFeats or inFeats
        hidFeats = hidFeats or inFeats

        self.fc1 = nn.Linear(inFeats, hidFeats)
        self.dwconv = DWConv(hidFeats)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidFeats, outFeats)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
 
        return x

class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qkv_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super(Attention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = self.dim // self.num_heads
        self.scale = qkv_scale or head_dim ** -0.5 # as in Eq 1 -> (1/(d_head)**1/2)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2*dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape # BxNxC
        q = self.q(x).reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3) # BxNxhx(C/h) -> BxhxNx(C/h)

        if self.sr_ratio > 1: # reduction ratio in EQ 2.
            x_ = x.permute(0,2,1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0,2,1) # BxCxHxW -> BxCxN -> BxNxC
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
            # BxNxC -> Bx-1x2xhx(C/h) -> 2xBxhx-1x(C/h) i.e. seprating Keys and values BxhxNxC/h
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2,-1)) * self.scale # EQ 1 in paper
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):

    def __init__(self,  dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0., sr_ratio=1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Block, self).__init__()

        self.norm1 = norm_layer(dim)

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qkv_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.norm2 = norm_layer(dim)

        ml_hid_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, ml_hid_dim, act_layer=act_layer, drop=drop)

    
    def forward(self, x, H, W):

        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)

        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, padding=3, inChannels=3, embed_dim=768):
        super(OverlapPatchEmbed, self).__init__()
        patch_size = (patch_size, patch_size)

        self.patch_size = patch_size

        self.proj = nn.Conv2d(inChannels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1,2) # B*C*H*W -> B*C*HW -> B*HW*C
        x = self.norm(x)

        return x, H, W


class MixVisionTransformer(nn.ModuleDict):

    def __init__(self, inChannels=3, embed_dims=[64, 128, 320, 512],
                 num_heads=[1,2,4,8], mlp_ratio=[4,4,4,4], qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.0, depths=[3,4,6,3], sr_ratio=[8,4,2,1], norm_layer=nn.LayerNorm):
        super(MixVisionTransformer, self).__init__()

        self.depths = depths

        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, padding=3, inChannels=inChannels, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, padding=1, inChannels=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, padding=1, inChannels=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, padding=1, inChannels=embed_dims[2], embed_dim=embed_dims[3])

        # stochastic depth decay rule (similar to linear decay) / just like matplot linspace
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio[0], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[cur+i], sr_ratio=sr_ratio[0], norm_layer=norm_layer
        ) for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio[1], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[cur+i], sr_ratio=sr_ratio[1], norm_layer=norm_layer
        ) for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio[2], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[cur+i], sr_ratio=sr_ratio[2], norm_layer=norm_layer
        ) for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio[3], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[cur+i], sr_ratio=sr_ratio[3], norm_layer=norm_layer
        ) for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1 
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs
    
    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x
    
class MLP(nn.Module):
    def __init__(self, inputDim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(inputDim, embed_dim)
    
    def forward(self, x):
        x = x.flatten(2).transpose(1,2)# B*C*H*W -> B*C*HW -> B*HW*C
        x = self.proj(x)
        return x
    
    
class SegFormerHead(nn.Module):

    def __init__(self, inChannels=[64, 128, 320, 512], feature_strides=[4, 8, 16, 32],
                dropout_ratio=0.1, act_layer=nn.ReLU, num_classes=20, embed_dim=768, align_corners=False):

        super().__init__()
    
        assert len(feature_strides) == len(inChannels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.inChannels = inChannels
        self.num_classes = num_classes
        embed_dim = embed_dim

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.inChannels

        # 1st step unify the channel dimensions
        self.linear_c4 = MLP(inputDim=c4_in_channels, embed_dim=embed_dim)
        self.linear_c3 = MLP(inputDim=c3_in_channels, embed_dim=embed_dim)
        self.linear_c2 = MLP(inputDim=c2_in_channels, embed_dim=embed_dim)
        self.linear_c1 = MLP(inputDim=c1_in_channels, embed_dim=embed_dim)

        self.linear_fuse = nn.Conv2d(embed_dim*4, embed_dim, kernel_size=3, padding=1) # 3 is in DAFormer confirmed88
        self.norm = nn.BatchNorm2d(embed_dim)
        self.act = act_layer()
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.linear_pred = nn.Conv2d(embed_dim, self.num_classes, kernel_size=1)

    
    def forward(self, inputs, height, width):

        c1, c2, c3, c4 = inputs
        N = c4.shape[0]

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(N, -1, c4.shape[2], c4.shape[3]) # 1st step unify the channel dimensions
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False) # 2nd step upsampling the dimensions
        
        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(N, -1, c3.shape[2], c3.shape[3]) # reshaped ot B*C*H*W
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(N, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(N, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1)) # 3rd setp adapting concatenated features
        _c = self.norm(_c)
        _c = self.act(_c)
        
        _c = F.upsample_bilinear(_c, size=(height, width))

        x = self.dropout(_c)
        x = self.linear_pred(x) # 4th step predict classes

        return x
    
class SegFormer(nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()
        
        self.encoder = MixVisionTransformer(3)
        self.decoder = SegFormerHead(num_classes=n_classes)
        
    def forward(self, x):
        _, _, h, w = x.size()
        feats = self.encoder(x)
        return self.decoder(feats, h, w)

# model = SegFormer(2).cuda()
# t = torch.rand(2, 3, 1601, 256).cuda()
# print(model(t).shape)