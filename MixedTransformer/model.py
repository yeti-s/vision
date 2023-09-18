import torch


import torch.nn as nn
import torch.nn.functional as F

# x : (b, r, c, dim), out : (b, h, w, dim)
def pad_tensor(x, h, w):
    d_y = h - x.size(1)
    d_x = w - x.size(2)
    return F.pad(x, [0, 0, d_x//2, d_x-d_x//2, d_y//2, d_y-d_y//2])


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=1, activation=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(cin, cout, 3, 1, padding=1),
            ConvBNReLU(cout, cout, 3, stride=1, padding=1, activation=False))
        self.conv1 = nn.Conv2d(cout, cout, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.conv(x)
        h = x
        x = self.conv1(x)
        x = self.bn(x)
        x = h + x
        x = self.relu(x)
        return x


class EncoderStem(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(EncoderStem, self).__init__()
        
        self.conv1 = DoubleConv(in_channels, out_channels//4)
        self.conv2 = DoubleConv(out_channels//4, out_channels//2)
        self.conv3 = DoubleConv(out_channels//2, out_channels)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.conv1(x)
        outputs = [x]
        
        x = self.pool(x)
        x = self.conv2(x)
        outputs.append(x)
        
        x = self.pool(x)
        x = self.conv3(x)
        outputs.append(x)
        
        return self.pool(x), outputs

class DecoderStem(nn.Module):
    def __init__(self, in_channels) -> None:
        super(DecoderStem, self).__init__()
        
        self.deconvs = nn.ModuleList([
            nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2),
            nn.ConvTranspose2d(in_channels, in_channels//2, 2, stride=2),
            nn.ConvTranspose2d(in_channels//2, in_channels//4, 2, stride=2)
        ])
        self.convs = nn.ModuleList([
            DoubleConv(in_channels*2, in_channels),
            DoubleConv(in_channels, in_channels//2),
            DoubleConv(in_channels//2, in_channels//4)
        ])
    
    # x, skips : (b, dim, h, w)
    def forward(self, x, skips):
        for i in range(3):
            skip = skips.pop()
            x = self.deconvs[i](x)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = pad_tensor(x, skip.size(2), skip.size(3)).permute(0, 3, 1, 2).contiguous()
            x = torch.cat([x, skip], dim=1)
            x = self.convs[i](x)
        
        return x

# official code 에서 구현된 코드는 https://arxiv.org/pdf/1901.10430.pdf 논문과는 조금 다른듯 한데
# 일단 나중에 더 찾아보도록 하고 여기서는 기존 코드를 가져와 쓰는 형식으로 하자
# https://github.com/Dootmaan/MT-UNet/blob/main/model/MTUNet.py#L250
class DlightConv(nn.Module):
    def __init__(self, dim, patch_size):
        super(DlightConv, self).__init__()
        self.linear = nn.Linear(dim, patch_size**2)
        self.softmax = nn.Softmax(dim=-1)

    # x : (b, n_rows, q_cols, p**2, dim), out : (b, n_rows, q_cols, dim)
    def forward(self, x):
        h = x
        avg_x = torch.mean(x, dim=-2)  # (b, n_rows, q_cols, 1, dim)
        x_prob = self.softmax(self.linear(avg_x))  # (b, n_rows, q_cols, p**2)

        x = torch.mul(h, x_prob.unsqueeze(-1))  # (b, p, p, p**2, dim)
        x = torch.sum(x, dim=-2)  # (b, n_rows, q_cols, 1, dim)
        
        return x.squeeze(-2)


class SelfAttention(nn.Module):
    def __init__(self, dim) -> None:
        super(SelfAttention, self).__init__()
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
    # x : (b, n_rows, n_cols, patch_size**2, dim)
    def forward(self, x):
        query = self.query(x)
        key = self.query(x)
        value = self.query(x)
        
        z = torch.matmul(query, key.transpose(-2, -1)) # (b, n_rows, n_cols, patch_size**2, patch_size**2)
        z = F.softmax(z, -1).matmul(value) # (b, n_rows, n_cols, patch_size**2, dim)
        
        return z


class LSALayer(nn.Module):
    def __init__(self, dim, patch_size) -> None:
        super(LSALayer, self).__init__()
        
        self.p = patch_size
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
    
    # x : (b, h, w, dim), out : (b, q_h, q_w, p**2, dim)
    def forward(self, x):
        _, h, w, _ = x.size()
        p_square = self.p ** 2
        q_h, r_h = h // p_square, (h % p_square) // 2
        q_w, r_w = w // p_square, (w % p_square) // 2
        
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x[:,:,r_h:q_h*p_square+r_h,r_w:q_w*p_square+r_w] # (b, dim, h - remainder, w - remainder)
        x = x.unfold(2, self.p, self.p).unfold(3, self.p, self.p) # (b, dim, q_h, q_w, p, p)
        x = x.flatten(-2, -1).permute(0, 2, 3, 4, 1).contiguous() # (b, q_h, q_w, p**2, dim)
        
        query = self.query(x)
        key = self.query(x)
        value = self.query(x)
        
        z = torch.matmul(query, key.transpose(-2, -1)) # (b, q_h, q_w, p**2, p**2)
        z = F.softmax(z, -1).matmul(value) # (b, q_h, q_w, p**2, dim)
        
        return self.out(z)

class GSALayer(nn.Module):
    def __init__(self, dim) -> None:
        super(GSALayer, self).__init__()
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        
        self.sigma = nn.Parameter(torch.ones(1))
    
    def distance_tensor(self, size, device):
        i = torch.arange(size).view(size, 1).to(device)
        j = torch.arange(size).view(1, size).to(device)

        return torch.square(i - j).float()
    
    # x : (b, h, w, dim)
    def forward(self, x):
        device = x.device
        b, h, w, dim = x.size()
        gaussian_weight = 1 / (2*torch.square(self.sigma))
        
        query = self.query(x) 
        key = self.key(x)
        value = self.value(x)
        
        # row-axis
        r_query = query.view(b*h, w, -1)
        r_key = key.view(b*h, w, -1)
        r_qk = torch.matmul(r_query, r_key.transpose(-2, -1)).view(b, h, w, w) 
        r_dist = self.distance_tensor(w, device) # (w, w)
        r_weight = r_dist.mul(gaussian_weight)
        r_qk = r_qk - r_weight
        
        # col-axis
        c_query = query.permute(0, 2, 1, 3).contiguous().view(b*w, h, -1)
        c_key = key.permute(0, 2, 1, 3).contiguous().view(b*w, h, -1)
        c_qk = torch.matmul(c_query, c_key.transpose(-2, -1)).view(b, w, h, h)
        c_dist = self.distance_tensor(h, device) # (h, h)
        c_weight = c_dist.mul(gaussian_weight)
        c_qk = c_qk - c_weight
        
        output = torch.zeros_like(value) # (b, h, w, dim)
        for r in range(h):
            for c in range(w):
                # row 와 col에서 계산에 관여하는 부분을 합치고 난 후 softmax 해야할 것 같은데,,
                r_attn = F.softmax(r_qk[:,r,c,:], dim=-1) # b, w
                c_attn = F.softmax(c_qk[:,c,r,:], dim=-1) # b, h
                
                r_value = torch.matmul(r_attn.unsqueeze(dim=-2), value[:,r,:,:]) # (b, 1, dim)
                c_value = torch.matmul(c_attn.unsqueeze(dim=-2), value[:,:,c,:]) # (b, 1, dim)
                output[:, r, c, :] = (r_value + c_value).squeeze(1)
                
        return self.out(output) # (b, h, w, dim)
                
    
class LGGSALayer(nn.Module):
    def __init__(self, dim, patch_size) -> None:
        super(LGGSALayer, self).__init__()
        
        self.patch_size = patch_size
        # self.layer_norm = nn.LayerNorm()
        self.lsa_layer = LSALayer(dim, patch_size)
        self.dlconv = DlightConv(dim, patch_size)
        self.gsa_layer = GSALayer(dim)
        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=patch_size)
        self.squeeze = nn.Conv2d(dim*2, dim, 1)
        
        
    # x : (b, h, w, dim), out (b, h, w, dim)
    def forward(self, x):
        lsa_out = self.lsa_layer(x) # (b, n_rows, n_cols, p**2, dim)
        aggr_out = self.dlconv(lsa_out) # (b, n_rows, n_cols, dim)
        gsa_out = self.gsa_layer(aggr_out)
        
        b, r, c, _, _ = lsa_out.size()
        lsa_out = lsa_out.view(b, r, c, self.patch_size, self.patch_size, -1).contiguous()
        lsa_out = lsa_out.permute(0, 5, 1, 3, 2, 4).contiguous()
        lsa_out = lsa_out.view(b, -1, r * self.patch_size, c * self.patch_size).contiguous() # (b, dim, r*p, c*p)
        
        gsa_out = gsa_out.permute(0, 3, 1, 2).contiguous()
        gsa_out = self.up_sample(gsa_out) # (b, dim, r*p, c*p)
        
        x = torch.cat([lsa_out, gsa_out], dim=1)
        x = self.squeeze(x).permute(0, 2, 3, 1).contiguous()
        
        return x


# 이 부분도 설명히 자세히 없어서 코드 일단 가져왓음 돌리면서 이해해보고 수정하자
class EALayer(nn.Module):
    def __init__(self, dim) -> None:
        super(EALayer, self).__init__()
        self.num_heads = 8
        self.coef = 4
        self.query_liner = nn.Linear(dim, dim * self.coef)
        self.num_heads = self.coef * self.num_heads
        self.k = 256 // self.coef
        self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
        self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)

        self.proj = nn.Linear(dim * self.coef, dim)

    # x : (b, h, w, dim)
    def forward(self, x):
        b, w, h, dim = x.size()
        n = w * h
        
        x = x.view(b, -1, dim)

        x = self.query_liner(x) # (b, n, dim * coef)
        x = x.view(b, n, self.num_heads, -1).permute(0, 2, 1, 3)  # (b, heads, n, dim * coef / head)

        attn = self.linear_0(x) # (b, heads, n, k)

        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))

        x = self.linear_1(attn).permute(0, 2, 1, 3).reshape(b, n, -1)

        x = self.proj(x)

        return x.view(b, w, h, dim)
        


class MTMLayer(nn.Module):
    def __init__(self, dim, patch_size) -> None:
        super(MTMLayer, self).__init__()
        
        self.lggsa_ln = nn.LayerNorm(dim)
        self.lggsa_layer = LGGSALayer(dim, patch_size)
        
        self.extern_ln = nn.LayerNorm(dim)
        self.ea_layer = EALayer(dim)
        
    # x : (b, h, w, dim)
    def forward(self, x):
        _, h, w, _ = x.size()
        
        out = self.lggsa_ln(x)
        out = self.lggsa_layer(out) # (b, h, w, dim)
        out = pad_tensor(out, h, w)
        x = x + out
        
        out = self.extern_ln(x)
        out = self.ea_layer(out)
        x = x + out
        
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, dim, patch_size, down_sample=True) -> None:
        super(EncoderBlock, self).__init__()
        
        self.down_sample = down_sample
        self.blocks = nn.Sequential(
            MTMLayer(dim, patch_size),
            MTMLayer(dim, patch_size)
        )
        if down_sample:
            self.down = nn.Sequential(
                DoubleConv(dim, dim*2),
                nn.MaxPool2d(2)
            )
    
    # x : (b, dim, h, w), return x, output : (b, dim, h, w)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.blocks(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        output = x
        
        if self.down_sample:
            x = self.down(x)
            
        return x, output

class DecoderBlock(nn.Module):
    def __init__(self, dim, patch_size) -> None:
        super(DecoderBlock, self).__init__()
        
        self.deconv = nn.ConvTranspose2d(dim, dim//2, 2, stride=2)
        self.conv = nn.Conv2d(dim, dim//2, 1)
        self.blocks = nn.Sequential(
            MTMLayer(dim//2, patch_size),
            MTMLayer(dim//2, patch_size)
        )
    
    # x, skip : (b, dim, h, w)
    def forward(self, x, skip):
        _, _, h, w = skip.size()
        x = self.deconv(x)
        x = pad_tensor(x.permute(0, 2, 3, 1).contiguous(), h, w)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.blocks(x)
        
        return x.permute(0, 3, 1, 2).contiguous()
        

class MixedTransformer(nn.Module):
    def __init__(self, in_channels, n_classes, patch_size=2) -> None:
        super(MixedTransformer, self).__init__()
        
        self.encoder_stem = EncoderStem(in_channels, 256)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(256, patch_size),
            EncoderBlock(512, patch_size),
            EncoderBlock(1024, patch_size, down_sample=False)
        ])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(1024, patch_size),
            DecoderBlock(512, patch_size)
        ])
        self.decoder_stem = DecoderStem(256)
        
        self.conv = nn.Conv2d(64, n_classes, 1)
        
    # x : (b, dim, h, w)
    def forward(self, x):
        
        x, stem_outs = self.encoder_stem(x)
        
        # encoder
        encoder_outs = []
        for block in self.encoder_blocks:
            x, encoder_out = block(x)
            encoder_outs.append(encoder_out)
        encoder_outs.pop()
        
        # decoder
        for block in self.decoder_blocks:
            x = block(x, encoder_outs.pop())
        x = self.decoder_stem(x, stem_outs)
        
        return self.conv(x)

mit = MixedTransformer(3, 2).cuda()
t = torch.rand(2, 3, 319, 256).cuda()
print(mit(t).shape)
# with torch.no_grad():
#     print(mit(t).shape)
