import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.):
        super().__init__()
        self.w_1 = nn.Linear(dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.w_2 = nn.Linear(mlp_dim, dim)
    
    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.to_out(x)
        
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0., init_values=None):
        super().__init__()
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.layer1 = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.layer2 = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        
    def forward(self, x):
        x = x + self.ls1(self.layer1(x))
        x = x + self.ls1(self.layer2(x))
        
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., init_values=None):
        super().__init__()
        self.layers = nn.Sequential(*[Block(dim, heads, dim_head, mlp_dim, dropout, init_values) for i in range(depth)])
    
    def forward(self, x):
        x = self.layers(x)
        
        return x


class Encoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, in_channels=3, out_channels=3, dim_head=64, dropout=0.):
        super().__init__()
        
        image_size = pair(image_size)
        patch_size = pair(patch_size)

        assert image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size, bias=False),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, dim) * .02)
        self.pos_drop = nn.Dropout(dropout)
        self.norm_pre = nn.LayerNorm(dim)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        x += self.position_embedding
        x = self.pos_drop(x)
        x = self.norm_pre(x)
        x = self.transformer(x)
        
        return x
 
       
class Decoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, in_channels=3, out_channels=3, dim_head=64, dropout=0.):
        super().__init__()
        
        image_size = pair(image_size)
        patch_size = pair(patch_size)

        assert image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, dim) * .02)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(
            nn.Linear(dim, out_channels * patch_size[0] * patch_size[1], bias=True),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=image_size[0]//patch_size[0], p1=patch_size[0], p2=patch_size[1]),
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        x += self.position_embedding
        x = self.transformer(x)
        x = self.norm(x)
        x = self.proj(x)
        
        return x
        
# x = torch.randn(1, 3, 256, 256)
# encoder = Encoder(256, 8, 512, 8, 8, 512, 3)
# decoder = Decoder(256, 8, 512, 8, 8, 512, 3)
# z = encoder(x)
# print(z.shape)
# y = decoder(z)
# print(y.shape)
# print(decoder)
    


