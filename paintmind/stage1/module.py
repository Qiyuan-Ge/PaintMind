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
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Encoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_h, image_w = pair(image_size)
        patch_h, patch_w = pair(patch_size)

        assert image_h % patch_h == 0 and image_w % patch_w == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_h // patch_h) * (image_w // patch_w)
        patch_dim = channels * patch_h * patch_w

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_h, p2=patch_w),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)

        return x
 
       
class Decoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_h, image_w = pair(image_size)
        patch_h, patch_w = pair(patch_size)

        assert image_h % patch_h == 0 and image_w % patch_w == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_h // patch_h) * (image_w // patch_w)
        patch_dim = channels * patch_h * patch_w

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.proj = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=image_h//patch_h, w=image_w//patch_w, p1=patch_h, p2=patch_w),
        )
        
    def forward(self, x):
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.proj(x)

        return x
    
    

# encoder = Encoder(image_size=256, patch_size=8, dim=512, depth=8, heads=64, mlp_dim=2048, channels=3, dim_head=64, dropout=0., emb_dropout=0.)
# decoder = Decoder(image_size=256, patch_size=8, dim=512, depth=8, heads=64, mlp_dim=2048, channels=3, dim_head=64, dropout=0., emb_dropout=0.)

# img = torch.randn(1, 3, 256, 256)
# x = encoder(img)
# print(x.shape) (1, 1024, 512)

# x = decoder(x)
# print(x.shape) (1, 3, 256, 256)