import math
import torch
import torch.nn as nn
from einops import rearrange
from inspect import isfunction
from timm.models.vision_transformer import PatchEmbed
from paintmind.util.pos_embed import get_2d_sincos_pos_embed


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def pair(t):
    return t if isinstance(t, tuple) else (t, t)
    

class CrossAttention(nn.Module):
    def __init__(self, dim, d_head=64, num_heads=8, dropout=0.1, context_dim=None):
        super().__init__()
        inner_dim = d_head * num_heads
        context_dim = default(context_dim, dim)
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(inner_dim, dim, bias=False)
        
    def forward(self, x, context=None, mask=None):
        context = default(context, x)
        
        x = self.norm(x)
        context = self.norm_context(context)
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = [rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads) for x in (q, k, v)]
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) / math.sqrt(q.shape[-1])
        
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            max_neg_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(~mask, max_neg_value)
            
        attn_p = sim.softmax(dim=-1)
        attn_p = self.dropout(attn_p)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn_p, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.w_2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, d_head, num_heads=8, dropout=0.1, context_dim=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(dim, d_head, num_heads, dropout, context_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout)
        
    def forward(self, x, text=None):  
        x = self.norm1(self.attn1(x, text) + x)   
        x = self.norm2(self.ffn(x) + x)

        return x


class Encoder(nn.Module):
    def __init__(self, dim, d_head, num_heads=8, depth=6, dropout=0.1, context_dim=None):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(depth):
            self.layers.add_module("block" + str(i), Block(dim, d_head, num_heads, dropout, context_dim))

    def forward(self, x, text=None):
        for layer in self.layers:
            x = layer(x, text)

        return x

           
class MaskedViT(nn.Module):
    def __init__(self, img_size, patch_size, dim, context_dim=None, in_channels=3, d_head=64, num_heads=12, depth=12, dropout=0.1, num_classes=256):
        super().__init__()
        self.patch_size = patch_size
        self.mask_token = nn.Parameter(torch.zeros(1, 1, in_channels))
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, dim)
        num_patches = self.patch_embed.num_patches
        
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim))
        self.encoder = Encoder(dim, d_head, num_heads, depth, dropout, context_dim)
        self.encoder_norm = nn.LayerNorm(dim)
        self.decoder = nn.Linear(dim, patch_size**2 * num_classes, bias=True)
        
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def inference(self, x, text=None):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.encoder(x, text)
        x = self.encoder_norm(x)
        x = self.decoder(x)
        x = rearrange(x, 'b l (n d) -> b (l n) d', n=self.patch_size**2)
        
        return x
    
    def forward(self, x, text=None, mask_ratio=0.75):
        b, c, h, w = x.shape
        
        x = rearrange(x, 'b c h w -> b (h w) c')
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1]-x.shape[1], 1)
        x = torch.cat([x[:, :, :], mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.inference(x, text)

        return x, mask

def create_model(img_size=32, patch_size=2, dim=1024, context_dim=1024, in_channels=4, d_head=64, num_heads=16, depth=10, dropout=0.1, num_classes=256):
    model = MaskedViT(img_size, patch_size, dim, context_dim, in_channels, d_head, num_heads, depth, dropout, num_classes)
    
    return model


# model = MaskedViT(img_size=32, patch_size=2, dim=1024, context_dim=1024, in_channels=3, d_head=64, num_heads=16, depth=10, dropout=0.1, num_classes=256)
# x = torch.randn(2, 3, 64, 64)
# y, mask = model(x)
# print(y.shape)