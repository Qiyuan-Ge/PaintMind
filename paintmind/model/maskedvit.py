import math
import torch
import torch.nn as nn
from einops import rearrange
from inspect import isfunction
from timm.models.vision_transformer import PatchEmbed


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class SelfAttention(nn.Module):
    def __init__(self, dim, d_head=64, num_heads=8, dropout=0.1):
        super().__init__()
        inner_dim = d_head * num_heads
        self.num_heads = num_heads
        self.scale = math.sqrt(d_head)
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(inner_dim, dim, bias=False)
        
    def forward(self, x, mask=None):
        x = self.norm(x)
        
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = [rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads) for x in (q, k, v)]
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) / self.scale
        
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            max_neg_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(~mask, max_neg_value)
            
        attn_p = sim.softmax(dim=-1)
        attn_p = self.dropout(attn_p)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn_p, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.proj(out)
    

class CrossAttention(nn.Module):
    def __init__(self, dim, d_head=64, num_heads=8, dropout=0.1, context_dim=None):
        super().__init__()
        inner_dim = d_head * num_heads
        context_dim = default(context_dim, dim)
        self.num_heads = num_heads
        self.scale = math.sqrt(d_head)
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
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) / self.scale
        
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            max_neg_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(~mask, max_neg_value)
            
        attn_p = sim.softmax(dim=-1)
        attn_p = self.dropout(attn_p)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn_p, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.proj(out)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim, d_ffn, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(dim, d_ffn)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.w_2 = nn.Linear(d_ffn, dim)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, d_head, d_ffn, context_dim=None, num_heads=8, dropout=0.1):
        super().__init__()
        # self.norm1 = nn.LayerNorm(dim)
        # self.attn1 = SelfAttention(dim, d_head, num_heads, dropout)
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(dim, d_head, num_heads, dropout, context_dim)
            
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = PositionwiseFeedForward(dim, d_ffn, dropout)
        
    def forward(self, x, text_emb=None, text_mask=None):
        # x = self.norm1(self.attn1(x) + x)
        
        x = self.norm1(self.attn1(x, text_emb, text_mask) + x)
            
        x = self.norm2(self.ffn(x) + x)

        return x


class Transformer(nn.Module):
    def __init__(self, dim, d_head, d_ffn, context_dim=None, num_heads=8, depth=6, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(depth):
            self.layers.add_module("block" + str(i), TransformerBlock(dim, d_head, d_ffn, context_dim, num_heads, dropout))

    def forward(self, x, text_emb=None, text_mask=None):
        for layer in self.layers:
            x = layer(x, text_emb, text_mask)

        return x

           
class MaskedViT(nn.Module):
    def __init__(self, image_size, patch_size, dim, d_ffn, context_dim=None, in_channels=3, d_head=64, num_heads=12, depth=12, dropout=0.1):
        super().__init__()
        
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        
        num_patches = (image_size // patch_size) ** 2
        
        self.in_channels = in_channels
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, in_channels))
        
        self.posi_embed = nn.Parameter(torch.randn(1, num_patches, dim))
        
        self.vision_transformer = Transformer(dim, d_head, d_ffn, context_dim, num_heads, depth, dropout)
        
        self.decoder = nn.Linear(dim, patch_size*patch_size*in_channels, bias=True)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
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
    
    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 *C)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_channels, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_channels))
        
        return x
    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *c)
        imgs: (N, C, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_channels, h * p, h * p))
        
        return imgs
    
    def forward(self, x, text_emb=None, text_mask=None, mask_ratio=0.75): #context (b, l) img (b c h w)
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1]-x.shape[1], 1)
        x = torch.cat([x[:, :, :], mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.patch_embed(x)
        x = x + self.posi_embed
        x = self.vision_transformer(x, text_emb, text_mask) #[N, L, p*p*3]
        x = self.decoder(x)
        xrec = self.unpatchify(x)

        return xrec
    
    # def forward(self, img, text_emb=None, text_mask=None, mask_ratio=0.75): #context (b, l) 
    #     x = self.patch_embed(img)
    #     x, mask, ids_restore = self.random_masking(x, mask_ratio)
    #     mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1]-x.shape[1], 1)
    #     x = torch.cat([x[:, :, :], mask_tokens], dim=1)
    #     x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    #     x = x + self.posi_embed
    #     x = self.vision_transformer(x, text_emb, text_mask.bool()) #[N, L, p*p*3]
    #     x = self.decoder(x)
    #     xrec = self.unpatchify(x)

    #     return xrec

def create_model(image_size, patch_size, dim, d_ffn, context_dim=None, in_channels=3, d_head=64, num_heads=12, depth=12, dropout=0.1):
    model = MaskedViT(image_size, patch_size, dim, d_ffn, context_dim, in_channels, d_head, num_heads, depth, dropout)
    
    return model
        
