import math
import torch
import torch.nn as nn
from einops import rearrange
from inspect import isfunction


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
    def __init__(self, dim, d_head, context_dim=None, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(dim, d_head, num_heads, dropout, context_dim) 
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout)
        
    def forward(self, x, text=None, mask=None):  
        x = self.norm1(self.attn1(x, text, mask) + x)   
        x = self.norm2(self.ffn(x) + x)

        return x


class AttentionLayers(nn.Module):
    def __init__(self, dim, d_head, context_dim=None, num_heads=8, depth=6, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(depth):
            self.layers.add_module("block" + str(i), Block(dim, d_head, context_dim, num_heads, dropout))

    def forward(self, x, text=None, mask=None):
        for layer in self.layers:
            x = layer(x, text, mask)

        return x

           
class Transformer(nn.Module):
    def __init__(self, dim, context_dim=None, in_channels=3, d_head=64, num_heads=12, depth=12, dropout=0.1):
        super().__init__() 
        self.proj = nn.Linear(in_channels, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, dim))
        self.encoder = AttentionLayers(dim, d_head, context_dim, num_heads, depth, dropout)
        self.decoder = nn.Linear(dim, 256)
    
    def forward(self, x, text=None, mask=None): # x=(b c h w)
        x = self.proj(x)
        x = x + self.pos_emb
        x = self.encoder(x, text, mask) #[N, L, D]
        x = self.decoder(x)

        return x

def create_model(dim, context_dim=None, in_channels=3, d_head=64, num_heads=12, depth=12, dropout=0.1):
    model = Transformer(dim, context_dim, in_channels, d_head, num_heads, depth, dropout)
    
    return model
        
