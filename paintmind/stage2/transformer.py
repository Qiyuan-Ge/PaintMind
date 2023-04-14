import math
import torch
import torch.nn as nn
from einops import rearrange
from inspect import isfunction

from typing import Optional, Any

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.num_head = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.attn_drop = nn.Dropout(p=dropout)
        self.to_o = nn.Linear(inner_dim, query_dim)
        
    def forward(self, x, context=None, mask=None):
        context = default(context, x)
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = [rearrange(x, 'b n (h d) -> b h n d', h=self.num_head) for x in (q, k, v)]
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k) / math.sqrt(q.shape[-1])
        
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            max_neg_value = -torch.finfo(attn.dtype).max
            attn = attn.masked_fill(~mask, max_neg_value)
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.to_o(x)

        return x


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)    
  
    
class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.1):
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
    

class Layer(nn.Module):
    ATTENTION_MODES = {
        "vanilla": CrossAttention,
        "xformer": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, dim_head, mlp_dim, num_head=8, dropout=0.0, dim_context=None):
        super().__init__()
        attn_mode = "xformer" if XFORMERS_IS_AVAILBLE else "vanilla"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = attn_cls(query_dim=dim, heads=num_head, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = attn_cls(query_dim=dim, context_dim=dim_context, heads=num_head, dim_head=dim_head, dropout=dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.ffnet = FeedForward(dim, mlp_dim, dropout)
        
    def forward(self, x, context=None): 
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context) + x
        x = self.ffnet(self.norm3(x)) + x

        return x
    

class CondTransformer(nn.Module):
    def __init__(self, in_dim, dim, len_seq, dim_head, mlp_dim, num_head=8, depth=6, dropout=0.1, dim_context=None, num_calsses=1000):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)
        self.position_embedding  = nn.Parameter(torch.randn(1, len_seq, dim))
        self.dropout = nn.Dropout(dropout)
        self.context_proj = nn.Linear(dim_context, dim, bias=False) if dim_context != dim else nn.Identity()
        self.layers = nn.Sequential()
        for i in range(depth):
            self.layers.add_module("layer" + str(i), Layer(dim, dim_head, mlp_dim, num_head, dropout, dim))
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_calsses)
        
        self.initialize_weights()

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, context=None):
        x = self.proj(x)
        x = x + self.position_embedding 
        x = self.dropout(x)
        
        if exists(context):
            context = self.context_proj(context)
            
        for layer in self.layers:
            x = layer(x, context)
        
        x = self.norm(x)
        x = self.to_logits(x)
        
        return x
