import torch
from torch import nn
from ..modules.attention import CrossAttention, MemoryEfficientCrossAttention, XFORMERS_IS_AVAILBLE
from ..modules.mlp import SwiGLUFFNFused


def exists(x):
    return x is not None
  
    
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
        self.ffnet = SwiGLUFFNFused(in_features=dim, hidden_features=mlp_dim)
        
    def forward(self, x, context=None): 
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context) + x
        x = self.ffnet(self.norm3(x)) + x

        return x
    

class CondTransformer(nn.Module):
    def __init__(self, in_dim, dim, len_seq, dim_head, mlp_dim, num_head=8, depth=6, dropout=0.1, context_dim=None, num_classes=8192):
        super().__init__()
        scale = dim ** -0.5
        self.token_proj = nn.Linear(in_dim, dim)
        self.position_embedding = nn.Parameter(torch.randn(1, len_seq, dim) * scale)
        self.context_proj = nn.Linear(context_dim, dim, bias=False) if context_dim != dim else nn.Identity()
        self.layers = nn.Sequential()
        for i in range(depth):
            self.layers.add_module("layer" + str(i), Layer(dim, dim_head, mlp_dim, num_head, dropout, dim))
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_classes)
        
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
        x = self.token_proj(x)
        x = x + self.position_embedding 
        
        if exists(context):
            context = self.context_proj(context)
            
        for layer in self.layers:
            x = layer(x, context)
        
        x = self.norm(x)
        x = self.to_logits(x)
        
        return x
