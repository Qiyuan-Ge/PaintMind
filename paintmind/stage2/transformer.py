import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import paintmind as pm
from einops import rearrange
from inspect import isfunction
from paintmind.stage2.clip import OpenCLIPEmbedder
from tqdm.auto import tqdm


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mask_schedule(ratio):
    return np.cos(math.pi / 2. * ratio)


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)


class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, num_head=8, dropout=0.1, dim_context=None):
        super().__init__()
        inner_dim = dim_head * num_head
        dim_context = default(dim_context, dim)
        self.num_head = num_head
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim_context, inner_dim, bias=False)
        self.to_v = nn.Linear(dim_context, inner_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.to_o = nn.Linear(inner_dim, dim, bias=False)
        
    def forward(self, x, context=None, mask=None):
        context = default(context, x)
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = [rearrange(x, 'b n (h d) -> b h n d', h=self.num_head) for x in (q, k, v)]
        scores = torch.einsum('b h i d, b h j d -> b h i j', q, k) / math.sqrt(q.shape[-1])
        
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            max_neg_value = -torch.finfo(scores.dtype).max
            scores = scores.masked_fill(~mask, max_neg_value)
            
        attn_p = scores.softmax(dim=-1)
        attn_p = self.dropout(attn_p)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn_p, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_o(out)
    
    
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
    def __init__(self, dim, dim_head, mlp_dim, num_head=8, dropout=0.1, dim_context=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(dim, dim_head, num_head, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = Attention(dim, dim_head, num_head, dropout, dim_context)
        self.norm3 = nn.LayerNorm(dim)
        self.ffnet = FeedForward(dim, mlp_dim, dropout)
        
    def forward(self, x, context=None): 
        x = self.norm1(self.attn1(x) + x)
        x = self.norm2(self.attn2(x, context) + x)
        x = self.norm3(self.ffnet(x) + x)

        return x
    

class TransformerLayers(nn.Module):
    def __init__(self, dim, dim_head, mlp_dim, num_head=8, depth=6, dropout=0.1, dim_context=None):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(depth):
            self.layers.add_module("layer" + str(i), Layer(dim, dim_head, mlp_dim, num_head, dropout, dim_context))

    def forward(self, x, context=None):
        for layer in self.layers:
            x = layer(x, context)

        return x
    

class PaintMind(nn.Module):
    def __init__(self, config, vae_pretrained=None, clip_precision='fp32'):
        super().__init__()
        
        self.vqvae = pm.build_model(name=config.vae_name, pretrained=vae_pretrained)
        self.vqvae.eval()
        self.vqvae.freeze()
        
        self.clip = OpenCLIPEmbedder(precision=clip_precision)
        self.clip.eval()
        self.clip.freeze()
        
        self.num_token = (self.vqvae.image_size // self.vqvae.patch_size) ** 2
        
        self.proj = nn.Linear(self.vqvae.embed_dim, config.dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_token, config.dim))
        self.text_proj = nn.Linear(config.dim_context, config.dim, bias=False) if config.dim_context != config.dim else nn.Identity() 
        self.attention = TransformerLayers(config.dim, config.dim_head, config.mlp_dim, config.num_head, config.depth, config.dropout, config.dim)
        self.to_logits = nn.Linear(config.dim, self.vqvae.n_embed)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.vqvae.embed_dim))
        self.initialize_weights()
        
        
    def initialize_weights(self):
        nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def from_pretrained(self, path):
        return self.load_state_dict(torch.load(path))
    
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
    
    def transformer_forward(self, x, text_emb=None):
        x = self.proj(x)
        x = x + self.pos_embed
        
        if exists(text_emb):
            text_emb = self.text_proj(text_emb)
            
        x = self.attention(x, text_emb)
        x = self.to_logits(x)
        
        return x
    
    def loss(self, logit, label, mask):
        """
        logit: [B, L, N]
        label: [B, L]
        mask : [B, L], 1 is mask
        """
        
        logit = rearrange(logit, 'b l d -> (b l) d')
        label = rearrange(label, 'b l -> (b l)')
        mask = rearrange(mask, 'b l -> (b l)')
        loss = F.cross_entropy(logit, label, label_smoothing=0.1, reduction='none')
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def forward(self, img, text=None, mask_ratio=0.75):
        
        with torch.no_grad():
            x, _, indices = self.vqvae.encode(img)
            text_emb = self.clip(text) if exists(text) else None
            
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1]-x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        
        logits = self.transformer_forward(x, text_emb)
        
        loss = self.loss(logits, indices, mask)

        return loss
    
    @torch.no_grad()
    def generate(self, text=None, timesteps=18, temperature=1.0, save_interval=2):
        imgs = []
        B = len(text)
        len_seq = self.num_token
        cur_seq = self.mask_token.repeat(B, len_seq, 1)
        text_emb = self.clip(text) if exists(text) else None
        for step in tqdm(range(timesteps)):
            cur_temp = temperature*(1-step/timesteps)
            
            logits = self.transformer_forward(cur_seq, text_emb)
            pred_ids = gumbel_sample(logits, temperature=cur_temp, dim=-1)
            pred_seq = self.vqvae.quantize.decode_from_indice(pred_ids)
            
            img = self.vqvae.decode(pred_seq)
            if step % save_interval == 0:
                imgs.append(img.cpu())           
            # Fill the mask, ignore the unmasked.
            is_mask = (cur_seq == self.mask_token)
            cur_seq = torch.where(is_mask, pred_seq, cur_seq)            
            # Masks tokens with lower confidence.
            probs = F.softmax(logits, dim=-1) + 0.5 * torch.zeros_like(logits).uniform_(0, 1) * cur_temp
            select_probs, indice = probs.max(dim=-1) #(b l)
            select_probs = select_probs.masked_fill(~is_mask.all(dim=-1), 2)
            
            ratio = 1. * (step + 1) / timesteps
            mask_ratio = mask_schedule(ratio)
            keep_len = int(len_seq * (1 - mask_ratio))
            
            prob_ids_descend = torch.argsort(select_probs, descending=True)
            code_ids_restore = torch.argsort(prob_ids_descend, dim=1)
            code_ids_descend = torch.gather(indice, dim=1, index=prob_ids_descend)
            
            code_ids_keep = code_ids_descend[:, :keep_len]
            cur_seq = self.vqvae.quantize.decode_from_indice(code_ids_keep)
            cur_seq = torch.cat([cur_seq, self.mask_token.repeat(B, len_seq, 1)], dim=1)
            cur_seq = torch.gather(cur_seq, dim=1, index=code_ids_restore.unsqueeze(-1).repeat(1, 1, cur_seq.shape[2]))  # restore seq
            
        return imgs
            
            
            
