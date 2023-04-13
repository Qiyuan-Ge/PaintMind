import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import paintmind as pm
from tqdm.auto import tqdm
from einops import rearrange
from inspect import isfunction
from paintmind.config import ver2cfg
from paintmind.stage2 import CondTransformer
from paintmind.encoder import T5TextEmbedder as TextEmbedder


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


class Pipeline(nn.Module):
    def __init__(self, config, stage1_pretrained=True):
        super().__init__()
        
        self.vqvae = pm.create_model(arch='vqgan', version=config.stage1, pretrained=stage1_pretrained)
        self.vqvae.freeze()
        
        self.text_model = TextEmbedder(freeze=True)
        
        vq_cfg = ver2cfg[config.stage1]
        self.num_tokens = (vq_cfg['encdec']['image_size'] // vq_cfg['encdec']['patch_size']) ** 2
        
        self.transformer = CondTransformer(
            vq_cfg['embed_dim'], config.dim, self.num_tokens, config.dim_head, config.mlp_dim, 
            config.num_head, config.depth, config.dropout, config.dim_context, vq_cfg['n_embed'],
        )
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, vq_cfg['embed_dim']))
        nn.init.normal_(self.mask_token, std=.02)
        
    def from_pretrained(self, path):
        return self.load_state_dict(torch.load(path))
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        
        N, L, D = x.shape  # batch, length, dim
        len_mask = max(int(L * mask_ratio), 1)
        len_keep = L - len_mask
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1]-x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        # unshuffle
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x, mask
    
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
    
    @torch.no_grad()
    def to_latent(self, img, text=None):
        x, _, indices = self.vqvae.encode(img)
        if exists(text):
            text = self.text_model(text)
        
        return x, indices, text
    
    def forward(self, img, text=None, mask_ratio=0.75):
        # stage1
        x, indices, text = self.to_latent(img, text)   
        # random mask
        x, mask = self.random_masking(x, mask_ratio)
        # stage2
        logits = self.transformer(x, text)
        # loss
        loss = self.loss(logits, indices, mask)

        return loss
    
    @torch.no_grad()
    def generate(self, text=None, timesteps=18, temperature=1.0, save_interval=2):
        imgs = []
        B = len(text)
        len_seq = self.num_tokens
        cur_seq = self.mask_token.repeat(B, len_seq, 1)
        text = self.text_model(text) if exists(text) else None
        for step in tqdm(range(timesteps)):
            cur_temp = temperature*(1-step/timesteps)
            
            logits = self.transformer(cur_seq, text)
            pred_ids = gumbel_sample(logits, temperature=cur_temp, dim=-1)
            pred_seq = self.vqvae.quantize.decode_from_indice(pred_ids)
            if step % save_interval == 0:
                img = self.vqvae.decode(pred_seq)
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
            cur_seq = torch.cat([cur_seq, self.mask_token.repeat(B, len_seq-keep_len, 1)], dim=1)
            cur_seq = torch.gather(cur_seq, dim=1, index=code_ids_restore.unsqueeze(-1).repeat(1, 1, cur_seq.shape[2]))  # restore seq
            
        return imgs
