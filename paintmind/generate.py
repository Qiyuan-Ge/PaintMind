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
from paintmind.encoder import CLIPTextEmbedder as TextEmbedder


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


def top_k(logits, k=1):
    val, ind = logits.topk(k, dim = -1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


class Pipeline(nn.Module):
    def __init__(self, config, stage1_pretrained=True):
        super().__init__()
        clip_version = {'ViT-L-14':'laion2b_s32b_b82k', 'ViT-H-14':'laion2b_s32b_b79k'}
        context_dim = {'ViT-L-14':768, 'ViT-H-14':1024}
        
        self.vqgan = pm.create_model(arch='vqgan', version=config.stage1, pretrained=stage1_pretrained)
        self.vqgan.freeze()
        
        self.text_model = TextEmbedder(arch=config.clip, version=clip_version[config.clip], freeze=True)
        
        vq_cfg = ver2cfg[config.stage1]
        self.num_tokens = (vq_cfg['enc']['image_size'] // vq_cfg['enc']['patch_size']) ** 2
        
        self.transformer = CondTransformer(
            vq_cfg['embed_dim'], config.dim, self.num_tokens, config.dim_head, config.mlp_dim, 
            config.num_head, config.depth, config.dropout, context_dim[config.clip], vq_cfg['n_embed'],
        )
        
        self.mask_token = nn.Parameter(torch.zeros(1, vq_cfg['embed_dim']))
        self.mask_token_id = self.num_tokens
        
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
        mask_tokens = self.mask_token.unsqueeze(0).repeat(x.shape[0], ids_restore.shape[1]-x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        # unshuffle
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x, mask
    
    def loss(self, logit, label, masks):
        """
        logit: [B, L, N]
        label: [B, L]
        mask : [B, L], 1 is mask
        """
        
        logit = rearrange(logit, 'b l d -> (b l) d')
        label = rearrange(label, 'b l -> (b l)')
        masks = rearrange(masks, 'b l -> (b l)')
        loss = F.cross_entropy(logit, label, label_smoothing=0.1, reduction='none')
        loss = (loss * masks).sum() / masks.sum()
        
        return loss
    
    @torch.no_grad()
    def to_latent(self, img, text=None):
        x, _, indices = self.vqgan.encode(img)
        if exists(text):
            text = self.text_model(text)
        
        return x, indices, text
    
    def token_to_logits(self, token, text=None):
        return self.transformer(token, text)
    
    def forward(self, img, text=None, mask_ratio=0.75):
        # stage1
        x, indices, text = self.to_latent(img, text)   
        # random mask
        x, mask = self.random_masking(x, mask_ratio)
        # stage2
        logits = self.token_to_logits(x, text)
        # loss
        loss = self.loss(logits, indices, mask)

        return loss
        
    @torch.no_grad()
    def generate(self, text, timesteps=18, temperature=1.0, topk=5, save_interval=2):
        B = len(text)
        len_seq = self.num_tokens
        imgs = []
        
        text = self.text_model(text)
        
        cat = torch.cat((self.vqgan.quantize.embedding.weight, self.mask_token))
        emb = nn.Embedding(len_seq, self.mask_token.shape[-1]).from_pretrained(cat)
        
        ids = torch.full((B, len_seq), self.mask_token_id, dtype=torch.long, device=self.mask_token.device)
        for step in tqdm(range(timesteps)):
            logits = self.token_to_logits(emb(ids), text)
            filtered_logits = top_k(logits, topk)
            cur_temp = temperature*(1-step/timesteps)
            pred_ids = gumbel_sample(filtered_logits, temperature=cur_temp, dim=-1)
            is_mask = ids == self.mask_token_id
            # Fill the mask, ignore the unmasked.
            ids = torch.where(is_mask, pred_ids, ids)
            
            if step % save_interval == 0:
                img = self.vqgan.decode_from_indice(ids)
                imgs.append(img.cpu())
                
            probs = logits.softmax(dim=-1)
            scores = 1 - probs.gather(2, pred_ids[..., None])
            scores = rearrange(scores, '... 1 -> ...')
            scores = scores.masked_fill(~is_mask, -1e5)
            
            progress = (step + 1) / timesteps
            mask_ratio = mask_schedule(progress)
            num_token_masked = max(int((mask_ratio * len_seq).item()), 1)
            
            masked_indices = scores.topk(num_token_masked, dim=-1).indices
            
            ids = ids.scatter(1, masked_indices, self.mask_token_id)
        
        return imgs
            
