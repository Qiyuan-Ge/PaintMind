import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import Encoder, Decoder
from .quantize import VectorQuantizer



class VQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.n_embed = config.n_embed
        self.channels = config.channels
        self.embed_dim = config.embed_dim
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        
        self.encoder = Encoder(config.image_size, config.patch_size, config.dim, config.depth, config.heads, config.mlp_dim, config.channels, config.dim_head, config.dropout)
        self.decoder = Decoder(config.image_size, config.patch_size, config.dim, config.depth, config.heads, config.mlp_dim, config.channels, config.dim_head, config.dropout)
        self.quantize = VectorQuantizer(config.n_embed, config.embed_dim, config.beta)
        self.prev_quant = nn.Linear(config.dim, config.embed_dim)
        self.post_quant = nn.Linear(config.embed_dim, config.dim)
        
        # reconstruction loss

        self.recon_loss_fn = F.l1_loss
            
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def encode(self, x):
        x, cls_feat = self.encoder(x)
        x = self.prev_quant(x)
        x, loss, indices = self.quantize(x)
        return x, loss, indices, cls_feat
    
    def decode(self, x):
        x = self.post_quant(x)
        x = self.decoder(x)
        return x
    
    def decode_from_indice(self, indice):
        z_q = self.quantize.decode_from_indice(indice)
        img = self.decode(z_q)
        return img
    
    def forward(self, img):
        latent, commit_loss, indices, cls_feat = self.encode(img)
        rec = self.decode(latent)
        
        # reconstruction loss
        recon_loss = self.recon_loss_fn(rec, img)
        
        # combine loss
        loss = recon_loss + commit_loss

        return rec, loss, cls_feat
    
    def from_pretrained(self, path):
        return self.load_state_dict(torch.load(path))
    


        