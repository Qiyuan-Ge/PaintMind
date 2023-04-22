import torch
import torch.nn as nn
from .layers import Encoder, Decoder
from .quantize import VectorQuantizer


class VQModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(**config.enc)
        self.decoder = Decoder(**config.dec)
        self.quantize = VectorQuantizer(config.n_embed, config.embed_dim, config.beta)
        self.prev_quant = nn.Linear(config.enc['dim'], config.embed_dim)
        self.post_quant = nn.Linear(config.embed_dim, config.dec['dim'])  
            
    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def encode(self, x):
        x = self.encoder(x)
        x = self.prev_quant(x)
        x, loss, indices = self.quantize(x)
        return x, loss, indices
    
    def decode(self, x):
        x = self.post_quant(x)
        x = self.decoder(x)
        return x.clamp(-1.0, 1.0)
    
    def forward(self, img):
        z, loss, indices = self.encode(img)
        rec = self.decode(z)

        return rec, loss
    
    def decode_from_indice(self, indice):
        z_q = self.quantize.decode_from_indice(indice)
        img = self.decode(z_q)
        return img
    
    def from_pretrained(self, path):
        return self.load_state_dict(torch.load(path))
    


        
