import torch
import torch.nn as nn
from .layers import Encoder, Decoder
from .quantize import VectorQuantizer


class VQModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(**config.encdec)
        self.decoder = Decoder(**config.encdec)
        self.quantize = VectorQuantizer(config.n_embed, config.embed_dim, config.beta)
        self.prev_conv = nn.Conv2d(config.encdec["z_channels"], config.embed_dim, 1)
        self.post_conv = nn.Conv2d(config.embed_dim, config.encdec["z_channels"], 1)
            
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def encode(self, x):
        x = self.encoder(x)
        x = self.prev_conv(x)
        x, loss, indices = self.quantize(x)
        return x, loss, indices
    
    def decode(self, x):
        x = self.post_conv(x)
        x = self.decoder(x)
        return x
    
    def forward(self, img):
        z, loss, indices = self.encode(img)
        xrec = self.decode(z)

        return xrec, loss
    
    def decode_from_indice(self, indice):
        z_q = self.quantize.get_codebook_entry(indice)
        img = self.decode(z_q)
        return img
    
    def from_pretrained(self, path):
        return self.load_state_dict(torch.load(path))
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight
    


        