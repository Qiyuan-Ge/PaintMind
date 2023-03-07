import torch
from omegaconf import OmegaConf
from .models.autoencoder import VQModel

class FrozenVQModel:
    def __init__(self, config_path, ckpt_path=None, device='cuda', freeze=True):
        cfg = OmegaConf.load(config_path)
        model = VQModel(**cfg.model.params)
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        self.model = model.to(device)
        if freeze:
            self.freeze()
    
    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
    def encode(self, x):
        quant, _, (_,_,indice) = self.model.encode(x)
        
        return quant, indice
    
    def decode(self, z):
        x = self.model.decode(z)
        
        return x
    
    def get_embedding(self):
        return self.model.quantize.embedding
    
    def get_codebook_entry(self, indice, shape=None):
        return self.model.quantize.get_codebook_entry(indice, shape)
        
        