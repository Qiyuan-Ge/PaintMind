import torch
from omegaconf import OmegaConf
from .vqgan import VQModel


def load_vqgan(config_path, ckpt_path=None):
    config = OmegaConf.load(config_path)
    model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    
    return model.eval()

