import torch
import open_clip
from typing import List
from einops import repeat

DEFAULT_CLIP_NAME = 'coca_ViT-L-14'
DEFAULT_CLIP_PRETRAINED = 'mscoco_finetuned_laion2b_s13b_b90k'

def get_tokenizer(version=DEFAULT_CLIP_NAME):
    tokenizer = open_clip.get_tokenizer(version)
    
    return tokenizer

def get_model(version=DEFAULT_CLIP_NAME, pretrained=DEFAULT_CLIP_PRETRAINED, precision='fp32'):
    model = open_clip.create_model(version, pretrained=pretrained, precision=precision)
    
    return model

def tokenize(text: List[str], version=DEFAULT_CLIP_NAME):
    tokenizer = get_tokenizer(version)
    
    return tokenizer(text)

class FrozenCLIP:
    def __init__(self, version=DEFAULT_CLIP_NAME, pretrained=DEFAULT_CLIP_PRETRAINED, precision='fp32', device='cuda', n_repeat=1):
        super().__init__()
        self.tokenizer = get_tokenizer(version)
        self.model = get_model(version, pretrained, precision).to(device)
        
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False
        
        self.n_repeat = n_repeat
    
    @torch.no_grad()
    def encode_tokens(self, token_ids):
        z = self.model.encode_text(token_ids)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        
        return z
