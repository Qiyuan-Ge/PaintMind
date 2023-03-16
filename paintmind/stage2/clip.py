import torch
import open_clip
import torch.nn as nn
from typing import List
from torch.utils.checkpoint import checkpoint

CLIP_ARCH = "ViT-H-14"
CLIP_VERSION = "laion2b_s32b_b79k"


def tokenize(text: List[str], arch=CLIP_ARCH):
    tokenizer = open_clip.get_tokenizer(arch)
    
    return tokenizer(text)


class OpenCLIPEmbedder(nn.Module):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        "last",
        "penultimate"
    ]
    def __init__(self, arch=CLIP_ARCH, version=CLIP_VERSION, device="cuda", max_length=77, layer="last", precision='fp32'):
        super().__init__()
        assert layer in self.LAYERS
        model = open_clip.create_model(arch, device=device, pretrained=version, precision=precision)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)
