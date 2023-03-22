import torch
import kornia
import open_clip
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# "ViT-H-14" "laion2b_s32b_b79k"
# 'ViT-L-14', version='laion2b_s32b_b82k'

CLIP_ARCH = 'ViT-H-14'
CLIP_VERSION = 'laion2b_s32b_b79k'

class CLIPTextEmbedder(nn.Module):
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


class CLIPImageEmbedder(nn.Module):
    """
    Uses the OpenCLIP transformer encoder for image
    """
    def __init__(self, arch=CLIP_ARCH, version=CLIP_VERSION, device="cuda", precision='fp32'):
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=version, device=device, precision=precision)
        self.model = model.visual
        self.device = device
        
    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False
            
    def preprocess(self, x):
        x = kornia.geometry.resize(x, (224, 224), interpolation='bicubic', align_corners=True, antialias=False)
        return x

    def forward(self, image, preprocess=False):
        if preprocess:
            image = self.preprocess(image)
        z = self.model(image)
        return z
    
    def encode(self, image):
        return self(image)