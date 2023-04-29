import torch
import kornia
import open_clip
import transformers
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import T5Tokenizer, T5EncoderModel

transformers.logging.set_verbosity_error()

# "ViT-L-14" "laion2b_s32b_b82k"
# "ViT-H-14" "laion2b_s32b_b79k"

CLIP_ARCH = 'ViT-L-14'
CLIP_VERSION = 'laion2b_s32b_b82k'


class T5TextEmbedder(nn.Module):
    def __init__(self, version="google/flan-t5-xl", device="cuda", max_length=77, freeze=True):  
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class CLIPTextEmbedder(nn.Module):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        "last",
        "penultimate"
    ]
    def __init__(self, arch=CLIP_ARCH, version=CLIP_VERSION, device="cuda", max_length=77, layer="last", precision='fp32', freeze=True):
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
        
        if freeze:
            self.freeze()

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

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
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
    def __init__(self, arch=CLIP_ARCH, version=CLIP_VERSION, device="cuda", precision='fp32', freeze=True):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=version, device=device, precision=precision)
        self.model = model.visual
        self.device = device
        
        if freeze:
            self.freeze()
        
    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False
            
    def preprocess(self, x):
        x = kornia.geometry.resize(x, (224, 224), interpolation='bicubic', align_corners=True, antialias=False)
        return x

    def forward(self, image):
        x = self.preprocess(image)
        z = self.encode_with_transformer(x)
        return z
    
    def encode_with_transformer(self, x):
        x = self.model.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        x = torch.cat([self.model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.model.positional_embedding.to(x.dtype)
        x = self.model.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:]
        
        return x
    
    def encode(self, image):
        return self(image)
