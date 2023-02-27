import torch
import transformers
from typing import List
from transformers import CLIPTokenizer, CLIPTextModel

transformers.logging.set_verbosity_error()

DEFAULT_CLIP_NAME = 'openai/clip-vit-large-patch14'
MAX_LENGTH = 77

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    
    return d() if callable(d) else d

def get_tokenizer(name=DEFAULT_CLIP_NAME):
    tokenizer = CLIPTokenizer.from_pretrained(name)
    
    return tokenizer

def get_model(name=DEFAULT_CLIP_NAME):
    model = CLIPTextModel.from_pretrained(name)
    
    return model

def tokenize(text: List[str], name=DEFAULT_CLIP_NAME):
    tokenizer = get_tokenizer(name)
    encodeddd = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    input_ids = encodeddd.input_ids
    attn_mask = encodeddd.attention_mask
    
    return input_ids, attn_mask

def encode_tokens(token_ids, name=DEFAULT_CLIP_NAME):
    model = get_model(name)
    model.eval()
    with torch.no_grad():
        output = model(input_ids=token_ids)
        encoded_text = output.last_hidden_state.detach()
        
    return encoded_text

def encode_text(texts: List[str], name=DEFAULT_CLIP_NAME):
    token_ids, attn_mask = tokenize(texts, name=name)
    encoded_text = encode_tokens(token_ids, name=name)

    return encoded_text, attn_mask


class CLIP:
    def __init__(self, name=DEFAULT_CLIP_NAME, max_length=MAX_LENGTH, device='cuda'):
        self.tokenizer = CLIPTokenizer.from_pretrained(name)
        self.clip_text = CLIPTextModel.from_pretrained(name).to(device)
        self.clip_text.eval()
        self.max_length = max_length
        self.device = device
        
    @torch.no_grad()
    def tokenize(self, text: List[str]):
        encodeddd = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        input_ids = encodeddd.input_ids
        attn_mask = encodeddd.attention_mask
    
        return input_ids, attn_mask
        
    @torch.no_grad()
    def encode_tokens(self, token_ids):
        output = self.clip_text(input_ids=token_ids)
        encoded_text = output.last_hidden_state.detach()
        
        return encoded_text
    
    @torch.no_grad()
    def encode_text(self, texts: List[str]):
        input_ids, attn_mask = self.tokenize(texts)
        encoded_text = self.encode_tokens(input_ids)
        
        return encoded_text, attn_mask 
        
        