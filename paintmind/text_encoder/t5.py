import torch
import transformers
from typing import List
from einops import rearrange
from transformers import T5Tokenizer, T5EncoderModel

transformers.logging.set_verbosity_error()

DEFAULT_T5_NAME = 'google/flan-t5-base' 

MAX_LENGTH = 256

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    
    return d() if callable(d) else d

def get_tokenizer(name=DEFAULT_T5_NAME):
    tokenizer = T5Tokenizer.from_pretrained(name, model_max_length=MAX_LENGTH)
    
    return tokenizer

def get_model(name=DEFAULT_T5_NAME, mixed_precision='no'):
    if mixed_precision == 'fp16':
        model = T5EncoderModel.from_pretrained(name, torch_dtype=torch.float16)
    else:
        model = T5EncoderModel.from_pretrained(name)
        
    return model

def tokenize(text: List[str], name=DEFAULT_T5_NAME):
    tokenizer = get_tokenizer(name)
    encodeddd = tokenizer.batch_encode_plus(text, return_tensors="pt", padding='longest', max_length=MAX_LENGTH, truncation=True)
    input_ids = encodeddd.input_ids
    attn_mask = encodeddd.attention_mask
    
    return input_ids, attn_mask

def encode_tokens(token_ids, attn_mask=None, pad_id=0, name=DEFAULT_T5_NAME):
    model = get_model(name)
    model.eval()
    attn_mask = default(attn_mask, lambda: (token_ids!=pad_id).long())
    with torch.no_grad():
        output = model(input_ids=token_ids, attention_mask=attn_mask)
        encoded_text = output.last_hidden_state.detach()
    encoded_text = encoded_text.masked_fill(~rearrange(attn_mask.bool(), '... -> ... 1'), 0.)  
      
    return encoded_text, attn_mask

def encode_text(texts: List[str], name=DEFAULT_T5_NAME):
    token_ids, attn_mask = tokenize(texts, name=name)
    encoded_text, _ = encode_tokens(token_ids, attn_mask=attn_mask, name=name)

    return encoded_text, attn_mask

class T5:
    def __init__(self, name=DEFAULT_T5_NAME, max_length=MAX_LENGTH, device='cuda', mixed_precision='no'):
        self.tokenizer = get_tokenizer(name)
        self.t5 = get_model(name, mixed_precision).to(device)
        self.t5.eval()
        self.max_length = max_length
        self.device = device
        
    @torch.no_grad()
    def tokenize(self, text: List[str]):
        encodeddd = self.tokenizer.batch_encode_plus(text, return_tensors="pt", padding='longest', max_length=MAX_LENGTH, truncation=True)
        input_ids = encodeddd.input_ids
        attn_mask = encodeddd.attention_mask
    
        return input_ids, attn_mask
        
    @torch.no_grad()
    def encode_tokens(self, token_ids, attn_mask=None, pad_id=0):
        attn_mask = default(attn_mask, lambda: (token_ids!=pad_id).long())
        output = self.t5(input_ids=token_ids, attention_mask=attn_mask)
        encoded_text = output.last_hidden_state.detach()
        encoded_text = encoded_text.masked_fill(~rearrange(attn_mask.bool(), '... -> ... 1'), 0.)
        
        return encoded_text
    
    @torch.no_grad()
    def encode_text(self, texts: List[str]):
        token_ids, attn_mask = self.tokenize(texts)
        encoded_text = self.encode_tokens(token_ids, attn_mask=attn_mask)
        
        return encoded_text, attn_mask 
        