import torch
from functools import partial
from torch.utils.data import DataLoader
from paintmind.text_encoder.clip import tokenize, DEFAULT_CLIP_NAME

    
class collate_fn:
    def __init__(self, text_model_name=DEFAULT_CLIP_NAME):
        self.text_model_name = text_model_name
        self.tokenize = partial(tokenize, name=text_model_name)
    
    def __call__(self, batch):
        img_batch = []
        txt_batch = []
        for img, txt in batch:
            img_batch.append(img.unsqueeze(0))
            txt_batch.append(txt)
        img_batch = torch.cat(img_batch, dim=0)
        input_ids, attn_mask = self.tokenize(txt_batch)

        return img_batch, input_ids, attn_mask

   
def TxtImgDataloader(dataset, batch_size, shuffle=True, text_model_name=DEFAULT_CLIP_NAME, num_workers=0, pin_memory=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn(text_model_name), pin_memory=pin_memory)
