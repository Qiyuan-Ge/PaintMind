import torch
from functools import partial
from torch.utils.data import DataLoader
from paintmind.stage2.clip import tokenize, CLIP_ARCH

    
class collate_fn:
    def __init__(self, clip_arch=CLIP_ARCH):
        self.tokenize = partial(tokenize, arch=clip_arch)
    
    def __call__(self, batch):
        img_batch = []
        txt_batch = []
        for img, txt in batch:
            img_batch.append(img.unsqueeze(0))
            txt_batch.append(txt)
        img_batch = torch.cat(img_batch, dim=0)
        input_ids = self.tokenize(txt_batch)

        return img_batch, input_ids

   
def TxtImgDataloader(dataset, batch_size, shuffle=True, clip_arch=CLIP_ARCH, num_workers=0, pin_memory=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn(clip_arch), pin_memory=pin_memory)
