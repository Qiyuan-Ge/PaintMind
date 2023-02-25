import os
import pathlib
import torch
from tqdm.auto import tqdm
from copy import deepcopy
from accelerate import Accelerator
from timm.scheduler.cosine_lr import CosineLRScheduler

from paintmind.text_encoder.clip import CLIP, DEFAULT_CLIP_NAME


def exists(x):
    return x is not None


class Log:
    def __init__(self):
        self.data = {}
    
    def add(self, name_value):
        for name, value in name_value.items():
            if name not in self.data:
                self.data[name] = value
            else:
                self.data[name] += value
    
    def update(self, name_value):
        for name, value in name_value.items():
            self.data[name] = value
    
    def reset(self):
        self.data = {}

    def __getitem__(self, name):
        return self.data[name]


class EMA:
    """Exponential Moving Average
    Args:
        model:
        decay:
        device:
    """    
    def __init__(self, model, decay=0.9999, device='cpu'):
        self.model = deepcopy(model)
        self.decay = decay
        self.device = device
        self.model.to(device)
        
    @torch.no_grad()
    def update(self, model2):
        for p_ema, p_model in zip(self.model.state_dict().values(), model2.state_dict().values()):
            p_model = p_model.to(self.device)
            p_ema.copy_(self.func(p_ema, p_model))
            
    def func(self, p1, p2):
        return self.decay * p1 + (1 - self.decay) * p2


class PaintMindTrainer:
    def __init__(self, 
                 model, 
                 lr, 
                 wd, 
                 dataloader,
                 num_epochs,
                 lr_min,
                 warmup_epochs,
                 warmup_lr_init,
                 ema_decay=0.9999,
                 max_grad_norm=1.0, 
                 clip_name=DEFAULT_CLIP_NAME, 
                 text_max_length=77,
                 checkpoint_path=None,
                 ):
        
        self.accelerator = Accelerator()
        
        self.ema = EMA(model, ema_decay)
        
        self.model = model
        self.num_epoch = num_epochs
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = CosineLRScheduler(self.optimizer, t_initial=num_epochs, lr_min=lr_min, warmup_t=warmup_epochs, warmup_lr_init=warmup_lr_init)
        
        self.model, self.optimizer, self.dataloader, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, dataloader, self.scheduler)
        
        self.max_grad_norm = max_grad_norm
        
        self.clip = CLIP(clip_name, text_max_length, device=self.accelerator.device)
        self.checkpoint_path = checkpoint_path
         
    def _ema_update(self, model):
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(model)
        self.ema.update(unwrapped_model)
        self.accelerator.save(self.ema.model.state_dict(), self.checkpoint_path)
    
    def train(self):
        for epoch in range(self.num_epoch):
            log = Log()
            with tqdm(self.dataloader, dynamic_ncols=True, disable=not self.accelerator.is_local_main_process) as tqdm_dataloader:
                for batch in tqdm_dataloader:
                    self.optimizer.zero_grad()
                    
                    imgs, token_ids, text_mask = batch #text is token ids
                    text_emb = self.clip.encode_tokens(token_ids)
                    loss, pred = self.model(imgs, text_emb, text_mask, mask_ratio=0.75)
                    
                    self.accelerator.backward(loss)
                    
                    if exists(self.max_grad_norm):  
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        
                    self.optimizer.step()
                    
                    self._ema_update(self.model)
                    
                    bs = imgs.shape[0]
                    log.add({'total_loss':loss.item()*bs, 'n_sample':bs})
                    log.update({'loss':loss.item(), 'lr':self.optimizer.param_groups[0]['lr']})
                    tqdm_dataloader.set_postfix(
                        ordered_dict={
                            "Epoch"      : epoch,
                            "Loss"       : log['loss'],
                            "MeanLoss"   : log['total_loss']/log['n_sample'],
                            "LR"         : log['lr'],
                        }
                    )
                
                self.scheduler.step(epoch)
        
        print("Train finished!")
        
