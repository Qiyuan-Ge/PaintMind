import os
import torch
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm
from accelerate import Accelerator
from torchvision.utils import save_image
from timm.scheduler.cosine_lr import CosineLRScheduler

from paintmind import taming
from paintmind.text_encoder.clip import CLIP, DEFAULT_CLIP_NAME
#from paintmind.text_encoder.t5 import T5, DEFAULT_T5_NAME


def exists(x):
    return x is not None

# def transform_reverse(x, m=IMAGENET_DEFAULT_MEAN, s=IMAGENET_DEFAULT_STD):
#     m = torch.tensor(m).to(x.device)
#     s = torch.tensor(s).to(x.device)
#     m = m.reshape(1, -1, 1, 1)
#     s = s.reshape(1, -1, 1, 1)
    
#     return x * s + m

def transform_reverse(x):
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    
    return x

def linear_masked_p_schedule(timesteps=75):
    p_min = 0.25
    p_max = 1.00
    
    return np.linspace(p_min, p_max, timesteps)
    
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
                 warmup_steps,
                 warmup_lr_init,
                 ema_decay=None,
                 max_grad_norm=1.0, 
                 text_model_name=DEFAULT_CLIP_NAME, 
                 text_max_length=77,
                 checkpoint_path=None,
                 sample_interval=1000,
                 save_every_n_step=1000,
                 vqgan_config_path=None,
                 vqgan_pretrained_path=None,
                 gradient_accumulation_steps=8,
                 ):
        
        self.accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
        self.device = self.accelerator.device
        
        self.use_ema = False
        if exists(ema_decay):
            self.use_ema = True
            self.ema = EMA(model, ema_decay)
        
        self.model = model
        self.num_epoch = num_epochs
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = CosineLRScheduler(self.optimizer, t_initial=num_epochs*len(dataloader), lr_min=lr_min, warmup_t=warmup_steps, warmup_lr_init=warmup_lr_init)
        self.model, self.optimizer, self.dataloader, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, dataloader, self.scheduler)
        
        self.max_grad_norm = max_grad_norm
        
        self.text = CLIP(text_model_name, text_max_length, device=self.device)
        self.vqae = taming.load_vqgan(vqgan_config_path, vqgan_pretrained_path).to(self.device)
        
        self.mask_p = linear_masked_p_schedule()
        self.sample_interval = sample_interval
        self.save_every_n_step = save_every_n_step
        
        self.checkpoint_path = checkpoint_path
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        self.sample_dir = os.path.join(checkpoint_path, 'sample')
        os.makedirs(self.sample_dir, exist_ok=True)
         
    def save(self):
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.accelerator.save(unwrapped_model.state_dict(), os.path.join(self.checkpoint_path, f'model_step_{self.steps}.pt'))
    
    def _ema_update(self):
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.ema.update(unwrapped_model)
        self.accelerator.save(self.ema.model.state_dict(), os.path.join(self.checkpoint_path, 'model_ema.pt'))
        
    @torch.no_grad()
    def vqae_encode(self, imgs):
        latent, _, [_, _, indices] = self.vqae.encode(imgs)
        
        return latent
    
    @torch.no_grad()
    def vqae_decode(self, latent):
        imgs = self.vqae.decode(latent)
        
        return imgs
    
    @torch.no_grad()
    def text_encode(self, token_ids):
        text_emb = self.text.encode_tokens(token_ids)
        
        return text_emb
           
    def sample(self, imgs, pred, n=4):
        unwrap_model = self.accelerator.unwrap_model(self.model)
        
        imgs = imgs[:n]
        pred = pred[:n]
        
        pred = unwrap_model.unpatchify(pred)
        pred = self.vqae_decode(pred)
        
        gen_images = torch.cat([imgs, pred], dim=0)
        
        image_path = os.path.join(self.sample_dir, f"sample-{self.steps}.png")
        
        save_image(transform_reverse(gen_images), image_path, nrow=n)
    
    def train(self):
        self.steps = 0
        for epoch in range(self.num_epoch):
            log = Log()
            with tqdm(self.dataloader, dynamic_ncols=True, disable=not self.accelerator.is_local_main_process) as tqdm_dataloader:
                for batch in tqdm_dataloader:
                    with self.accelerator.accumulate(self.model):
                        imgs, token_ids, text_mask = batch #text is token ids
                        latent = self.vqae_encode(imgs)
                        text_emb = self.text_encode(token_ids)
                        
                        loss, pred = self.model(latent, text_emb, text_mask, mask_ratio=np.random.choice(self.mask_p))
                    
                        self.accelerator.backward(loss)
                    
                        if exists(self.max_grad_norm):  
                            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        
                        self.optimizer.step()
                        self.steps += 1
                        self.scheduler.step(self.steps)
                        self.optimizer.zero_grad()
                    
                    if self.use_ema:
                        self._ema_update()
                    
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
                    
                    if self.steps % self.save_every_n_step == 0:
                        self.save()
                        
                    if self.steps % self.sample_interval == 0:
                        self.sample(imgs, pred)
                
        print("Train finished!")
        
