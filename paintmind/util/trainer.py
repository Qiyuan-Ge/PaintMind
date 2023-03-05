import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from accelerate import Accelerator
from torchvision.utils import save_image
from timm.scheduler.cosine_lr import CosineLRScheduler
from paintmind.ldm.models.autoencoder import VQModel
from paintmind.text_encoder.clip import FrozenCLIP, DEFAULT_CLIP_NAME, DEFAULT_CLIP_PRETRAINED


def exists(x):
    return x is not None


def transform_reverse(x):
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    
    return x

def load_first_stage(config_path, ckpt_path=None):
    config = OmegaConf.load(config_path)
    model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    
    return model.eval()

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
                 clip_version=DEFAULT_CLIP_NAME,
                 clip_pretrained=DEFAULT_CLIP_PRETRAINED,
                 checkpoint_path=None,
                 sample_interval=1000,
                 save_every_n_step=1000,
                 first_stage_config_path=None,
                 first_stage_pretrained_path=None,
                 gradient_accumulation_steps=4,
                 mixed_precision='no',
                 log_dir="./log"
                 ):
        
        self.accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, 
                                       mixed_precision=mixed_precision,
                                       log_with="tensorboard",
                                       logging_dir=log_dir,
                           )
        self.device = self.accelerator.device
        os.makedirs(log_dir, exist_ok=True)
        
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
        precision = 'fp16' if mixed_precision == 'fp16' else 'fp32'
        self.text = FrozenCLIP(version=clip_version, pretrained=clip_pretrained, precision=precision, device=self.device, n_repeat=1)
        self.first_stage = load_first_stage(first_stage_config_path, first_stage_pretrained_path).to(self.device)
        for param in self.first_stage.parameters():
            param.requires_grad = False
        
        self.mask_ratio = linear_masked_p_schedule()
        self.sample_interval = sample_interval
        self.save_every_n_step = save_every_n_step
        
        self.checkpoint_path = checkpoint_path
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        self.sample_dir = os.path.join(checkpoint_path, 'sample')
        os.makedirs(self.sample_dir, exist_ok=True)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
         
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
        quant, diff, (_,_,ind) = self.first_stage.encode(imgs)
        
        return quant, ind
    
    @torch.no_grad()
    def vqae_decode(self, quant):
        imgs = self.first_stage.decode(quant)
        
        return imgs
    
    @torch.no_grad()
    def text_encode(self, token_ids):
        text_emb = self.text.encode_tokens(token_ids)
        
        return text_emb
    
    @torch.no_grad()       
    def sample(self, images, scores, n, h, w, c):
        
        images = images[:n]
        scores = scores[:n]
        
        indice = scores.argmax(dim=-1)
        
        quants = self.first_stage.quantize.get_codebook_entry(indice, shape=(n, h, w, c))
        imgrec = self.vqae_decode(quants)
        
        image_path = os.path.join(self.sample_dir, f"sample-{self.steps}.png")
        
        save_image(transform_reverse(torch.cat([images, imgrec], dim=0)), image_path, nrow=n)
        
    def loss_func(self, x, label):
        loss = F.cross_entropy(x, label)
        
        return loss
    
    def train(self):
        self.steps = 0
        self.accelerator.init_trackers("my_project")
        log = Log()
        for epoch in range(self.num_epoch):
            with tqdm(self.dataloader, dynamic_ncols=True, disable=not self.accelerator.is_local_main_process) as tqdm_dataloader:
                for batch in tqdm_dataloader:
                    with self.accelerator.accumulate(self.model):
                        images, tokens = batch

                        with self.accelerator.autocast():
                            quants, indice = self.vqae_encode(images)
                            
                            mask_ratio = np.random.choice(self.mask_ratio)
                            free_guide = np.random.random()
                            
                            if mask_ratio > 0.25 and free_guide > 0.9:
                                text_embs = None
                            else:
                                text_embs = self.text_encode(tokens)
                                
                            logits = self.model(quants, text_embs, text_mask=None, mask_ratio=mask_ratio)
                            logits = rearrange(logits, 'b c h w -> b (h w) c')
                            scores = self.logit_scale.exp() * torch.matmul(F.normalize(logits, dim=2), self.first_stage.quantize.embedding.weight.T)
                            loss = self.loss_func(scores.reshape(-1, scores.shape[-1]), indice.reshape(-1))
                    
                        self.accelerator.backward(loss)
                        
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        
                        self.steps += 1
                        self.optimizer.step()
                        self.scheduler.step(self.steps)
                        self.optimizer.zero_grad()
                    
                    if self.use_ema:
                        self._ema_update()
                    
                    bs = images.shape[0]
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
                    self.accelerator.log({"loss": log['total_loss']/log['n_sample'], "lr": log['lr']}, step=self.steps)
                    
                    if self.steps % self.save_every_n_step == 0:
                        self.save()
                        
                    if self.steps % self.sample_interval == 0:
                        _, c, h, w = quants.shape
                        self.sample(images, scores.detach(), n=3, h=h, w=w, c=c)
                       
            log.reset()
            torch.cuda.empty_cache()
            
        self.accelerator.end_training()        
        print("Train finished!")
        
