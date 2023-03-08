import os
import torch
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm
from einops import rearrange
from accelerate import Accelerator
from torchvision.utils import save_image
from timm.scheduler.cosine_lr import CosineLRScheduler
from paintmind.ldm import FrozenVQModel
from paintmind.encoder.clip import FrozenOpenCLIPEmbedder, CLIP_ARCH, CLIP_VERSION


def exists(x):
    return x is not None


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
                 max_grad_norm=1.0, 
                 clip_arch=CLIP_ARCH,
                 clip_version=CLIP_VERSION,
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
        
        self.num_epoch = num_epochs
        self.stage_one_model = FrozenVQModel(first_stage_config_path, first_stage_pretrained_path, device=self.device, freeze=True)
        self.stage_two_model = model
        self.optimizer = torch.optim.AdamW(self.stage_two_model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = CosineLRScheduler(self.optimizer, t_initial=num_epochs*len(dataloader), lr_min=lr_min, warmup_t=warmup_steps, warmup_lr_init=warmup_lr_init)
        self.stage_two_model, self.optimizer, self.dataloader, self.scheduler = self.accelerator.prepare(self.stage_two_model, self.optimizer, dataloader, self.scheduler)
        
        self.max_grad_norm = max_grad_norm
        precision = 'fp16' if mixed_precision == 'fp16' else 'fp32'
        self.clip = FrozenOpenCLIPEmbedder(arch=clip_arch, version=clip_version, device=self.device, max_length=77, freeze=True, layer="last", precision=precision)
        
        self.mask_ratio = linear_masked_p_schedule()
        self.sample_interval = sample_interval
        self.save_every_n_step = save_every_n_step
        
        self.checkpoint_path = checkpoint_path
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        self.sample_dir = os.path.join(checkpoint_path, 'sample')
        os.makedirs(self.sample_dir, exist_ok=True)
         
    def save(self):
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.stage_two_model)
        self.accelerator.save(unwrapped_model.state_dict(), os.path.join(self.checkpoint_path, f'model_step_{self.steps}.pt'))
        
    @torch.no_grad()
    def text_embbed(self, tokens):
        text = self.clip.encode_with_transformer(tokens)
        
        return text
    
    @torch.no_grad()       
    def sample(self, images, scores, n, h, w, c):
        
        images = images[:n]
        scores = scores[:n]
        
        indice = scores.argmax(dim=-1)
        
        quants = self.stage_one_model.get_codebook_entry(indice, shape=(n, h, w, c))
        imgrec = self.stage_one_model.decode(quants)
        
        image_path = os.path.join(self.sample_dir, f"sample-{self.steps}.png")
        
        save_image(transform_reverse(torch.cat([images, imgrec], dim=0)), image_path, nrow=n)
        
    def loss_func(self, x, label, mask):
        loss = F.cross_entropy(x, label, reduction='none')
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def compute_acc(self, x, y):
        acc = (x.argmax(dim=-1) == y).sum() / y.numel()
        
        return acc.item()
    
    def train(self):
        self.steps = 0
        self.accelerator.init_trackers("paintmind")
        log = Log()
        for epoch in range(self.num_epoch):
            with tqdm(self.dataloader, dynamic_ncols=True, disable=not self.accelerator.is_local_main_process) as tqdm_dataloader:
                for batch in tqdm_dataloader:
                    with self.accelerator.accumulate(self.stage_two_model):
                        imgs, text = batch

                        with self.accelerator.autocast():
                            
                            z, indices = self.stage_one_model.encode(imgs)
                            # mask_ratio = np.random.choice(self.mask_ratio)
                            text = self.text_embbed(text)                      
                            pred, mask = self.stage_two_model(z, text, mask_ratio=0.75)                     
                            loss = self.loss_func(pred.reshape(-1, pred.shape[-1]), indices.reshape(-1), mask.reshape(-1))
                            acc = self.compute_acc(pred.reshape(-1, pred.shape[-1]), indices.reshape(-1))
                    
                        self.accelerator.backward(loss)
                        
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.stage_two_model.parameters(), self.max_grad_norm)
                        
                        self.steps += 1
                        self.optimizer.step()
                        self.scheduler.step(self.steps)
                        self.optimizer.zero_grad()
                    
                    log.add({'total_loss':loss.item()*b, 'total_acc':acc*b, 'n_sample':b})
                    log.update({'loss':loss.item(), 'lr':self.optimizer.param_groups[0]['lr']})
   
                    tqdm_dataloader.set_postfix(
                        ordered_dict={
                            "Epoch"      : epoch,
                            "Loss"       : log['loss'],
                            "MeanLoss"   : log['total_loss']/log['n_sample'],
                            "Acc"        : log['total_acc']/log['n_sample'],
                            "LR"         : log['lr'],
                        }
                    )
                    self.accelerator.log({"loss": log['total_loss']/log['n_sample'], "lr": log['lr']}, step=self.steps)
                    
                    if self.steps % self.save_every_n_step == 0:
                        self.save()
                        
                    if self.steps % self.sample_interval == 0:
                        b, c, h, w = z.shape
                        self.sample(imgs, pred.detach(), n=4, h=h, w=w, c=c)
                       
            log.reset()
            torch.cuda.empty_cache()
            
        self.accelerator.end_training()        
        print("Train finished!")
        
