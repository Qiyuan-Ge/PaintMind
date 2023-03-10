import os
import torch
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm
from einops import rearrange
from accelerate import Accelerator
from torchvision.utils import save_image
from paintmind.util.lr_scheduler import create_scheduler
from paintmind.ldm import FrozenVQModel
from paintmind.encoder.clip import FrozenOpenCLIPEmbedder, CLIP_ARCH, CLIP_VERSION


def exists(x):
    return x is not None

def transform_reverse(x):
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    
    return x

def cosine_masked_p_generator():
    p = np.sin(0.5 * np.pi * (0.02 + np.random.rand(1)))
    
    return p.item()
    
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
        self.scheduler = create_scheduler(self.optimizer, num_epochs, len(dataloader), lr_min, warmup_steps, warmup_lr_init)
        self.stage_two_model, self.optimizer, self.dataloader, self.scheduler = self.accelerator.prepare(self.stage_two_model, self.optimizer, dataloader, self.scheduler)
        
        self.max_grad_norm = max_grad_norm
        precision = 'fp16' if mixed_precision == 'fp16' else 'fp32'
        self.clip = FrozenOpenCLIPEmbedder(arch=clip_arch, version=clip_version, device=self.device, max_length=77, freeze=True, layer="last", precision=precision)
        
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
    def sample(self, b, h, w, text=None, timesteps=20):
        stage_two_model = self.accelerator.unwrap_model(self.stage_two_model)
        mask_token = stage_two_model.mask_token
        l = h * w
        x = mask_token.repeat(b, l, 1)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        img0 = self.stage_one_model.decode(x).cpu()
        imgs = [img0]
        for i in tqdm(range(1, timesteps+1), desc='sampling loop time step', total=timesteps):
            pred = stage_two_model.inference(x, text)
            pred = F.softmax(pred, dim=-1)
            values, indice = pred.max(dim=-1)
            
            prob_ids_descend = torch.argsort(values, descending=True)
            code_ids_restore = torch.argsort(prob_ids_descend, dim=1)
            code_ids_descend = torch.gather(indice, dim=1, index=prob_ids_descend)
            
            len_keep = l - int(l*np.cos(0.5*np.pi*i/timesteps))
            code_ids_keep = code_ids_descend[:, :len_keep]
            x = self.stage_one_model.get_embedding()(code_ids_keep)
  
            mask_tokens = mask_token.repeat(b, l-len_keep, 1)
  
            x = torch.cat([x, mask_tokens], dim=1)
            x = torch.gather(x, dim=1, index=code_ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            
            img = self.stage_one_model.decode(x)
            imgs.append(img.cpu())
            
        images = torch.cat(imgs, dim=0)
        
        image_path = os.path.join(self.sample_dir, f"sample-{self.steps}.png")
        
        save_image(transform_reverse(images), image_path, nrow=7)
        
    def loss_func(self, x, y, mask):
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1)
        mask = mask.reshape(-1)
        
        loss = F.cross_entropy(x, y, reduction='none')
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def compute_acc(self, x, y, mask):
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1)
        mask = mask.reshape(-1)
        
        n = (x.argmax(dim=-1) == y)
        acc = (n * mask).sum() / mask.sum()
        
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
                        bs = imgs.shape[0]
                        with self.accelerator.autocast():
                            
                            z, indices = self.stage_one_model.encode(imgs)
                            text = self.text_embbed(text)                      
                            pred, mask = self.stage_two_model(z, text, mask_ratio=cosine_masked_p_generator())                     
                            loss = self.loss_func(pred, indices, mask)
                            accu = self.compute_acc(pred, indices, mask)
                    
                        self.accelerator.backward(loss)
                        
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.stage_two_model.parameters(), self.max_grad_norm)
                        
                        self.optimizer.step()
                        self.scheduler.step_update(self.steps)
                        self.optimizer.zero_grad()
                    
                    log.add({'total_loss':loss.item()*bs, 'total_acc':accu*bs, 'n_sample':bs})
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
                        _, c, h, w = z.shape
                        n = 1
                        self.sample(n, h, w, text=text[:n], timesteps=20)
                        
                    self.steps += 1
                       
            log.reset()
            torch.cuda.empty_cache()
            
        self.accelerator.end_training()        
        print("Train finished!")
        
