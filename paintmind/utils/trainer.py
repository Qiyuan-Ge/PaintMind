import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader, random_split
from torch.autograd import grad as torch_grad
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
from einops import rearrange
from paintmind import build_model, build_config
from paintmind.stage1.discriminator import Discriminator
from paintmind.stage2.transformer import PaintMind
from paintmind.utils.lr_scheduler import build_scheduler




# discriminator

def log(t, eps = 1e-10):
    return torch.log(t + eps)


def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()


def hinge_gen_loss(fake):
    return -fake.mean()


def gradient_penalty(images, output, weight=10):

    gradients = torch_grad(
        outputs=output,
        inputs=images,
        grad_outputs=torch.ones(output.size(), device=images.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


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


class VQGANTrainer(nn.Module):
    def __init__(
        self, 
        vae_name,
        dataset,
        num_epoch,
        valid_size=10,
        base_lr=3e-4,
        batch_size=32,
        grad_accum_steps=1,
        mixed_precision='fp16',
        max_grad_norm=1.0,
        save_every_n=10000,
        sample_every_n=1000,
        result_folder=None,
        log_dir="./log"
    ):
        super().__init__()
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            kwargs_handlers=[kwargs],
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=grad_accum_steps, 
            log_with="tensorboard",
            logging_dir=log_dir,
        )
        self.vqvae = build_model(vae_name)
        
        discr_layers = 4
        dim=64
        layer_mults = list(map(lambda t: 2 ** t, range(discr_layers)))
        layer_dims = [dim * mult for mult in layer_mults]
        dims = (dim, *layer_dims)
        
        self.discr = Discriminator(dims=dims, channels=self.vqvae.channels)
        
        train_size = len(dataset) - valid_size
        self.train_ds, self.valid_ds = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))
        print(f"train dataset size: {train_size}, valid dataset size: {valid_size}")
        
        self.train_dl = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=batch_size, shuffle=False)
        
        self.optim = torch.optim.Adam(self.vqvae.parameters(), lr=base_lr)
        self.discr_optim = torch.optim.Adam(self.discr.parameters(), lr=base_lr)
        
        self.gen_loss = hinge_gen_loss
        self.discr_loss = hinge_discr_loss
        
        (
            self.vqvae,
            self.discr,
            self.optim,
            self.discr_optim,
            self.train_dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.vqvae,
            self.discr,
            self.optim,
            self.discr_optim,
            self.train_dl,
            self.valid_dl
        )
        
        self.num_epoch = num_epoch
        self.save_every_n = save_every_n
        self.max_grad_norm = max_grad_norm
        self.sample_every_n = sample_every_n
        
        self.model_saved_dir = os.path.join(result_folder, 'models')
        os.makedirs(self.model_saved_dir, exist_ok=True)
        
        self.image_saved_dir = os.path.join(result_folder, 'images')
        os.makedirs(self.image_saved_dir, exist_ok=True)
        
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'number of learnable parameters: {n_parameters//1e6}M')    
        
    def train(self):
        self.steps = 0
        self.accelerator.init_trackers("vqgan")
        log = Log()
        for epoch in range(self.num_epoch):
            self.vqvae.train()
            with tqdm(self.train_dl, dynamic_ncols=True, disable=not self.accelerator.is_main_process) as train_dl:
                for batch in train_dl:
                    if isinstance(batch, tuple) or isinstance(batch, list):
                        img = batch[0]
                    else:
                        img = batch
                    
                    with self.accelerator.accumulate(self.vqvae):
                        with self.accelerator.autocast():
                            rec, loss = self.vqvae(img)
                            rec.detach_()
                            gen_loss = self.gen_loss(self.discr(rec))
                            loss = loss + gen_loss   
                            
                        self.accelerator.backward(loss)
                        
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.vqvae.parameters(), self.max_grad_norm)
                            
                        self.optim.step()
                        self.optim.zero_grad()
                    
                    # update discriminator
                    img.requires_grad_()
                    with self.accelerator.accumulate(self.discr):
                        with self.accelerator.autocast():
                            rec_logits, img_logits = map(self.discr, (rec, img))
                            discr_loss = self.discr_loss(rec_logits, img_logits)
                            gp = gradient_penalty(img, img_logits)
                            discr_loss = discr_loss + gp  
                            
                        self.accelerator.backward(discr_loss)
                        
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.discr.parameters(), self.max_grad_norm)
                            
                        self.discr_optim.step()
                        self.discr_optim.zero_grad()
                        
                    self.steps += 1
                    
                    if (self.steps % self.sample_every_n) == 0:
                        self.evaluate()
                        
                    if self.steps % self.save_every_n == 0:
                        self.save()
                        
                    log.update({'vae loss':loss.item(), 'gen loss':gen_loss.item(), 'discr loss':discr_loss.item(), 'lr':self.optim.param_groups[0]['lr']})
   
                    train_dl.set_postfix(
                        ordered_dict={
                            "epoch"      : epoch,
                            "vae loss"   : log['vae loss'],
                            "gen loss"   : log['gen loss'],
                            "discr loss" : log['discr loss'],
                            "LR"         : log['lr'],
                        }
                    )
                    self.accelerator.log({"vae loss":log['vae loss'], "gen loss":log['gen loss'], "discr loss":log['discr loss'], "lr": log['lr']}, step=self.steps)
        
        self.accelerator.end_training()        
        print("Train finished!")
                    
    def save(self):
        self.accelerator.wait_for_everyone()
        state_dict = self.accelerator.unwrap_model(self.vqvae).state_dict()
        self.accelerator.save(state_dict, os.path.join(self.model_saved_dir, f'vit_vqvae_step_{self.steps}.pt'))
                                                       
    @torch.no_grad()
    def evaluate(self):
        self.vqvae.eval()
        with tqdm(self.valid_dl, dynamic_ncols=True, disable=not self.accelerator.is_local_main_process) as valid_dl:
            for i, batch in enumerate(valid_dl):
                if isinstance(batch, tuple) or isinstance(batch, list):
                    img = batch[0]
                else:
                    img = batch
                
                rec, loss = self.vqvae(img)
                
                imgs_and_recs = torch.stack((img, rec), dim=0)
                imgs_and_recs = rearrange(imgs_and_recs, 'r b ... -> (b r) ...')
                imgs_and_recs = imgs_and_recs.detach().cpu().float().clamp(0., 1.)
                
                grid = make_grid(imgs_and_recs, nrow=2, normalize=True, value_range=(0, 1))
                save_image(grid, os.path.join(self.image_saved_dir, f'step_{self.steps}_{i}.png'))
        self.vqvae.train()
        

class PaintMindTrainer(nn.Module):
    def __init__(
        self, 
        paintmind_version,
        dataset,
        num_epoch,
        vae_pretrained=None,
        valid_size=10,
        base_lr=3e-4,
        lr_min=3e-5,
        warmup_steps=5000,
        weight_decay=0.05,
        warmup_lr_init=1e-5,
        batch_size=32,
        grad_accum_steps=1,
        mixed_precision='fp16',
        max_grad_norm=1.0,
        save_every_n=10000,
        sample_every_n=1000,
        result_folder=None,
        log_dir="./log"
        ):
        super().__init__()
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            kwargs_handlers=[kwargs],
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=grad_accum_steps, 
            log_with="tensorboard",
            logging_dir=log_dir,
        )
        
        train_size = len(dataset) - valid_size
        self.train_ds, self.valid_ds = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))
        print(f"train dataset size: {train_size}, valid dataset size: {valid_size}")
        
        self.train_dl = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=batch_size, shuffle=False)
        
        clip_precision = 'fp16' if mixed_precision == 'fp16' else 'fp32'
        self.model = PaintMind(build_config(paintmind_version), vae_pretrained=vae_pretrained, clip_precision=clip_precision)
        
        self.optim = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=base_lr, weight_decay=weight_decay)
        self.scheduler = build_scheduler(self.optim, num_epoch, len(self.train_dl), lr_min, warmup_steps, warmup_lr_init)
        
        (
            self.model,
            self.optim,
            self.scheduler,
            self.train_dl,
            self.valid_dl,
        ) = self.accelerator.prepare(
            self.model,
            self.optim,
            self.scheduler,
            self.train_dl,
            self.valid_dl
        )
        
        self.num_epoch = num_epoch
        self.save_every_n = save_every_n
        self.max_grad_norm = max_grad_norm
        self.sample_every_n = sample_every_n
        
        self.model_saved_dir = os.path.join(result_folder, 'models')
        os.makedirs(self.model_saved_dir, exist_ok=True)
        
        self.image_saved_dir = os.path.join(result_folder, 'images')
        os.makedirs(self.image_saved_dir, exist_ok=True)
        
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'number of learnable parameters: {n_parameters//1e6}M')
         
    def save(self):
        self.accelerator.wait_for_everyone()
        state_dict = self.accelerator.unwrap_model(self.model).state_dict()
        self.accelerator.save(state_dict, os.path.join(self.model_saved_dir, f'paintmind_step_{self.steps}.pt'))
    
    def train(self):
        self.steps = 0
        self.accelerator.init_trackers("paintmind")
        log = Log()
        for epoch in range(self.num_epoch):
            self.model.train()
            with tqdm(self.train_dl, dynamic_ncols=True, disable=not self.accelerator.is_main_process) as train_dl:
                for batch in train_dl:
                    with self.accelerator.accumulate(self.model):
                        imgs, text = batch
                        
                        with self.accelerator.autocast():
                            loss = self.model(imgs, text, mask_ratio=cosine_masked_p_generator())                     

                        self.accelerator.backward(loss)
                        
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        
                        self.optim.step()
                        self.scheduler.step_update(self.steps)
                        self.optim.zero_grad()
                    
                    log.update({'loss':loss.item(), 'lr':self.optim.param_groups[0]['lr']})
   
                    train_dl.set_postfix(
                        ordered_dict={
                            "Epoch"      : epoch,
                            "Loss"       : log['loss'],
                            "LR"         : log['lr'],
                        }
                    )
                    self.accelerator.log({"loss": log['loss'], "lr": log['lr']}, step=self.steps)
                    
                    self.steps += 1
                    
                    if self.steps % self.sample_every_n == 0:
                        self.evaluate()
                        
                    if self.steps % self.save_every_n == 0:
                        self.save()
                        
        self.accelerator.end_training()        
        print("Train finished!")
        
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        with tqdm(self.valid_dl, dynamic_ncols=True, disable=not self.accelerator.is_main_process) as valid_dl:
            for i, batch in enumerate(valid_dl):
                imgs, text = batch

                with self.accelerator.autocast():
                    gens = self.model.generate(text=text, timesteps=18, temperature=1.0, save_interval=2)

                imgs_and_gens = [imgs.cpu()] + gens
                imgs_and_gens = torch.cat(imgs_and_gens, dim=0)
                imgs_and_gens = imgs_and_gens.detach().cpu().float().clamp(0., 1.)
                
                grid = make_grid(imgs_and_gens, nrow=2, normalize=True, value_range=(0, 1))
                save_image(grid, os.path.join(self.image_saved_dir, f'step_{self.steps}_{i}.png'))
        self.model.train()