import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
from einops import rearrange
from paintmind.stage1 import Discriminator
from paintmind.utils.lr_scheduler import build_scheduler
from paintmind.utils.loss import hinge_g_loss, hinge_d_loss, PerceptualLoss


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


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
        vae,
        dataset,
        num_epoch,
        valid_size=32,
        base_lr=4.5e-6,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        max_grad_norm=1.0,
        grad_accum_steps=1,
        mixed_precision='fp16',
        save_every=10000,
        sample_every=1000,
        result_folder=None,
        discriminator_iter_start=10000, # 250001 in taming
        log_dir="./log",
    ):
        super().__init__()
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=grad_accum_steps, 
            log_with="tensorboard",
            logging_dir=log_dir,
        )

        self.vqvae = vae
        
        self.discr = self.build_discr()
        
        train_size = len(dataset) - valid_size
        self.train_ds, self.valid_ds = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))
        print(f"train dataset size: {train_size}, valid dataset size: {valid_size}")
        
        self.train_dl = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        
        n_gpu = self.accelerator.num_processes
        lr = base_lr * batch_size * grad_accum_steps * n_gpu
        self.optim = torch.optim.Adam(self.vqvae.parameters(), lr=lr, betas=(0.5, 0.9))
        self.discr_optim = torch.optim.Adam(self.discr.parameters(), lr=lr, betas=(0.5, 0.9))
        print(f"Setting learning rate to {lr} = {base_lr}(base_lr) * {batch_size}(bs) * {grad_accum_steps}(grad_accum_steps) * {n_gpu}(n_gpu)")
        
        self.rec_loss = F.l1_loss
        self.gen_loss = hinge_g_loss
        self.dis_loss = hinge_d_loss
        self.per_loss = PerceptualLoss(device=self.device)
        
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
        self.save_every = save_every
        self.sample_every = sample_every
        self.max_grad_norm = max_grad_norm
        self.discr_weight = 0.8
        self.discr_factor = 1.0
        self.discr_iter_start = discriminator_iter_start
        
        self.model_saved_dir = os.path.join(result_folder, 'models')
        os.makedirs(self.model_saved_dir, exist_ok=True)
        
        self.image_saved_dir = os.path.join(result_folder, 'images')
        os.makedirs(self.image_saved_dir, exist_ok=True)
        
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'number of learnable parameters: {n_parameters//1e6}M')
    
    @property
    def device(self):
        return self.accelerator.device
    
    def build_discr(self):
        discr_layers = 4
        dim=64
        layer_mults = list(map(lambda t: 2 ** t, range(discr_layers)))
        layer_dims = [dim * mult for mult in layer_mults]
        dims = (dim, *layer_dims) 
        discr = Discriminator(dims=dims, channels=self.vqvae.channels)
        return discr
    
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discr_weight
        return d_weight
    
    def generator_update(self, img):
        with self.accelerator.accumulate(self.vqvae):
            with self.accelerator.autocast():
                rec, codebook_loss = self.vqvae(img)                       
                # reconstruction loss
                rec_loss = self.rec_loss(rec, img)
                # perceptual loss
                per_loss = self.per_loss(rec, img)
                # generator loss
                gen_loss = self.gen_loss(self.discr(rec))
                # combine
                nll_loss = rec_loss + per_loss
                d_weight = self.calculate_adaptive_weight(nll_loss, gen_loss, last_layer=self.vqvae.get_last_layer())
                discr_factor = adopt_weight(self.discr_factor, self.steps, threshold=self.discr_iter_start)
                loss = codebook_loss + nll_loss + d_weight * discr_factor * gen_loss
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.vqvae.parameters(), self.max_grad_norm)
            self.optim.step()
            self.optim.zero_grad()
            
            self.log.update(
                {
                    'rec loss' : rec_loss.item(),
                    'per loss' : per_loss.item(), 
                    'gen loss' : gen_loss.item(),
                }
            )
        
    def discriminator_update(self, img):
        with self.accelerator.accumulate(self.discr):
            with self.accelerator.autocast():
                rec, _ = self.vqvae(img)
                logits_fake = self.discr(rec.detach())
                logits_real = self.discr(img.detach())
                discr_factor = adopt_weight(self.discr_factor, self.steps, threshold=self.discr_iter_start)
                d_loss = discr_factor * self.dis_loss(logits_fake, logits_real)
            self.accelerator.backward(d_loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.discr.parameters(), self.max_grad_norm)
            self.discr_optim.step()
            self.discr_optim.zero_grad()
            
            self.log.update({'dis loss' : d_loss.item()})
        
    
    def train(self):
        self.steps = 0
        self.accelerator.init_trackers("vqgan")
        self.log = Log()
        for epoch in range(self.num_epoch):
            self.vqvae.train()
            with tqdm(self.train_dl, dynamic_ncols=True, disable=not self.accelerator.is_main_process) as train_dl:
                for batch in train_dl:
                    if isinstance(batch, tuple) or isinstance(batch, list):
                        img = batch[0]
                    else:
                        img = batch
                    
                    self.generator_update(img)
                    
                    self.discriminator_update(img)
                    
                    self.steps += 1
                    
                    if not (self.steps % self.save_every):
                        self.save()
                    
                    if not (self.steps % self.sample_every):
                        self.evaluate()
   
                    train_dl.set_postfix(
                        ordered_dict={
                            "epoch"               : epoch,
                            "reconstruction loss" : self.log['rec loss'],
                            "perceptual loss"     : self.log['per loss'],
                            "gen loss"            : self.log['gen loss'],
                            "discr loss"          : self.log['dis loss'],
                        }
                    )
                    self.accelerator.log(
                        {
                            "reconstruction loss" : self.log['rec loss'], 
                            "perceptual loss"     : self.log['per loss'],
                            "gen loss"            : self.log['gen loss'],
                            "discr loss"          : self.log['dis loss'],
                        }, 
                        step=self.steps
                    )
        
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
                
                rec, _ = self.vqvae(img)
                
                imgs_and_recs = torch.stack((img, rec), dim=0)
                imgs_and_recs = rearrange(imgs_and_recs, 'r b ... -> (b r) ...')
                imgs_and_recs = imgs_and_recs.detach().cpu().float()
                
                grid = make_grid(imgs_and_recs, nrow=4, normalize=True)
                save_image(grid, os.path.join(self.image_saved_dir, f'step_{self.steps}_{i}.png'))
        self.vqvae.train()


def masked_p_generator():
    p = np.cos(0.5 * np.pi * np.random.rand(1))
    return p.item()
      

class PaintMindTrainer(nn.Module):
    def __init__(
        self, 
        model,
        dataset,
        num_epoch,
        valid_size=10,
        base_lr=3e-4,
        lr_min=3e-5,
        warmup_steps=5000,
        weight_decay=0.05,
        warmup_lr_init=1e-5,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
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
        
        self.train_dl = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        
        self.model = model
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
                            loss = self.model(imgs, text, mask_ratio=masked_p_generator())                     

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
                imgs_and_gens = imgs_and_gens.detach().cpu().float().clamp(-1., 1.)
                
                grid = make_grid(imgs_and_gens, nrow=4, normalize=True, value_range=(-1, 1))
                save_image(grid, os.path.join(self.image_saved_dir, f'step_{self.steps}_{i}.png'))
        self.model.train()
