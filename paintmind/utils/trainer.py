import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.autograd import Variable
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
from lpips import LPIPS
from einops import rearrange
from paintmind.optim import Lion
from paintmind.utils.lr_scheduler import build_scheduler
from paintmind.stage1.discriminator import NLayerDiscriminator


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def hinge_d_loss(fake, real):
    loss_fake = torch.mean(F.relu(1. + fake))
    loss_real = torch.mean(F.relu(1. - real))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def g_nonsaturating_loss(fake):
    loss = F.softplus(-fake).mean()

    return loss


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
        vqvae,
        dataset,
        num_epoch,
        valid_size=32,
        lr=1e-4,
        lr_min=5e-5, 
        warmup_steps=50000, 
        warmup_lr_init=1e-6,
        decay_steps=None,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        max_grad_norm=1.0,
        grad_accum_steps=1,
        mixed_precision='bf16',
        save_every=10000,
        sample_every=1000,
        result_folder=None,
        log_dir="./log",
    ):
        super().__init__()
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=grad_accum_steps, 
            log_with="tensorboard",
            logging_dir=log_dir,
        )

        self.vqvae = vqvae
        
        self.discr = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3)
        
        train_size = len(dataset) - valid_size
        self.train_ds, self.valid_ds = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))
        print(f"train dataset size: {train_size}, valid dataset size: {valid_size}")
        
        self.train_dl = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        
        self.g_optim = Adam(self.vqvae.parameters(), lr=lr, betas=(0.9, 0.99))
        self.d_optim = Adam(self.discr.parameters(), lr=lr, betas=(0.9, 0.99))
        self.g_sched = build_scheduler(self.g_optim, num_epoch, len(self.train_dl), lr_min, warmup_steps, warmup_lr_init, decay_steps)
        self.d_sched = build_scheduler(self.d_optim, num_epoch, len(self.train_dl), lr_min, warmup_steps, warmup_lr_init, decay_steps)
        
        self.per_loss = LPIPS(net='vgg').to(self.device).eval()
        for param in self.per_loss.parameters():
            param.requires_grad = False
        self.d_loss = hinge_d_loss
        self.g_loss = g_nonsaturating_loss
        self.d_weight = 0.1
        
        (
            self.vqvae,
            self.discr,
            self.g_optim,
            self.d_optim,
            self.g_sched,
            self.d_sched,
            self.train_dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.vqvae,
            self.discr,
            self.g_optim,
            self.d_optim,
            self.g_sched,
            self.d_sched,
            self.train_dl,
            self.valid_dl
        )
        
        self.num_epoch = num_epoch
        self.save_every = save_every
        self.samp_every = sample_every
        self.max_grad_norm = max_grad_norm
        
        self.model_saved_dir = os.path.join(result_folder, 'models')
        os.makedirs(self.model_saved_dir, exist_ok=True)
        
        self.image_saved_dir = os.path.join(result_folder, 'images')
        os.makedirs(self.image_saved_dir, exist_ok=True)
        
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'number of learnable parameters: {n_parameters//1e6}M')
    
    @property
    def device(self):
        return self.accelerator.device
    
    def calculate_gradient_penalty(self, real_images, fake_images, lambda_term=10):
        eta = torch.FloatTensor(real_images.shape[0],1,1,1).uniform_(0,1).to(self.device)
        eta = eta.expand(real_images.shape[0], real_images.size(1), real_images.size(2), real_images.size(3))
        
        interpolated = eta * real_images + ((1 - eta) * fake_images)
        interpolated = Variable(interpolated, requires_grad=True)
        prob_interpolated = self.discr(interpolated)
        
        gradients = torch.autograd.grad(
            outputs=prob_interpolated, 
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
            create_graph=True, 
            retain_graph=True,)[0]
        
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
        return grad_penalty
    
    def train(self):
        self.steps = 0
        self.accelerator.init_trackers("vqgan")
        self.log = Log()
        for epoch in range(self.num_epoch):
            with tqdm(self.train_dl, dynamic_ncols=True, disable=not self.accelerator.is_main_process) as train_dl:
                for batch in train_dl:
                    if isinstance(batch, tuple) or isinstance(batch, list):
                        img = batch[0]
                    else:
                        img = batch
                    
                    # discriminator part
                    requires_grad(self.vqvae, False)
                    requires_grad(self.discr, True)
                    with self.accelerator.accumulate(self.discr):
                        with self.accelerator.autocast():
                            rec, codebook_loss = self.vqvae(img)
        
                            fake_pred = self.discr(rec)
                            real_pred = self.discr(img)
                            
                            gp = self.calculate_gradient_penalty(img, rec)
                            d_loss = self.d_loss(fake_pred, real_pred) + gp
                            
                        self.accelerator.backward(d_loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.discr.parameters(), self.max_grad_norm)
                        self.d_optim.step()
                        self.d_sched.step_update(self.steps)
                        self.d_optim.zero_grad()
                        
                        self.log.update({'d loss':d_loss.item(), 'd lr':self.d_optim.param_groups[0]['lr']})
                        
                    # generator part
                    requires_grad(self.vqvae, True)
                    requires_grad(self.discr, False)
                    with self.accelerator.accumulate(self.vqvae):
                        with self.accelerator.autocast():
                            rec, codebook_loss = self.vqvae(img)
                            # reconstruction loss
                            rec_loss = F.l1_loss(rec, img) + F.mse_loss(rec, img)
                            # perception loss
                            per_loss = self.per_loss(rec, img).mean()
                            # gan loss
                            g_loss = self.g_loss(self.discr(rec))
                            # combine
                            loss = codebook_loss + rec_loss + per_loss + self.d_weight * g_loss
                        
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.vqvae.parameters(), self.max_grad_norm)
                        self.g_optim.step()
                        self.g_sched.step_update(self.steps)
                        self.g_optim.zero_grad()   

                    self.steps += 1
                    self.log.update({'rec loss':rec_loss.item(), 'per loss':per_loss.item(), 'g loss':g_loss.item(), 'g lr':self.g_optim.param_groups[0]['lr']})
                    
                    if not (self.steps % self.save_every):
                        self.save()
                    
                    if not (self.steps % self.samp_every):
                        self.evaluate()
   
                    train_dl.set_postfix(
                        ordered_dict={
                            "epoch"               : epoch,
                            "reconstruct loss"    : self.log['rec loss'],
                            "perceptual loss"     : self.log['per loss'],
                            "g_loss"              : self.log['g loss'],
                            "d_loss"              : self.log['d loss'],
                            "g_lr"                : self.log['g lr'],
                        }
                    )
                    self.accelerator.log(
                        {
                            "reconstruct loss"    : self.log['rec loss'], 
                            "perceptual loss"     : self.log['per loss'],
                            "g_loss"              : self.log['g loss'],
                            "d_loss"              : self.log['d loss'],
                            "g_lr"                : self.log['g lr'],
                            "d_lr"                : self.log['d lr'],
                        }, 
                        step=self.steps
                    )
        
        self.accelerator.end_training()        
        print("Train finished!")
                    
    def save(self):
        self.accelerator.wait_for_everyone()
        state_dict = self.accelerator.unwrap_model(self.vqvae).state_dict()
        self.accelerator.save(state_dict, os.path.join(self.model_saved_dir, f'vit_vq_step_{self.steps}.pt'))
                                                       
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
                
                grid = make_grid(imgs_and_recs, nrow=6, normalize=True, value_range=(-1, 1))
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
        optim='lion', # or 'adamw'
        lr=6e-5,
        lr_min=1e-5,
        warmup_steps=5000,
        warmup_lr_init=1e-6,
        decay_steps=80000,
        weight_decay=0.05,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        grad_accum_steps=1,
        mixed_precision='fp16',
        max_grad_norm=1.0,
        save_every=10000,
        sample_every=1000,
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
        
        self.train_dl = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=6, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        
        self.model = model
        
        if optim == 'lion':
            self.optim = Lion([p for p in self.model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
        elif optim == 'adamw':
            self.optim = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=lr, betas=(0.9, 0.96), weight_decay=weight_decay)
        else:
            raise NotImplementedError
        
        self.scheduler = build_scheduler(self.optim, num_epoch, len(self.train_dl), lr_min, warmup_steps, warmup_lr_init, decay_steps)
        
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
        self.save_every = save_every
        self.sample_every = sample_every
        self.max_grad_norm = max_grad_norm
        
        self.model_saved_dir = os.path.join(result_folder, 'models')
        os.makedirs(self.model_saved_dir, exist_ok=True)
        
        self.image_saved_dir = os.path.join(result_folder, 'images')
        os.makedirs(self.image_saved_dir, exist_ok=True)
        
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'number of learnable parameters: {n_parameters//1e6}M')
        print(f"train dataset size: {train_size}, valid dataset size: {valid_size}")
         
    def save(self):
        self.accelerator.wait_for_everyone()
        state_dict = self.accelerator.unwrap_model(self.model).state_dict()
        self.accelerator.save(state_dict, os.path.join(self.model_saved_dir, f'paintmind_step_{self.steps}.pt'))
    
    def train(self):
        self.steps = 0
        self.cfg_p = 0.1
        self.accelerator.init_trackers("paintmind")
        self.log = Log()
        for epoch in range(self.num_epoch):
            with tqdm(self.train_dl, dynamic_ncols=True, disable=not self.accelerator.is_main_process) as train_dl:
                for batch in train_dl:
                    with self.accelerator.accumulate(self.model):
                        imgs, text = batch
                        if random.random() < self.cfg_p:
                            text = None
                        
                        with self.accelerator.autocast():
                            loss = self.model(imgs, text, mask_ratio=masked_p_generator())                     
                        
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optim.step()
                        self.scheduler.step_update(self.steps)
                        self.optim.zero_grad()
                    
                    self.steps += 1
                    self.log.update({'loss':loss.item(), 'lr':self.optim.param_groups[0]['lr']})
                    
                    if not (self.steps % self.sample_every):
                        self.evaluate()
                    
                    if not (self.steps % self.save_every):    
                        self.save()
                    
                    train_dl.set_postfix(
                        ordered_dict={
                            "Epoch"      : epoch,
                            "Loss"       : self.log['loss'],
                            "LR"         : self.log['lr'],
                        }
                    )
                    self.accelerator.log({"loss": self.log['loss'], "lr": self.log['lr']}, step=self.steps)
                        
        self.accelerator.end_training()        
        print("Train finished!")
        
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        with tqdm(self.valid_dl, dynamic_ncols=True, disable=not self.accelerator.is_main_process) as valid_dl:
            for i, batch in enumerate(valid_dl):
                imgs, text = batch

                with self.accelerator.autocast():
                    gens = self.model.generate(text=text, timesteps=18, temperature=1.0, topk=5, save_interval=2)

                imgs_and_gens = [imgs.cpu()] + gens
                imgs_and_gens = torch.cat(imgs_and_gens, dim=0)
                imgs_and_gens = imgs_and_gens.detach().cpu().float()
                
                grid = make_grid(imgs_and_gens, nrow=6, normalize=True, value_range=(-1, 1))
                save_image(grid, os.path.join(self.image_saved_dir, f'step_{self.steps}_{i}.png'))
        self.model.train()
