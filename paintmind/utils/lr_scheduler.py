from timm.scheduler.cosine_lr import CosineLRScheduler

def build_scheduler(optimizer, n_epoch, n_iter_per_epoch, lr_min, warmup_steps, warmup_lr_init, decay_steps=None):
    if decay_steps is None:
        decay_steps = n_epoch * n_iter_per_epoch
    
    scheduler = CosineLRScheduler(optimizer, t_initial=decay_steps, lr_min=lr_min, warmup_t=warmup_steps, warmup_lr_init=warmup_lr_init, 
                                  cycle_limit=1, t_in_epochs=False, warmup_prefix=True)
    
    return scheduler
