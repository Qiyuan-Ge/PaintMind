import paintmind as pm
from paintmind.utils import datasets

data_path = '/home/pranoy/datasets/coco2017'
transform = pm.stage1_transform(img_size=256, is_train=True, scale=0.66)
dataset = datasets.CoCo(root=data_path, transform=transform)
# or your own dataset, the output format should be image: torch.Tensor or (image: torch.Tensor, _)

model = pm.create_model(arch='vqgan', version='vit-s-vqgan', pretrained=False)

trainer = pm.VQGANTrainer(
    vqvae                    = model,
    dataset                  = dataset,
    num_epoch                = 100,
    valid_size               = 64,
    lr                       = 1e-4,
    lr_min                   = 5e-5,
    warmup_steps             = 50000,
    warmup_lr_init           = 1e-6,
    decay_steps              = 100000,
    batch_size               = 16,
    num_workers              = 2,
    pin_memory               = True,
    grad_accum_steps         = 8,
    mixed_precision          = 'bf16',
    max_grad_norm            = 1.0,
    save_every               = 5000,
    sample_every             = 5000,
    result_folder            = "your/result/folder",
    log_dir                  = "your/log/dir",
)
trainer.train()