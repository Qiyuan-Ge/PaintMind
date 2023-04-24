# PaintMind
<div align=center>
<img src="https://github.com/Qiyuan-Ge/PaintMind/blob/main/assets/A_beautiful_girl_celebrating_her_birthday.png?raw=true" width="512">
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Framework-Pytorch-green?style=flat&logo=appveyor" alt="Badge 1">
  <img src="https://img.shields.io/badge/License-Apache--2.0-green?style=flat&logo=appveyor" alt="Badge 2">
  <img src="https://img.shields.io/badge/Contact-542801615@qq.com-green?style=flat&logo=appveyor" alt="Badge 3">
</div>

````
- 2023/4/19  
Note: Hi, I am preparing for a new release with better vitvqgan and generate model, pretrained weights 
from old version will not be available.
````

## Install
````
pip install git+https://github.com/Qiyuan-Ge/PaintMind.git
````

## Import
````
import paintmind as pm
````

## Reconstruction
Play with [Colab Notebook](https://colab.research.google.com/drive/1J8M97_HDAVXWQB4qp6yIBI7nPs-ZGXQz?usp=sharing).
### Usage
if you set 'pretrained=True', the code will then try to downlaod the pretrained weights.
````
import paintmind as pm

img = Image.open(img_path).convert('RGB')
img = pm.stage1_transform(is_train=False)(img)
# load pretrained vit-vqgan
model = pm.create_model(arch='vqgan', version='vit-s-vqgan', pretrained=True)
# encode image to latent
z, _, _ = model.encode(img.unsqueeze(0))
# decode latent to image
rec = model.decode(z).squeeze(0)
rec = torch.clamp(rec, -1., 1.)
````
You could also download the weights of the pretrained vit-vqgan to local from https://huggingface.co/RootYuan/vit-s-vqgan.  
To load the pretrained weights from local:
````
model = pm.create_model(arch='vqgan', version='vit-s-vqgan', pretrained=True, checkpoint_path='your/model/path')
````
### Training
````
import paintmind as pm
from paintmind.utils import datasets

data_path = 'your/data/path'
transform = pm.stage1_transform(img_size=256, is_train=True, scale=0.66)
dataset = datasets.ImageNet(root=data_path, transform=transform) # or your own dataset

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
````
### Performance
Below was the reconstruction ability of the vit-s-vqgan after training on 3M images with batchsize 16 and constant learning rate for 200000 steps. Because of limited time and computing resource, I only train the model for one eopch. The results was quite good, but the human face(especially the eyes) still need to be improved. By trying other techniques(warmup, cosine lr decay, larger batchsize, add more faces...). I'll release a better version in the future.
#### 1.
````
pm.reconstruction(img_path='https://cdn.pixabay.com/photo/2014/10/22/15/47/squirrel-498139_960_720.jpg')
````
<div align=center>
<img src="https://github.com/Qiyuan-Ge/PaintMind/blob/main/assets/rec_1.png?raw=true">
</div>

#### 2.
````
pm.reconstruction(img_path='https://cdn.pixabay.com/photo/2017/04/09/10/44/sea-shells-2215408_960_720.jpg')
````
<div align=center>
<img src="https://github.com/Qiyuan-Ge/PaintMind/blob/main/assets/rec_2.png?raw=true">
</div>

#### 3.
````
pm.reconstruction(img_path='https://cdn.pixabay.com/photo/2015/06/19/21/24/avenue-815297_960_720.jpg')
````
<div align=center>
<img src="https://github.com/Qiyuan-Ge/PaintMind/blob/main/assets/rec_3.png?raw=true">
</div>

#### 4.
````
pm.reconstruction(img_path='https://cdn.pixabay.com/photo/2017/03/30/18/17/girl-2189247_960_720.jpg')
````
<div align=center>
<img src="https://github.com/Qiyuan-Ge/PaintMind/blob/main/assets/rec_4.png?raw=true">
</div>

#### 5.
````
pm.reconstruction(img_path='https://cdn.pixabay.com/photo/2017/10/28/07/47/woman-2896389_960_720.jpg')
````
<div align=center>
<img src="https://github.com/Qiyuan-Ge/PaintMind/blob/main/assets/rec_5.png?raw=true">
</div>

## Text2Image
Not finish yet~~
### Training
````
import paintmind as pm
from paintmind.utils import datasets

data_path = 'your/data/path'
transform = pm.stage2_transform(img_size=256, is_train=True, scale=0.8)
dataset = datasets.CoCo(root=data_path, transform=transform) # or your own dataset, the output format should be (image, caption)

# load pretrained weights I upload to huggingface, not finish yet
model = pm.create_pipeline_for_train(version='paintmindv1', stage1_pretrained=True)
# or load your pretrained weights
model = pm.create_pipeline_for_train(version='paintmindv1', stage1_pretrained=True, stage1_checkpoint_path='your/pretrained/vitvqgan')

trainer = pm.PaintMindTrainer(
    model                       = model,
    dataset                     = dataset,
    num_epoch                   = 40,
    valid_size                  = 64,
    optim                       = 'adamw',
    lr                          = 1e-4,
    lr_min                      = 1e-5,
    warmup_steps                = 10000,
    weight_decay                = 0.05,
    warmup_lr_init              = 1e-6,
    decay_steps                 = 80000,
    batch_size                  = 16,
    num_workers                 = 2,
    pin_memory                  = True,
    grad_accum_steps            = 8,
    mixed_precision             = 'bf16',
    max_grad_norm               = 1.0,
    save_every                  = 5000,
    sample_every                = 5000,
    result_folder               = "your/result/folder",
    log_dir                     = "your/log/dir",
    )
trainer.train()
````
