import io
import torch
import requests
import numpy as np
import paintmind as pm
from PIL import Image, ImageDraw, ImageFont

def exists(x):
    return x is not None

def restore(x):
    x = (x + 1) * 0.5
    x = x.permute(1,2,0).detach().numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    return x

def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))

def reconstuction(img_path=None, img_url=None, model_args=None, titles=['origin', 'reconstruct'], pretrained='./vit_vq_step_80000.pt', scale=0.8):
    w, h = 256, 256
    if exists(img_path):
        img = Image.open(img_path).convert('RGB')
    elif exists(img_url):
        img = download_image(img_url)
    img = pm.stage1_transform(is_train=False, p=scale)(img)    
    
    if not exists(model_args):
        model_args = {'arch':'vqgan', 'version':'vit_s_vqgan'}
    model = pm.create_model(arch=model_args['arch'], version=model_args['version'], pretrained=pretrained)
    
    z, _, _ = model.encode(img.unsqueeze(0))
    rec = model.decode(z).squeeze(0)
    rec = torch.clamp(rec, -1., 1.)
    img = restore(img)
    rec = restore(rec)
    
    fig = Image.new("RGB", (2*w, h))
    fig.paste(img, (0,0))
    fig.paste(rec, (1*w,0))
    font = ImageFont.truetype('arialbi.ttf', 16)
    for i, title in enumerate(titles):
        ImageDraw.Draw(fig).text((i*w, 0), f'{title}', (255, 255, 255), font=font)
        
    return fig
