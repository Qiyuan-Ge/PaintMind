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
    x = x.permute(1,2,0).detach().cpu().numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    return x

def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))

def reconstruction(img_path=None, model_name='vit-s-vqgan', titles=['origin', 'reconstruct'], checkpoint_path=None, scale=0.8, device='cuda'):
    w, h = 256, 256
    
    if img_path.startswith('http'):
        img = download_image(img_path)
    else:
        img = Image.open(img_path).convert('RGB')
    
    img = pm.stage1_transform(is_train=False, scale=scale)(img)    
    img = img.to(device)
    model = pm.create_model(arch='vqgan', version=model_name, pretrained=True, checkpoint_path=checkpoint_path)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        z, _, _ = model.encode(img.unsqueeze(0))
        rec = model.decode(z).squeeze(0)
    img = restore(img)
    rec = restore(rec)
    
    fig = Image.new("RGB", (2*w, h))
    fig.paste(img, (0,0))
    fig.paste(rec, (1*w,0))
    try:
        font = ImageFont.truetype('arialbi.ttf', 16)
    except:
        font = None
    for i, title in enumerate(titles):
        ImageDraw.Draw(fig).text((i*w, 0), f'{title}', (255, 255, 255), font=font)
        
    return fig
