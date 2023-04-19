import PIL
import torchvision.transforms as T

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def stage1_transform(img_size=256, is_train=True, scale=0.8):
    resize = pair(int(img_size/scale))
    t = []
    t.append(T.Resize(resize, interpolation=PIL.Image.BICUBIC))
    if is_train:
        t.append(T.RandomCrop(img_size))
        t.append(T.RandomHorizontalFlip(p=0.5))
    else:
        t.append(T.CenterCrop(img_size))
        
    t.append(T.ToTensor())
    t.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))),
    
    return T.Compose(t)
        
def stage2_transform(img_size=256, is_train=True, scale=0.8):
    resize = pair(int(img_size/scale))
    t = []
    t.append(T.Resize(resize, interpolation=PIL.Image.BICUBIC))
    if is_train:
        t.append(T.RandomCrop(img_size))
    else:
        t.append(T.CenterCrop(img_size))
        
    t.append(T.ToTensor())
    t.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))),
    
    return T.Compose(t)        
