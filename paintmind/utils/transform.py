import PIL
import torchvision.transforms as T

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def stage1_transform(img_size=256, is_train=True, p=0.8):
    resize = pair(int(img_size/p))
    t = []
    t.append(T.Resize(resize, interpolation=PIL.Image.BICUBIC))
    if is_train:
        t.append(T.RandomCrop(img_size))
        t.append(T.RandomHorizontalFlip(p=0.5))
    else:
        t.append(T.CenterCrop(img_size))
        
    t.append(T.ToTensor())
    t.append(T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))),
    
    return T.Compose(t)
        
def stage2_transform(img_size=256, is_train=True, p=0.8):
    resize = pair(int(img_size/p))
    t = []
    t.append(T.Resize(resize, interpolation=PIL.Image.BICUBIC))
    if is_train:
        t.append(T.RandomCrop(img_size))
    else:
        t.append(T.CenterCrop(img_size))
        
    t.append(T.ToTensor())
    t.append(T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))),
    
    return T.Compose(t)        