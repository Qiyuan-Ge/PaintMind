import PIL
import torchvision.transforms as T
#from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def create_transform(resize=320, crop_size=256, crop=True, is_train=False):
    resize = pair(resize)
    crop_size = pair(crop_size)
    t = []
    t.append(T.Resize(resize, interpolation=PIL.Image.BICUBIC))
    if crop:
        if is_train:
            t.append(T.RandomCrop(crop_size))
        else:
            t.append(T.CenterCrop(crop_size))
    t.append(T.ToTensor())
    t.append(T.Lambda(lambda t: (t * 2) - 1))
    
    return T.Compose(t)