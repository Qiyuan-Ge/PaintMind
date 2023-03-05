import os
import torch
import zipfile
import numpy as np
from PIL import Image
from datasets import load_dataset
from pycocotools.coco import COCO


# def load_dataset(name='coco', root=None, transform=None):
#     if name == 'coco':
#         return coco(root, transform=transform)


def unzip_file(zip_src, tgt_dir):
    if zipfile.is_zipfile(zip_src):
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, tgt_dir)       
    else:
        raise RuntimeError("This is not zip file.")

class LaionCoco:
    def __init__(self, parquet_folder, image_folder=None, transform=None, download=True):
        self.image_folder = image_folder
        df = pd.DataFrame()
        for name in os.listdir(parquet_folder):
            if name.endswith('.parquet'):
                df = pd.concat([df, pd.read_parquet(os.path.join(parquet_folder, name))])
        
        df = df.reset_index(drop=True)
        df['img_name'] = df.index
        df['img_name'] = df['img_name'].apply(lambda x: str(x)+'.png')
        self.df = df
        
        if download:
            self.download_drom_url()
            
        self.transform = transform
  
    def download_drom_url(self):
        cnt = 0
        for i in tqdm(self.df.index):
            url = self.df.loc[i, 'URL']
            img_name = self.df.loc[i, 'img_name']
            try:
                chunks = requests.get(url, stream=True)
                with open(f'{self.image_folder}/{img_name}', 'wb') as f:
                    for c in chunks:
                        f.write(c)
            except:
                pass
    
    def __getitem__(self, idx):
        
    def __len__(self): 
        
        
class DiffusionDB:
    def __init__(self, version='large_random_100k', transform=None):
        self.dataset = load_dataset("poloclub/diffusiondb", version)['train']
        self.transform = transform
        
    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        image = data['image']
        prompt = data['prompt']
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, prompt
        
    def __len__(self):
        return len(self.data)        

    
class CoCo:
    def __init__(self, root, dataType='train2017', annType='captions', transform=None):
        self.root = root
        self.img_dir = '{}/{}'.format(root, dataType)
        annFile = '{}/annotations/{}_{}.json'.format(root, annType, dataType)
        self.coco = COCO(annFile)
        self.imgids = self.coco.getImgIds()
        self.transform = transform
    
    def __getitem__(self, idx):
        imgid = self.imgids[idx]
        img_name = self.coco.loadImgs(imgid)[0]['file_name']
        annid = self.coco.getAnnIds(imgIds=imgid)
        img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        ann = np.random.choice(self.coco.loadAnns(annid))['caption']
        if self.transform is not None:
            img = self.transform(img)
        
        return img, ann     
        
    def __len__(self):
        return len(self.imgids)


class CelebA:
    def __init__(self, root, type='identity', transform=None):
        """CelebA Dataset http://personal.ie.cuhk.edu.hk/~lz013/projects/CelebA.html
        Args:
            root (str): CelebA Dataset folder path
            type (str, optional): 'identity' or 'attr'. Defaults to 'identity'.
            transform (torchvision.transforms, optional): torchvision.transforms. Defaults to None.
        """        
        ann_dir = os.path.join(root, 'Anno')
        base_dir = os.path.join(root, 'Img')
        zfile_path = os.path.join(base_dir, 'img_align_celeba.zip')
        self.img_dir = os.path.join(base_dir, 'img_align_celeba')
        if os.path.exists(self.img_dir):
            pass
        elif os.path.exists(zfile_path):
            unzip_file(zfile_path, base_dir)
        else:
            raise RuntimeError("Dataset not found.")
        self.imgs = os.listdir(self.img_dir)
        if type == 'identity':
            self.img2id = {}
            with open(os.path.join(ann_dir, 'identity_CelebA.txt'), 'r') as f:
                for line in f.readlines():
                    name, id = line.strip().split(' ')
                    self.img2id[name] = int(id)
        self.transform = transform
        
    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        ann = self.img2id[img_name]
        if self.transform is not None:
            img = self.transform(img)
        
        ann = torch.tensor(ann)
            
        return img, ann
    
    def __len__(self):
        return len(self.imgs)
