import json
from copy import deepcopy
from paintmind.stage1.vqvae import VQVAE
from paintmind.stage2.transformer import PaintMind


class Config:
    def __init__(self):
        self.n_embed = 8192
        self.embed_dim = 16
    
    def __repr__(self):
        return str(self.to_json_string())
    
    def to_dict(self):
        return deepcopy(self.__dict__)
    
    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2)
            
    def from_dict(self, dct):
        for key, value in dct.items():
            self.__dict__[key] = value
            
        return self.to_dict()
    
    def from_json(self, json_path):
        with open(json_path, 'r') as f:
            config = json.load(f)
            self.from_dict(config)
            
        return self.to_dict()
    
    def clear(self):
        del self.__dict__
        

vit_s_vqvae_config = {
    'n_embed'     :8192,
    'embed_dim'   :16,
    'beta'        :0.25,
    'image_size'  :256, 
    'patch_size'  :8,
    'dim'         :512,
    'depth'       :8,
    'heads'       :8,
    'mlp_dim'     :2048,
    'channels'    :3,
    'dim_head'    :64, 
    'dropout'     :0.1, 
    'emb_dropout' :0.1
}


vit_b_vqvae_config = {
    'n_embed'     :8192,
    'embed_dim'   :16,
    'beta'        :0.25,
    'image_size'  :256, 
    'patch_size'  :8,
    'dim'         :768,
    'depth'       :12,
    'heads'       :12,
    'mlp_dim'     :3072,
    'channels'    :3,
    'dim_head'    :64, 
    'dropout'     :0.1, 
    'emb_dropout' :0.1
}

paintmind_v1_config = {
    'vae_name'    :'vit_small_vqvae',
    'dim'         :768, 
    'dim_context' :1024, 
    'dim_head'    :64,
    'mlp_dim'     :3072,
    'num_head'    :12, 
    'depth'       :6, 
    'dropout'     :0.1, 
}


name2config = {
    'vit_small_vqvae' : vit_s_vqvae_config,
    'vit_base_vqvae'  : vit_b_vqvae_config,
    'paintmind_v1'    : paintmind_v1_config,
}


def build_config(name):
    config = Config()
    config.from_dict(name2config[name])
    
    return config
 
   
def build_model(name, pretrained=None):
    config = build_config(name)

    if name == 'vit_small_vqvae':
        model = VQVAE(config) 
    elif name == 'vit_base_vqvae':
        model = VQVAE(config) 
    elif name == 'paintmind_v1':
        model = PaintMind(config)
    
    if pretrained is not None:
        model.from_pretrained(pretrained)
    
    return model