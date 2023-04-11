import json
from copy import deepcopy
from .stage1 import VQModel
from .pipeline import Pipeline


class Config:
    def __init__(self, config=None):
        if config is not None:
            self.from_dict(config)
    
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
        self.clear()
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
        

vit_s_vqgan_config = {
    'n_embed'     :8192,
    'embed_dim'   :32,
    'beta'        :0.25,
    'encdec':{
        'image_size':256, 
        'patch_size':8, 
        'dim':512, 
        'depth':8, 
        'heads':8, 
        'mlp_dim':2048, 
        'in_channels':3, 
        'dim_head':64, 
        'dropout':0.0,
    }, 
}


vit_b_vqgan_config = {
    'n_embed'     :8192,
    'embed_dim'   :32,
    'beta'        :0.25,
    'encdec':{
        'image_size':256, 
        'patch_size':8, 
        'dim':768, 
        'depth':12, 
        'heads':12, 
        'mlp_dim':3072, 
        'in_channels':3, 
        'out_channels':3,
        'dim_head':64, 
        'dropout':0.1,
    }, 
}


pipeline_v1_config = {
    'vae'         :vit_s_vqgan_config,
    'dim'         :768, 
    'dim_context' :768, 
    'dim_head'    :64,
    'mlp_dim'     :3072,
    'num_head'    :12, 
    'depth'       :8, 
    'dropout'     :0.1, 
}


ver2cfg = {
    'vit-s-vqgan'  : vit_s_vqgan_config,
    'vit-b-vqgan'  : vit_b_vqgan_config,
    'pipeline-v1'  : pipeline_v1_config,
}


def create_model(arch='pipeline', version='pipeline_v1', pretrained=None, stage1_pretrained=None):
    config = Config(config=ver2cfg[version])
    if arch == 'vqgan':
        model = VQModel(config)
    elif arch == 'pipeline':
        model = Pipeline(config, vae_pretrained=stage1_pretrained)
    else:
        raise ValueError(f"failed to load arch named {arch}")
        
    if pretrained is not None:
        model.from_pretrained(pretrained)
        
    return model
        
        
    
