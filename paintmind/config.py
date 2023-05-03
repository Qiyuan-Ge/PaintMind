import json
from copy import deepcopy

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
    'enc':{
        'image_size':256, 
        'patch_size':8, 
        'dim':512, 
        'depth':8, 
        'num_head':8, 
        'mlp_dim':2048, 
        'in_channels':3, 
        'dim_head':64, 
        'dropout':0.0,
    }, 
    'dec':{
        'image_size':256, 
        'patch_size':8, 
        'dim':512, 
        'depth':8, 
        'num_head':8, 
        'mlp_dim':2048, 
        'out_channels':3, 
        'dim_head':64, 
        'dropout':0.0,
    },     
}

pipeline_v1_config = {
    'stage1'         :'vit-s-vqgan',
    't5'             :'t5-l',
    'dim'            :1024, 
    'dim_head'       :64,
    'mlp_dim'        :4096,
    'num_head'       :16, 
    'depth'          :12,
    'dropout'        :0.1, 
}

ver2cfg = {
    'vit-s-vqgan'  : vit_s_vqgan_config,
    'paintmindv1'  : pipeline_v1_config,
}
        
