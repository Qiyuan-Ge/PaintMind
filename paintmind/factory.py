import paintmind as pm
from .stage1 import VQModel
from .pipeline import Pipeline
from huggingface_hub import hf_hub_download

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
    'stage1'         :'vit-s-vqgan',
    'dim'            :768, 
    'dim_context'    :1024, 
    'dim_head'       :64,
    'mlp_dim'        :3072,
    'num_head'       :12, 
    'depth'          :8, 
    'dropout'        :0.1, 
}

ver2cfg = {
    'vit-s-vqgan'  : vit_s_vqgan_config,
    'vit-b-vqgan'  : vit_b_vqgan_config,
    'pipeline-v1'  : pipeline_v1_config,
}


def create_model(arch='pipeline', version='pipeline-v1', pretrained=True, checkpoint_path=None):
    config = pm.Config(config=ver2cfg[version])
    
    if arch == 'vqgan':
        model = VQModel(config)
    elif arch == 'pipeline':
        model = Pipeline(config)
    else:
        raise ValueError(f"failed to load arch named {arch}")
        
    if pretrained:
        if checkpoint_path is None:
            checkpoint_path = hf_hub_download("RootYuan/" + version, f"{version}.pt")
        model.from_pretrained(checkpoint_path)
        
    return model
        

def create_pipeline_for_train(version='pipeline-v1', stage1_pretrained=True):
    config = Config(config=ver2cfg[version])
    model = Pipeline(config, stage1_pretrained=stage1_pretrained)
    
    return model
