from .config import *
from .stage1 import VQModel
from .pipeline import Pipeline
from huggingface_hub import hf_hub_download

def create_model(arch='pipeline', version='pipeline-v1', pretrained=True, checkpoint_path=None):
    config = Config(config=ver2cfg[version])
    
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
