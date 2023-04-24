from .config import *
from .stage1 import VQModel
from .generate import Pipeline
from huggingface_hub import hf_hub_download

def create_model(arch='pipeline', version='paintmindv1', pretrained=True, checkpoint_path=None):
    config = Config(ver2cfg[version])
    
    if arch == 'vqgan':
        model = VQModel(config)
    elif arch == 'pipeline':
        model = Pipeline(config, stage1_pretrained=False)
    else:
        raise ValueError(f"failed to load arch named {arch}")
        
    if pretrained:
        if checkpoint_path is None:
            checkpoint_path = hf_hub_download("RootYuan/" + version, f"{version}.pt")
        model.from_pretrained(checkpoint_path)
        
    return model
        
def create_pipeline_for_train(version='paintmindv1', stage1_pretrained=True, stage1_checkpoint_path=None):
    model = Pipeline(Config(ver2cfg[version]), stage1_pretrained=stage1_pretrained, stage1_checkpoint_path=stage1_checkpoint_path)
    
    return model
