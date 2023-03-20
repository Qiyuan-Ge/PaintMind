from .version import __version__
from .config import Config, create_model
from .utils.transform import stage1_transform, stage2_transform
from .utils.trainer import VQGANTrainer, PaintMindTrainer

model_list = [
    {'arch':'vqvae', 'version':['vit_s_vqvae', 'vit_b_vqvae']},
    {'arch':'paintmind', 'version':['paintmindv1']}
]
