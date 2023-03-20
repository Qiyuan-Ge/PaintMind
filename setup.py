from setuptools import setup, find_packages

setup(
    name = 'paintmind',
    version = 0.0,
    author = 'Qiyuan Ge',
    author_email = '542801615@qq.com',
    keywords = [
        'artificial intelligence',
        'deep learning',
        'generate model',
    ],      
    packages = find_packages(),
    install_requires=[
        'einops',
        'numpy',
        'pillow',
        'torch>=1.6',
        'torchvision',
        'timm',
        'kornia',
        'accelerate',
        'tqdm',
        'pycocotools',
        'datasets',
        'sentencepiece',
        'pytorch_lightning',
        'open_clip_torch',
    ],
)
