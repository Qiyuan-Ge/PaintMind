from setuptools import setup, find_packages

setup(
    name='paintmind',
    version='0.0.0',
    author='Qiyuan Ge',
    author_email='542801615@qq.com',
    url='https://github.com/Qiyuan-Ge/PaintMind',
    keywords = [
        'artificial intelligence',
        'deep learning',
        'text-to-image',
        'generate model',
    ],      
    packages = find_packages(),
    install_requires=[
        'einops',
        'numpy',
        'pillow',
        'torch>=1.13',
        'torchvision',
        'lpips',
        'kornia',
        'accelerate',
        'tqdm',
        'pycocotools',
        'datasets',
        'sentencepiece',
        'open_clip_torch',
        'transformers',
    ],
)
