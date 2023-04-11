# PaintMind

## Install
````
pip install git+https://github.com/Qiyuan-Ge/PaintMind.git
````

## Import
````
import paintmind as pm
````

## Reconstruction
You could download the weights of the pretrained vit-vqgan to local from https://huggingface.co/RootYuan/vit-s-vqgan

#### 1.
````
pm.reconstruction(img_url='https://cdn.pixabay.com/photo/2014/10/22/15/47/squirrel-498139_960_720.jpg')
````
<div align=center>
<img src="https://github.com/Qiyuan-Ge/PaintMind/blob/main/assets/rec_1.png?raw=true">
</div>

#### 2.
````
pm.reconstruction(img_url='https://cdn.pixabay.com/photo/2017/04/09/10/44/sea-shells-2215408_960_720.jpg')
````
<div align=center>
<img src="https://github.com/Qiyuan-Ge/PaintMind/blob/main/assets/rec_2.png?raw=true">
</div>

#### 3.
````
pm.reconstruction(img_url='https://cdn.pixabay.com/photo/2015/06/19/21/24/avenue-815297_960_720.jpg')
````
<div align=center>
<img src="https://github.com/Qiyuan-Ge/PaintMind/blob/main/assets/rec_3.png?raw=true">
</div>

#### 4.
````
pm.reconstruction(img_url='https://cdn.pixabay.com/photo/2017/03/30/18/17/girl-2189247_960_720.jpg')
````
<div align=center>
<img src="https://github.com/Qiyuan-Ge/PaintMind/blob/main/assets/rec_4.png?raw=true">
</div>

#### 5.
````
pm.reconstruction(img_url='https://cdn.pixabay.com/photo/2017/10/28/07/47/woman-2896389_960_720.jpg')
````
<div align=center>
<img src="https://github.com/Qiyuan-Ge/PaintMind/blob/main/assets/rec_5.png?raw=true">
</div>

````

## Text2Image
Soon Coming~~


## Citation
````
@misc{paintmind,
  author = {Qiyuan Ge},
  title = {PaindMind},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Qiyuan-Ge/PaintMind}},
}
````
