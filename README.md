# PaintMind
<div align=center>
<img src="https://github.com/Qiyuan-Ge/PaintMind/blob/main/assets/A_beautiful_girl_celebrating_her_birthday.png?raw=true" width="512">
</div>

## Install
````
pip install git+https://github.com/Qiyuan-Ge/PaintMind.git
````

## Import
````
import paintmind as pm
````

## Reconstruction
````
img = Image.open(img_path).convert('RGB')
img = pm.stage1_transform(is_train=False)(img)
# load pretrained vit-vqgan
model = pm.create_model(arch='vqgan', version='vit-s-vqgan', pretrained=True)
# encode image to latent
z, _, _ = model.encode(img.unsqueeze(0))
# decode latent to image
rec = model.decode(z).squeeze(0)
rec = torch.clamp(rec, -1., 1.)
````
- You could also download the weights of the pretrained vit-vqgan to local from https://huggingface.co/RootYuan/vit-s-vqgan
- Play with [Colab Notebook](https://colab.research.google.com/drive/1J8M97_HDAVXWQB4qp6yIBI7nPs-ZGXQz?usp=sharing).

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

## Text2Image
Not finish yet~~
