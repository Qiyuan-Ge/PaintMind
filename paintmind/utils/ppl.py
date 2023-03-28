import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class PerceptualLoss(nn.Module):
    def __init__(self, layer_weights=[1.0, 0.8, 0.5, 0.3, 0.2], device='cuda'):
        super().__init__()
        self.model = timm.create_model(model_name='vgg19', pretrained=True).features
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.layer_weights = layer_weights
        self.mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(IMAGENET_DEFAULT_STD).view(1, -1, 1, 1).to(device)
    
    def renormalize(self, x):
        x = (x + 1) * 0.5
        x = (x - self.mean) / self.std
        return x
    
    def forward(self, x, target):
        x = self.renormalize(x)
        target = self.renormalize(target)
        
        idx = 0
        loss = 0
        for layer in self.model.children():
            if isinstance(layer, nn.MaxPool2d):
                loss += self.layer_weights[idx]*F.mse_loss(x, target)
                idx += 1   
            x = layer(x)
            target = layer(target)
        
        return loss