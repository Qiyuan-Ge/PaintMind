import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def hinge_g_loss(fake):
    return -fake.mean()


def hinge_d_loss(fake, real):
    loss_fake = torch.mean(F.relu(1. + fake))
    loss_real = torch.mean(F.relu(1. - real))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


class PerceptualLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.model = timm.create_model(model_name='vgg16', pretrained=True).features
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, rec, img):
        n = 0
        loss = 0
        for id, layer in self.model.named_children():
            if isinstance(layer, nn.MaxPool2d):
                loss += F.mse_loss(rec, img)
                n += 1
            rec = layer(rec)
            img = layer(img)
        
        return loss / n