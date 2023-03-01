import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha),0.3,0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)

    return new_data, target, shuffled_target, lam

def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    assert alpha > 0, "alpha should be larger than 0"
    assert x.size(0) > 1, "Mixup cannot be applied to a single instance."

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = torch.nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

class Backbone(nn.Module):
    def __init__(self, name="efficientnet_b0", pretrained=False, **kwargs):
        super(Backbone, self).__init__()
        self.net = timm.create_model(name, pretrained=pretrained, in_chans=3, **kwargs)
        
        if "regnet" in name:
            self.out_features = self.net.head.fc.in_features
        elif "rexnet" in name:
            self.out_features = self.net.head.fc.in_features
        elif "swin" in name:
            self.out_features = self.net.num_features
        elif "cait" in name:
            self.out_features = self.net.num_features
        elif "mixer" in name or "resmlp" in name:
            self.out_features = self.net.num_features
        elif "xcit" in name:
            self.out_features = self.net.num_features
        elif "crossvit" in name:
            self.out_features = self.net.num_features
        elif "levit" in name:
            self.out_features = self.net.num_features
        elif "vit" in name:
            self.out_features = self.net.head.in_features
        elif name == "vit_deit_base_distilled_patch16_384":
            self.out_features = 768
        elif "csp" in name:
            self.out_features = self.net.head.fc.in_features
        elif "tresnet" in name:
            self.out_features = self.net.head.fc.in_features
        elif "res" in name: # works also for resnest
            self.out_features = self.net.fc.in_features
        elif "efficientnet" in name:
            self.out_features = self.net.classifier.in_features
        elif "densenet" in name:
            self.out_features = self.net.classifier.in_features
        elif "senet" in name:
            self.out_features = self.net.fc.in_features
        elif "inception" in name:
            self.out_features = self.net.last_linear.in_features
        elif "nfnet" in name:
            self.out_features = self.net.head.fc.in_features
        else:
            self.out_features = self.net.num_features

    def forward(self, x):
        x = self.net.forward_features(x)

        return x

def divide_norm_bias(model): 
    norm_bias_params = []
    non_norm_bias_params = []
    except_wd_layers = ['norm', '.bias']
    for n, p in model.named_parameters():
        if any([nd in n for nd in except_wd_layers]):
            norm_bias_params.append(p)
        else:
            non_norm_bias_params.append(p)
    return norm_bias_params, non_norm_bias_params