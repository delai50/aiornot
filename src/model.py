import os
import numpy as np

from sklearn.metrics import log_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from model_utils import GeM, Backbone, mixup, cutmix, divide_norm_bias


class AIorNotModel(pl.LightningModule):
    def __init__(
        self, cfg
    ):
        super().__init__()
        
        self.cfg = cfg
        
        transformer_models = ["swin", "vit", "xcit", "cait", "mixer", "resmlp", "crossvit", "beit"]
        if any([t in self.cfg.model_name for t in transformer_models]):
            self.transformer = True
        else:
            self.transformer = False
        
        self.backbone = Backbone(cfg.model_name, pretrained=self.cfg.pretrained)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Head 1
        self.head = nn.Sequential(
            nn.Linear(self.backbone.out_features, self.cfg.output_classes)
        )
        
        # Head 2
        # self.head = nn.Sequential(
        #         nn.Dropout(0.2),
        #         nn.Linear(self.backbone.out_features, 256, bias=True),
        #         nn.BatchNorm1d(256),
        #         torch.nn.PReLU(),
        #         nn.Linear(256, self.cfg.output_classes)  
        # )
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.best_loss = None


    def forward(self, x):
        x = self.backbone(x)
        
        if not self.transformer:
            x = self.global_pool(x).squeeze()
        else:
            # CLS token
            x = x[:,0,:]
            
        out = self.head(x)
        return out

    
    def get_optimizer_params(self):
        norm_bias_params, non_norm_bias_params = divide_norm_bias(self)
        optimizer_params = [
            {"params": norm_bias_params, "weight_decay": self.cfg.wd},
            {"params": non_norm_bias_params, "weight_decay": 0.}
        ]
        return optimizer_params
    
    
    # def get_optimizer_params(self):
    #     except_wd_layers = ['norm', '.bias']
        
    #     norm_bias_params_backbone = []
    #     non_norm_bias_params_backbone = []
    #     for n, p in self.backbone.named_parameters():
    #         if any([nd in n for nd in except_wd_layers]):
    #             norm_bias_params_backbone.append(p)
    #         else:
    #             non_norm_bias_params_backbone.append(p)
                
    #     norm_bias_params_head = []
    #     non_norm_bias_params_head = []
    #     for n, p in self.head.named_parameters():
    #         if any([nd in n for nd in except_wd_layers]):
    #             norm_bias_params_head.append(p)
    #         else:
    #             non_norm_bias_params_head.append(p)
        
    #     optimizer_params = [
    #         {"params": norm_bias_params_backbone, "lr": self.cfg.lr_backbone, "weight_decay": self.cfg.wd},
    #         {"params": non_norm_bias_params_backbone, "lr": self.cfg.lr_backbone, "weight_decay": 0.},
    #         {"params": norm_bias_params_head, "lr": self.cfg.lr_head, "weight_decay": self.cfg.wd},
    #         {"params": non_norm_bias_params_head, "lr": self.cfg.lr_head, "weight_decay": 0.}
    #     ]
    #     return optimizer_params


    def configure_optimizers(self):
        optimizer_params = self.get_optimizer_params()
        optimizer = getattr(torch.optim, self.cfg.optimizer)(optimizer_params, lr=self.cfg.lr)
        # optimizer = getattr(torch.optim, self.cfg.optimizer)(optimizer_params)
        
        train_batches = len(self.trainer.datamodule.train_dataloader())
        self.train_steps = (self.cfg.epochs * train_batches) // self.cfg.accumulate_grad_batches
        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.train_steps, eta_min=1e-7),
            "monitor": "val_loss",
            "interval": "step"
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
    
        
    def _take_training_step(self, batch, batch_idx):
        images, targets = batch["images"], batch["targets"]

        if self.cfg.mixup and torch.rand(1)[0] < self.cfg.mixup_p:
            mix_images, target_a, target_b, lam = mixup(images, targets, alpha=self.cfg.mixup_alpha)
            logits = self(mix_images).squeeze(1)
            loss = self.loss_fn(logits, target_a) * lam + \
                (1 - lam) * self.loss_fn(logits, target_b)
        elif self.cfg.cutmix and torch.rand(1)[0] < self.cfg.cutmix_p:
            mix_images, target_a, target_b, lam = cutmix(images, targets, alpha=self.cfg.mixup_alpha)
            logits = self(mix_images).squeeze(1)
            loss = self.loss_fn(logits, target_a) * lam + \
                (1 - lam) * self.loss_fn(logits, target_b)
        else:
            logits = self(images)
            loss = self.loss_fn(logits.squeeze(1), targets)
            
        return loss
    
    
    def on_train_start(self):
        seed_everything(self.cfg.seed, workers=True)
        os.environ["PYTHONHASHSEED"] = str(self.cfg.seed)
        torch.backends.cudnn.deterministic = True
        

    def training_step(self, batch, batch_idx):
        loss = self._take_training_step(batch, batch_idx)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        images, targets = batch["images"], batch["targets"]
        logits = self(images)
        
        val_loss = self.loss_fn(logits.squeeze(1), targets)
        probs = torch.sigmoid(logits)
        
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"logits": logits, "probs": probs, "targets": targets}


    def validation_epoch_end(self, outputs):
        probs_flat = torch.cat([v["probs"] for v in outputs], dim=0)
        probs_flat_np = probs_flat.cpu().detach().numpy().astype(np.float64)
        targets_flat = torch.cat([v["targets"] for v in outputs], dim=0)
        targets_flat_np = targets_flat.cpu().detach().numpy()
        
        val_ove_metric = log_loss(targets_flat_np, probs_flat_np)
        
        self.log("val_ove_metric", val_ove_metric, prog_bar=True)


    def predict_step(self, batch, batch_idx):
        if "images_1" in batch:
            logits = 0.
            for images in batch.values():
                logits += self(images) / len(batch.values())
            probs = torch.sigmoid(logits)
        else:
            images = batch["images"]
            logits = self(images)
            probs = torch.sigmoid(logits)

        return probs



# EMBEDDINGS AND BOOSTER MODELS

# class PetFinderEmbeddingsModel(Pog3Model):
#     def __init__(
#         self, model_name, epochs, lr, wd, 
#         accumulate_grad_batches, 
#         drop_rate, drop_path_rate,
#         mixup, mixup_p, mixup_alpha,
#         cutmix, cutmix_p, cutmix_alpha,
#         classification=True, 
#         pretrained=False
#     ):
#         super().__init__(
#             model_name, epochs, lr, wd, 
#             accumulate_grad_batches, 
#             drop_rate, drop_path_rate,
#             mixup, mixup_p, mixup_alpha,
#             cutmix, cutmix_p, cutmix_alpha,
#             classification, 
#             pretrained
#         )

#     def forward(self, x):
#         return self.backbone(x)

#     def predict_step(self, batch, batch_idx):
#         images = batch['images']

#         logits = self(images)

#         return logits.view(logits.shape[0], -1)


# class PetFinderBoostedModel(Pog3Model):
#     def __init__(
#         self, model_name, epochs, lr, wd, 
#         accumulate_grad_batches, 
#         drop_rate, drop_path_rate,
#         mixup, mixup_p, mixup_alpha,
#         cutmix, cutmix_p, cutmix_alpha,
#         classification=True, 
#         pretrained=False, booster=None
#     ):
#         super().__init__(
#             model_name, epochs, lr, wd, 
#             accumulate_grad_batches, 
#             drop_rate, drop_path_rate,
#             mixup, mixup_p, mixup_alpha,
#             cutmix, cutmix_p, cutmix_alpha,
#             classification, 
#             pretrained
#         )
        
#         self.booster = booster

#     def forward(self, x):
#         emb = self.backbone(x)
#         return emb, self.head(emb)

#     def predict_step(self, batch, batch_idx):
#         images = batch['images']

#         emb, logits = self(images)
#         booster_logits = self.booster.predict(emb.detach())
        
#         logits = (logits + booster_logits) / 2

#         return logits.view(logits.shape[0], -1)