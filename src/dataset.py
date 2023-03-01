from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
# from augmix import RandomAugMix

import albumentations as A
import pytorch_lightning as pl
import torch
import cv2

from config_main import CFG



def augmentation_group(img_size):   
    if CFG.augmentation_group == "HT_soft":
        augs = A.Compose([
            A.Resize(height=img_size[0], width=img_size[1], p=1),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
                max_pixel_value = 255.0,
                p = 1.0,
            ),
            ToTensorV2()
        ])
    elif CFG.augmentation_group == "HT_medium":
        augs = A.Compose([
            A.Resize(height=img_size[0], width=img_size[1], p=1),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=45, p=0.5),
            A.CoarseDropout(p=0.5),
            A.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
                max_pixel_value = 255.0,
                p = 1.0,
            ),
            ToTensorV2()
        ])
    elif CFG.augmentation_group == "HT_hard":
        augs = A.Compose([
            A.RandomResizedCrop(height=img_size[0], width=img_size[1], p=1),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=15, p=0.5),
            A.CoarseDropout(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
                max_pixel_value = 255.0,
                p = 1.0,
            ),
            ToTensorV2()
        ])
    elif CFG.augmentation_group == "custom_soft":
        augs = A.Compose([
            A.Resize(height=img_size[0], width=img_size[1], p=1),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
                max_pixel_value = 255.0,
                p = 1.0,
            ),
            ToTensorV2()
        ])
    elif CFG.augmentation_group == "custom_medium":
        augs = A.Compose([
            A.Resize(height=img_size[0], width=img_size[1], p=1),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(rotate_limit=0, border_mode=0, p=0.5),
            A.CoarseDropout(max_height=24, max_width=24, p=0.5),
            A.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
                max_pixel_value = 255.0,
                p = 1.0,
            ),
            ToTensorV2()
        ])
    elif CFG.augmentation_group == "custom_hard":
        augs = A.Compose([
            A.Resize(height=img_size[0], width=img_size[1], p=1),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(rotate_limit=0, border_mode=0, p=0.5),
            A.RGBShift(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.CoarseDropout(max_height=24, max_width=24, p=0.5),
            A.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
                max_pixel_value = 255.0,
                p = 1.0,
            ),
            ToTensorV2()
        ])   
    return augs



def get_default_transforms(img_size):
    transform = {
        "train": augmentation_group(img_size),
            
            # A.Compose([
                # Augs 1
                # A.Resize(height=img_size[0], width=img_size[1], p=1),
                # A.HorizontalFlip(p=0.5),
                        
                # Augs 2
                # A.Resize(height=img_size[0], width=img_size[1], p=1),
                # A.HorizontalFlip(p=0.5),
                # A.ShiftScaleRotate(rotate_limit=0, border_mode=0, p=0.5),
                # A.CoarseDropout(max_height=24, max_width=24, p=0.5),
                
                # Augs 3
                # A.Resize(height=img_size[0], width=img_size[1], p=1),
                # A.HorizontalFlip(p=0.5),
                # A.RGBShift(p=0.5),
                # A.RandomBrightnessContrast(p=0.5),
                # A.CoarseDropout(max_height=16, max_width=16, p=0.5),
                
                # Augs 4
                # A.Resize(height=img_size[0], width=img_size[1], p=1),
                # A.HorizontalFlip(p=0.5),
                # A.ShiftScaleRotate(rotate_limit=0, border_mode=0, p=0.5),
                # A.RGBShift(p=0.5),
                # A.RandomBrightnessContrast(p=0.5),
                # A.CoarseDropout(max_height=16, max_width=16, p=0.5),
                
                # Augs 5
                # A.Resize(height=img_size[0], width=img_size[1], p=1),
                # A.HorizontalFlip(p=0.5),
                # A.ShiftScaleRotate(rotate_limit=0, border_mode=0, p=0.5),
                # A.RGBShift(p=0.5),
                # A.CoarseDropout(max_height=16, max_width=16, p=0.5),
                
                # Augs 6
                # A.Resize(height=img_size[0], width=img_size[1], p=1),
                # A.HorizontalFlip(p=0.5),
                # A.ShiftScaleRotate(rotate_limit=0, border_mode=0, p=0.5),
                # A.RandomBrightnessContrast(p=0.5),
                # A.CoarseDropout(max_height=16, max_width=16, p=0.5),
                
                # A.Normalize(
                #     mean = [0.485, 0.456, 0.406],
                #     std = [0.229, 0.224, 0.225],
                #     max_pixel_value = 255.0,
                #      p = 1.0,
                #  ),
                
                #  ToTensorV2()
            # ]),
        "inference": A.Compose([
            A.Resize(height=img_size[0], width=img_size[1], p=1),
            A.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
                max_pixel_value = 255.0,
                p = 1.0,
            ),
            ToTensorV2()
        ]),
        "tta": A.Compose([
            A.Resize(height=img_size[0], width=img_size[1], p=1),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
                max_pixel_value = 255.0,
                p = 1.0,
            ),
            ToTensorV2()
        ]),
    }
    return transform




class AIorNotDataset(Dataset):
    def __init__(self, img_paths, targets=None, img_size=(224,224), inference=False, tta=False):
        self.img_paths = img_paths
        self.targets = targets
        self.tta = tta
        if tta:
            self.augs = get_default_transforms(img_size)["tta"]
        elif inference:
            self.augs = get_default_transforms(img_size)["inference"]
        else:
            self.augs = get_default_transforms(img_size)["train"]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if not self.tta:
            image = self.augs(image=image)["image"]
            
        if self.targets is not None:
            target = torch.tensor(self.targets[idx]).float()
            return {
                "images": image,
                "targets": target
            }
        else:
            if self.tta:
                return {f"images_{k}": self.augs(image=image)["image"] for k in range(len(self.augs))}
            else:
                return {"images": image}


class AIorNotDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_df=None, val_df=None, test_df=None,
        cfg=None
    ):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.cfg = cfg

    def setup(self, stage=None, pred_type=None):
               
        if stage == "fit" or stage is None:
                            
            print(f"Training samples: {len(self.train_df)}")
            print(f"Validation samples: {len(self.val_df)}")
            
            # Train dataset
            img_paths = self.train_df["img_path"].values
            targets = self.train_df[self.cfg.target_col].values
            self.train_ds = AIorNotDataset(img_paths, targets, img_size=self.cfg.img_size)
            
            # Valid dataset
            img_paths = self.val_df["img_path"].values
            targets = self.val_df[self.cfg.target_col].values
            self.val_ds = AIorNotDataset(img_paths, targets, img_size=self.cfg.img_size, inference=True)
            
        elif stage == "predict":
            
            if pred_type == "oofs":
                # Predict test dataset
                print(f"Prediction samples: {len(self.val_df)}")
                img_paths = self.val_df["img_path"].values
                self.pred_ds = AIorNotDataset(img_paths, img_size=self.cfg.img_size, inference=True, tta=self.cfg.tta)
                
            elif pred_type == "test":
                # Predict valid dataset
                print(f"Prediction samples: {len(self.test_df)}")
                img_paths = self.test_df["img_path"].values
                self.pred_ds = AIorNotDataset(img_paths, img_size=self.cfg.img_size, inference=True, tta=self.cfg.tta)
            
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, 
                          shuffle = True, 
                          num_workers = self.cfg.num_workers,
                          pin_memory = self.cfg.pin_memory, 
                          batch_size = self.cfg.batch_size, 
                          drop_last = True
                          )

    def val_dataloader(self):
        return DataLoader(self.val_ds, 
                          shuffle = False, 
                          num_workers = self.cfg.num_workers,
                          pin_memory = self.cfg.pin_memory, 
                          batch_size = self.cfg.batch_size
                          )

    def predict_dataloader(self):
        return DataLoader(self.pred_ds, 
                          shuffle = False, 
                          num_workers = self.cfg.num_workers,
                          pin_memory = self.cfg.pin_memory, 
                          batch_size = self.cfg.batch_size)