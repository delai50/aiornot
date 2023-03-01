import os
import shutil
import glob
import inspect
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

import torch
import torchvision.transforms as T
import pytorch_lightning as pl
import argparse
import albumentations as A

from dataset import AIorNotDataModule
from model import AIorNotModel
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, StochasticWeightAveraging  
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import optuna
import joblib
import gc

from config_main import CFG

from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
import sandesh



# Create folds
def create_folds(df_train, seed):
    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=seed)
    df_train["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df_train, df_train[CFG.target_col])):
        df_train.loc[val_idx, "fold"] = fold
    return df_train




# Logging
def init_logger(logs_path):
    log_file = os.path.join(logs_path, "train.log")
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger




# Log config file
def log_cfg(logger):
    attrs = inspect.getmembers(CFG, lambda a:not(inspect.isroutine(a)))
    [logger.info(f"{a[0]}: {a[1]}") for a in attrs if not(a[0].startswith('__') and a[0].endswith('__'))]
    logger.info(" ")




# Update CFG object with parser arguments
def update_cfg(args):
    attrs = inspect.getmembers(CFG, lambda a:not(inspect.isroutine(a)))
    attrs = [a for a in attrs if not(a[0].startswith('__') and a[0].endswith('__'))]
    for a in args.__dict__: setattr(CFG, a, args.__dict__[a])
    



# Draw learning curves
def draw_curves():
    
    fig, ax = plt.subplots(len(CFG.folds_used), 2, figsize=(12,4*len(CFG.folds_used)))
    
    for fold in CFG.folds_used:
        path = os.path.join(CFG.logs_path, "curves", f"version_{fold}", "events*")
        files = glob.glob(path)[0]
        event_acc = EventAccumulator(files, size_guidance={"scalars": 0})
        event_acc.Reload()
        
        scalars = {}
        for tag in event_acc.Tags()["scalars"]:
            events = event_acc.Scalars(tag)
            scalars[tag] = [event.value for event in events]

        ax[fold,0].plot(range(len(scalars["lr-AdamW/pg1"])), scalars["lr-AdamW/pg1"])
        ax[fold,0].set_xlabel("epoch")
        ax[fold,0].set_ylabel("lr")
        ax[fold,0].set_title(f"Fold{fold}")

        ax[fold,1].plot(range(len(scalars["train_loss_epoch"])), scalars["train_loss_epoch"], label="train_loss_epoch")
        ax[fold,1].plot(range(len(scalars["val_loss_epoch"])), scalars["val_loss_epoch"], label="val_loss_epoch")
        ax[fold,1].plot(range(len(scalars["val_loss_epoch"])), scalars["val_ove_metric"], label="val_ove_epoch")
        ax[fold,1].set_xlabel("epoch")
        ax[fold,1].set_ylabel("loss")
        ax[fold,1].set_title(f"Fold{fold}")
        ax[fold,1].legend()

    fig.savefig(os.path.join(CFG.logs_path, "curves", "curves.png"))




# Crossvalidation
def crossvalidation(df_train, df_test):
    
    oofs = np.zeros([df_train.shape[0], CFG.output_classes])
    fold_metric_list = []

    for fold in CFG.folds_used:
        
        # Train/val split
        train_idx = df_train["fold"] != fold
        val_idx = df_train["fold"] == fold
        df_train_cv = df_train.loc[train_idx,:].reset_index(drop=True)
        df_val_cv = df_train.loc[val_idx,:].reset_index(drop=True)
        
        # Tensorboard logger
        tb_logger = TensorBoardLogger(
            save_dir = CFG.logs_path,
            version = fold,
            name = "curves"
        )
        
        # Data module
        dm = AIorNotDataModule(train_df = df_train_cv, 
                               val_df = df_val_cv,
                               test_df = df_test,
                               cfg = CFG)
        dm.setup(stage="fit")
        
        # Callbacks
        swa_cb = StochasticWeightAveraging(swa_epoch_start = 0.8, 
                                           swa_lrs = CFG.lr, 
                                           annealing_epochs = 1, 
                                           annealing_strategy = "cos", 
                                           avg_fn = None, 
                                           device = "cuda")
        es_cb = EarlyStopping(monitor = CFG.monitor, 
                              min_delta = 0.00, 
                              patience = CFG.es_patience, 
                              verbose = True, 
                              mode = CFG.opt_mode)
        ckpt_cb = ModelCheckpoint(monitor = CFG.monitor,
                                  dirpath = CFG.models_path,
                                  save_top_k = 1,
                                  save_last = True,
                                  save_weights_only = True,
                                  filename = f"model_fold{fold}" + "_{epoch}_{val_loss:.5f}_{val_ove_metric:.4f}_best",
                                  verbose = True,
                                  mode = CFG.opt_mode)
        ckpt_cb.CHECKPOINT_NAME_LAST = f"model_fold{fold}" + "_{epoch}_{val_loss:.5f}_{val_ove_metric:.4f}_last"
        lr_cb = LearningRateMonitor()
        
        
        # Train model
        model = AIorNotModel(CFG)
        trainer = pl.Trainer(max_epochs = CFG.epochs, 
                             val_check_interval = CFG.val_check_interval,
                             accumulate_grad_batches = CFG.accumulate_grad_batches, 
                             gpus = CFG.gpus, 
                             precision = CFG.precision, 
                             accelerator = CFG.accelerator,
                             deterministic = True,
                             logger = tb_logger,
                             callbacks = [ckpt_cb, es_cb, lr_cb]) # swa_cb
        seed_everything(CFG.seed, workers=True)
        trainer.fit(model, dm)
        dm._has_setup_fit = False
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        
        # Pseudolabeling
        if CFG.do_pseudolabeling:
            df_test_pseudo = df_test.copy()
            df_test_pseudo[CFG.target_col] = 0
            
            checkpoints = glob.glob(os.path.join(CFG.models_path, f"*fold{fold}*{CFG.oofs_mode}.ckpt"))
            test_probs = np.zeros([len(df_test_pseudo), CFG.output_classes])
            
            for ckpt in checkpoints:
                model = AIorNotModel.load_from_checkpoint(ckpt, cfg=CFG)
                dm.setup(stage="predict", pred_type="test")
                probs = trainer.predict(model, dm)
                probs = torch.cat(probs, dim=0)
                probs = probs.detach().cpu().numpy()
                test_probs += probs / len(checkpoints)
                dm._has_setup_predict = False
            
            checkpoints = glob.glob(os.path.join(CFG.models_path, f"*fold{fold}*.ckpt"))
            for ckpt in checkpoints: os.remove(ckpt)
            del dm 
            del model
            del trainer
            gc.collect()
            torch.cuda.empty_cache()

            df_test_pseudo[CFG.target_col] = test_probs
            df_test_pseudo = df_test_pseudo.loc[(df_test_pseudo[CFG.target_col]>=(1-CFG.pseudo_prob)) | (df_test_pseudo[CFG.target_col]<=CFG.pseudo_prob)]
            logger.info(f"Fold{fold} pseudolabeling shape: {len(df_test_pseudo)}")
            df_train_cv = pd.concat([df_train_cv, df_test_pseudo], axis=0).reset_index(drop=True)
            
            dm = AIorNotDataModule(train_df = df_train_cv, 
                                   val_df = df_val_cv,
                                   test_df = df_test,
                                   cfg = CFG)
            seed_everything(CFG.seed, workers=True)
            dm.setup(stage="fit")
            
            model = AIorNotModel(CFG)
            
            ckpt_cb = ModelCheckpoint(monitor = CFG.monitor,
                                  dirpath = CFG.models_path,
                                  save_top_k = 1,
                                  save_last = True,
                                  save_weights_only = True,
                                  filename = f"model_fold{fold}" + "_{epoch}_{val_loss:.5f}_{val_ove_metric:.4f}_best",
                                  verbose = True,
                                  mode = CFG.opt_mode)
            ckpt_cb.CHECKPOINT_NAME_LAST = f"model_fold{fold}" + "_{epoch}_{val_loss:.5f}_{val_ove_metric:.4f}_last"
            lr_cb = LearningRateMonitor()
            
            trainer = pl.Trainer(max_epochs = CFG.epochs, 
                                 val_check_interval = CFG.val_check_interval,
                                 accumulate_grad_batches = CFG.accumulate_grad_batches, 
                                 gpus = CFG.gpus, 
                                 precision = CFG.precision, 
                                 accelerator = CFG.accelerator,
                                 deterministic = True,
                                 logger = tb_logger,
                                 callbacks = [ckpt_cb, es_cb, lr_cb]) # swa_cb
            seed_everything(CFG.seed, workers=True)
            trainer.fit(model, dm)
            dm._has_setup_fit = False
        
        
        # Log best/last epoch/metric per fold
        best_ckpt = glob.glob(os.path.join(CFG.models_path, f"*fold{fold}*best.ckpt"))[0]
        last_ckpt = glob.glob(os.path.join(CFG.models_path, f"*fold{fold}*last.ckpt"))[0]
        best_epoch = re.search("epoch=(.+?)_", best_ckpt).group(1)
        best_metric = re.search("val_ove_metric=(.+?)_", best_ckpt).group(1)
        last_epoch = re.search("epoch=(.+?)_", last_ckpt).group(1)
        last_metric = re.search("val_ove_metric=(.+?)_", last_ckpt).group(1)
        if CFG.log:
            logger.info(f"Fold{fold} best metric: {best_metric}, best epoch: {best_epoch}")
            logger.info(f"Fold{fold} last metric: {last_metric}, last epoch: {last_epoch}")
        
        # Get oofs
        checkpoints = glob.glob(os.path.join(CFG.models_path, f"*fold{fold}*{CFG.oofs_mode}.ckpt"))
        for ckpt in checkpoints:
            model = AIorNotModel.load_from_checkpoint(ckpt, cfg=CFG)
            dm.setup(stage="predict", pred_type="oofs")
            probs = trainer.predict(model, dm)
            probs = torch.cat(probs, dim=0)
            probs = probs.detach().cpu().numpy() 
            oofs[val_idx,:] += probs / len(checkpoints)
            dm._has_setup_predict = False
            
        # Fold metric
        targets = df_val_cv[CFG.target_col].values
        preds = oofs[val_idx,:]
        fold_metric = log_loss(targets, preds)
        fold_metric_list.append(fold_metric)
        if CFG.log:
            logger.info(f"Fold{fold} metric: {fold_metric:.5f}")
            logger.info(" ")
        
        # Delete objects
        del model
        del dm
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    # Save oofs
    oofs_df = df_train.copy()
    oofs_df["preds"] = oofs
    oofs_df.to_csv(os.path.join(CFG.oofs_path, "oofs.csv"), index=False)
    
    # Logging final metric
    avg_metric = np.mean(fold_metric_list)
    std_metric = np.std(fold_metric_list)
    ove_targets = oofs_df.loc[oofs_df["fold"].isin(CFG.folds_used), CFG.target_col].values
    ove_preds = oofs_df.loc[oofs_df["fold"].isin(CFG.folds_used), "preds"].values
    ove_metric = log_loss(ove_targets, ove_preds)
    if CFG.log:
        logger.info(f"Avg metric: {avg_metric:.5f}, Std metric: {std_metric:.5f}")
        logger.info(f"Overall metric: {ove_metric:.5f}")
        sandesh.send(CFG.sim_name, webhook=CFG.webhook)
        sandesh.send(f"Avg metric: {avg_metric:.5f}, Std metric: {std_metric:.5f}", webhook=CFG.webhook)
        sandesh.send(f"Overall metric: {ove_metric:.5f}", webhook=CFG.webhook)
    
    return ove_metric




# Full train
def full_train(df_train, df_test):
    
    dm = AIorNotDataModule(train_df = df_train,
                           val_df = df_train.sample(1),
                           cfg = CFG)
    dm.setup(stage="fit")
    
    # Logger
    tb_logger = TensorBoardLogger(save_dir = CFG.logs_path,
                                  version = CFG.n_folds+1,
                                  name = "curves")
    # Callbacks
    swa_cb = StochasticWeightAveraging(swa_epoch_start = 0.8,
                                       swa_lrs = CFG.lr,
                                       annealing_epochs = 1,
                                       annealing_strategy = "cos",
                                       avg_fn = None,
                                       device = "cuda")
    ckpt_cb = ModelCheckpoint(monitor = CFG.monitor,
                              dirpath = CFG.models_path,
                              save_top_k = 0,
                              save_last = True,
                              save_weights_only = True,
                              verbose = True,
                              mode = CFG.opt_mode)
    ckpt_cb.CHECKPOINT_NAME_LAST = f"model_all" + "_{epoch}"
    lr_cb = LearningRateMonitor() 
    
    model = AIorNotModel(CFG)
    trainer = pl.Trainer(max_epochs = CFG.epochs,
                         limit_val_batches = 0,
                         num_sanity_val_steps = 0,  
                         accumulate_grad_batches = CFG.accumulate_grad_batches,
                         gpus = CFG.gpus, 
                         precision = CFG.precision, 
                         accelerator = CFG.accelerator,
                         deterministic = True,
                         logger = tb_logger,
                         callbacks = [ckpt_cb, lr_cb]) # swa_cb
    
    seed_everything(CFG.seed, workers=True)
    os.environ["PYTHONHASHSEED"] = str(CFG.seed)
    torch.backends.cudnn.deterministic = True
    trainer.fit(model, dm)
    dm._has_setup_fit = False
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    
    # Pseudolabeling
    if CFG.do_pseudolabeling:
        df_test_pseudo = df_test.copy()
        df_test_pseudo[CFG.target_col] = 0
        
        checkpoints = glob.glob(os.path.join(CFG.models_path, f"*all*.ckpt"))
        test_probs = np.zeros([len(df_test_pseudo), CFG.output_classes])
            
        for ckpt in checkpoints:
            model = AIorNotModel.load_from_checkpoint(ckpt, cfg=CFG)
            dm.setup(stage="predict", pred_type="test")
            probs = trainer.predict(model, dm)
            probs = torch.cat(probs, dim=0)
            probs = probs.detach().cpu().numpy()
            test_probs += probs / len(checkpoints)
            dm._has_setup_predict = False
        
        checkpoints = glob.glob(os.path.join(CFG.models_path, f"*all*.ckpt"))
        for ckpt in checkpoints: os.remove(ckpt)
        del model
        del dm
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        df_test_pseudo[CFG.target_col] = test_probs
        df_test_pseudo = df_test_pseudo.loc[(df_test_pseudo[CFG.target_col]>=(1-CFG.pseudo_prob)) | (df_test_pseudo[CFG.target_col]<=CFG.pseudo_prob)]
        logger.info(f"Pseudolabeling shape: {len(df_test_pseudo)}")
        df_train = pd.concat([df_train, df_test_pseudo], axis=0).reset_index(drop=True)
                
        dm = AIorNotDataModule(train_df = df_train,
                               val_df = df_train.sample(1),
                               cfg = CFG)
        seed_everything(CFG.seed, workers=True)
        dm.setup(stage="fit")

        model = AIorNotModel(CFG)
        
        ckpt_cb = ModelCheckpoint(monitor = CFG.monitor,
                              dirpath = CFG.models_path,
                              save_top_k = 0,
                              save_last = True,
                              save_weights_only = True,
                              verbose = True,
                              mode = CFG.opt_mode)
        ckpt_cb.CHECKPOINT_NAME_LAST = f"model_all" + "_{epoch}"
        lr_cb = LearningRateMonitor() 
        
        trainer = pl.Trainer(max_epochs = CFG.epochs, 
                             val_check_interval = CFG.val_check_interval,
                             accumulate_grad_batches = CFG.accumulate_grad_batches, 
                             gpus = CFG.gpus, 
                             precision = CFG.precision, 
                             accelerator = CFG.accelerator,
                             deterministic = True,
                             logger = tb_logger,
                             callbacks = [ckpt_cb, lr_cb]) # swa_cb
        seed_everything(CFG.seed, workers=True)
        trainer.fit(model, dm)
        dm._has_setup_fit = False

    del dm
    del trainer
    gc.collect()
    torch.cuda.empty_cache()


# Inference
def inference(df_test):
    
    if CFG.infer_mode == "all":
        checkpoints = glob.glob(os.path.join(CFG.models_path, f"*all*.ckpt"))
    else:
        checkpoints = glob.glob(os.path.join(CFG.models_path, f"*fold*{CFG.oofs_mode}*.ckpt"))
        
    dm = AIorNotDataModule(test_df = df_test,
                           cfg = CFG)
    dm.setup(stage="predict", pred_type="test")
    trainer = pl.Trainer(gpus = CFG.gpus, 
                         precision = CFG.precision, 
                         accelerator = CFG.accelerator)

    test_probs = np.zeros([df_test.shape[0], CFG.output_classes])
    for ckpt in checkpoints:
        model = AIorNotModel.load_from_checkpoint(ckpt, cfg=CFG)
        probs = trainer.predict(model, dm)
        probs = torch.cat(probs, dim=0)
        probs = probs.detach().cpu().numpy()
        test_probs += probs / len(checkpoints)
        
        del model
        gc.collect()
        torch.cuda.empty_cache()

    sub = df_test[["id", CFG.target_col]]
    sub[CFG.target_col] = test_probs
    sub.to_csv(os.path.join(CFG.subs_path, "submission.csv"), index=False)




# Hyperparameter optimization
def optimize_hparams(df_train, df_test):

    def objective(trial):
        
        # NN parameters
        CFG.epochs = trial.suggest_int("epochs", 5, 9)
        CFG.lr = trial.suggest_categorical("lr", [3e-4, 5e-4, 7e-4])
        
        # Crossvalidation
        metric = crossvalidation(df_train, df_test)
        
        # Delete models
        for f in os.listdir(CFG.models_path):
            os.remove(os.path.join(CFG.models_path, f))
        
        return metric

    if os.path.isfile(os.path.join(CFG.hopts_path, "hpopt.pkl")):
        study = joblib.load(os.path.join(CFG.hopts_path, "hpopt.pkl"))
    else:
        study = optuna.create_study(direction="maximize" if CFG.opt_mode=="max" else "minimize")
    
    study.optimize(objective, timeout=CFG.hopt_timeout)
    joblib.dump(study, os.path.join(CFG.hopts_path, "hpopt.pkl"))



if __name__ == "__main__": 

    if CFG.parser:
        parser = argparse.ArgumentParser()
        parser.add_argument("--sim_name", type=str, required=True)
        parser.add_argument("--model_name", type=str, default="resnet50")
        parser.add_argument("--img_size", nargs="+", type=int)
        parser.add_argument("--seed", type=int, default=34)
        parser.add_argument("--n_folds", type=int, default=5)
        parser.add_argument("--folds_used", nargs="+", type=int)
        parser.add_argument("--do_crossvalidation", action="store_true")
        parser.add_argument("--do_full_train", action="store_true")
        parser.add_argument("--infer_mode", type=str)
        # parser.add_argument("--do_pseudolabeling", action="store_true")
        # parser.add_argument("--pseudo_prob", type=float)
        parser.add_argument("--epochs", type=int, default=5)
        parser.add_argument("--lr", type=float, default=1e-3)
        # parser.add_argument("--lr_backbone", type=float, default=1e-4)
        # parser.add_argument("--lr_head", type=float, default=1e-4)
        parser.add_argument("--batch_size", type=int, default=128)
        # parser.add_argument("--wd", type=float, default=1e-4)
        parser.add_argument("--augmentation_group", type=str)
        parser.add_argument("--mixup", action="store_true")
        parser.add_argument("--cutmix", action="store_true")
        parser.add_argument("--tta", action="store_true")
        parser.add_argument("--gpus", nargs="+", type=int)
        parser.add_argument("--accumulate_grad_batches", type=int, default=1)
        args = parser.parse_args()
        update_cfg(args)
    
    # Create directories
    CFG.oofs_path = os.path.join(CFG.base_path, "outs", "oofs", CFG.sim_name); os.makedirs(CFG.oofs_path, exist_ok=True)
    CFG.models_path = os.path.join(CFG.base_path, "outs", "models", CFG.sim_name); os.makedirs(CFG.models_path, exist_ok=True)
    CFG.logs_path = os.path.join(CFG.base_path, "outs", "logs", CFG.sim_name); os.makedirs(CFG.logs_path, exist_ok=True)
    CFG.subs_path = os.path.join(CFG.base_path, "outs", "subs", CFG.sim_name); os.makedirs(CFG.subs_path, exist_ok=True)
    CFG.hopts_path = os.path.join(CFG.base_path, "outs", "hopts", CFG.sim_name); os.makedirs(CFG.hopts_path, exist_ok=True)
    CFG.code_path = os.path.join(CFG.base_path, "outs", "code", CFG.sim_name); os.makedirs(CFG.code_path, exist_ok=True)
    
    
    
    seed_everything(CFG.seed, workers=True)
    os.environ["PYTHONHASHSEED"] = str(CFG.seed)
    torch.backends.cudnn.deterministic = True
    logger = init_logger(CFG.logs_path)
    log_cfg(logger)
    
    
    
    # Copy src files to sim src folder
    src_path = os.path.join(CFG.base_path, "src", "parser")
    src_files = os.listdir(src_path)
    for fname in src_files:
        fpath = os.path.join(src_path, fname)
        if os.path.isfile(fpath):
            shutil.copy(fpath, CFG.code_path)
    


    # Read the data
    df_train = pd.read_csv(os.path.join(CFG.base_path, "data", "train.csv"))
    df_test = pd.read_csv(os.path.join(CFG.base_path, "data", "sample_submission.csv")) 
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    
    # Debug mode
    if CFG.debug:
        df_train = df_train.sample(n=500, random_state=CFG.seed).reset_index(drop=True)


    # Add full path of the images
    df_train["img_path"] = df_train["id"].apply(lambda x: os.path.join(CFG.base_path, "data", "train", x))
    df_test["img_path"] = df_test["id"].apply(lambda x: os.path.join(CFG.base_path, "data", "test", x))


    # Create folds
    df_train = create_folds(df_train, CFG.seed)
    

    # Crossvalidation
    if CFG.do_hpoptimization:
        optimize_hparams(df_train, df_test)
    
    
    # Crossvalidation
    if CFG.do_crossvalidation:
        crossvalidation(df_train, df_test)
        
    
    # Full train
    if CFG.do_full_train:
        full_train(df_train, df_test)
    
    
    # Inference
    if CFG.do_inference:
        inference(df_test)
        
    
    # Draw learning curves
    if CFG.draw_curves:
        draw_curves()
    
    
    # Error analysis
