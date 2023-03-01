
import os
import inspect
import joblib

from itertools import combinations
from time import time

import pandas as pd
import numpy as np

from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression, LinearRegression

from functools import partial
from scipy.optimize import fmin, minimize

from config_ensemble import CFG

from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler

    
# Initialize logger
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


# Optimization created by Mark Tennenholtz
class OptimizeScore_MT():
    def __init__(self):
        self.coef_ = 0
    
    def metric(self, coef):
        x_coef = self.X * coef
        predictions = np.sum(x_coef, axis=1)
        score = log_loss(self.y, predictions)
        return score
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        
        tol = 1e-10
        init_guess = [1 / self.X.shape[1]] * self.X.shape[1]
        bnds = [(0, 1) for _ in range(self.X.shape[1])]
        cons = {"type": "eq", 
                "fun": lambda x: np.sum(x) - 1, 
                "jac": lambda x: [1] * len(x)}

        print(f"Initial Blend OOF: {self.metric(init_guess):.6f}", )

        res_scipy = minimize(fun = self.metric, 
                            x0 = init_guess, 
                            method = "Powell", 
                            # method = "SLSQP",
                            bounds = bnds,
                            options = dict(maxiter=1_000_000),
                            tol = tol)

        print(f"Optimised Blend OOF: {res_scipy.fun:.6f}")
        print(f"Optimised Weights: {res_scipy.x}")
        self.coef_ = res_scipy.x
        self.coef_ /= np.sum(self.coef_)
        
    def predict(self, X):
        x_coef = X * self.coef_
        predictions = np.sum(x_coef, axis=1)
        return predictions



# Model selection by hill climbing
def select_models_hill_climbing(oof_dict):
    
    # List names of the initial candidates for the ensemble
    models_list_names = list(oof_dict.keys())
    # List of indices of the initial candidates for the ensemble 
    models_list = list(range(len(models_list_names))) 
    folds = pd.read_csv(oof_dict[models_list_names[0]])["fold"].values

    oofs_matrix = []

    # Create matrix with the probabilities of all the models
    for m in models_list_names:
        oofs_matrix.append(
            pd.read_csv(oof_dict[m])["preds"].values
        )

    y_train = pd.read_csv(oof_dict[models_list_names[0]])[CFG.target_col].values
    X_train = np.stack(oofs_matrix, axis=1)

    # Compute the individual scores
    ini_scores = []
    for k in range(X_train.shape[1]):
        preds = X_train[:,k]
        score = log_loss(y_train, preds)
        ini_scores.append(score)
        logger.info(f"Initial score {models_list_names[k]}: {score:.5f}")

    # Initial best score and best model selected
    best_score = np.min(ini_scores)
    best_models_list = [models_list[np.argmin(ini_scores)]]
    
    # Best score of the current step
    step_best_score = 0

    while step_best_score <= best_score:
        # The new model list is the former without the best models already selected
        models_list = [m for m in models_list if m not in best_models_list]

        # It breaks when there are no remaining models to check
        if len(models_list) == 0:
            break
        
        # Loop for all remaining models and check which one improves the score the most
        scores = []
        for mdl in models_list:
            oofs_probs_model = np.zeros([len(y_train), 1])
        
            sel_models = best_models_list + [mdl]
            X_train_sel = X_train[:,sel_models]
            
            for fold in CFG.folds_used:            
                train_idx = folds!=fold
                val_idx = folds==fold
            
                X_train_cv = X_train_sel[train_idx,:]
                X_val_cv = X_train_sel[val_idx,:]
                y_train_cv = y_train[train_idx]
                y_val_cv = y_train[val_idx]

                import pdb; pdb.set_trace()
                model = OptimizeScore_MT()
                model.fit(X_train_cv, y_train_cv)
                y_val_cv_pred = model.predict(X_val_cv)
            
                oofs_probs_model[val_idx,:] = y_val_cv_pred.reshape(-1,1)

            # Calculate score for the current model
            scores.append(
                log_loss(y_train, oofs_probs_model)
            )
        
        step_best_score = np.min(scores)
    
        if step_best_score < best_score:
            # Append the best model to the list if improves the score
            best_models_list.append(models_list[np.argmin(scores)])
            best_score = step_best_score
        else:
            break
    
    # Final names of the selected models
    best_models_list_names = list(np.array(models_list_names)[best_models_list])
    logger.info(f"Selected models for the ensemble hill climbing: {best_models_list_names}")
    logger.info(f"Final score for the ensemble hill climbing {best_score:.5f}")
    
    return best_models_list_names, best_score



# Model selection by hill descent
def select_models_hill_descent(oof_dict):
    
    # List names of the candidates for the ensemble
    models_list_names = list(oof_dict.keys())
    # List of indices of the candidates for the ensemble 
    models_list = list(range(len(models_list_names))) 
    folds = pd.read_csv(oof_dict[models_list_names[0]])["fold"].values

    # Create matrix with the probabilities of all the models
    oofs_matrix = []
    for m in models_list_names:
        oofs_matrix.append(
            pd.read_csv(oof_dict[m])["preds"].values
        )

    y_train = pd.read_csv(oof_dict[models_list_names[0]])[CFG.target_col].values
    X_train = np.stack(oofs_matrix, axis=1)

    # Compute the individual scores
    ini_scores = []
    for k in range(X_train.shape[1]):
        preds = X_train[:,k]
        score = log_loss(y_train, preds)
        ini_scores.append(score)
        logger.info(f"Initial score {models_list_names[k]}: {score:.5f}")
    
    # Initial best score with all models selected
    oofs_probs_all_models = np.zeros([len(y_train), 1])
    
    for fold in CFG.folds_used:        
        train_idx = folds!=fold
        val_idx = folds==fold
            
        X_train_cv = X_train[train_idx,:]
        X_val_cv = X_train[val_idx,:]
        y_train_cv = y_train[train_idx]
        y_val_cv = y_train[val_idx]
        
        import pdb; pdb.set_trace()
        model = OptimizeScore_MT()
        model.fit(X_train_cv, y_train_cv)
        y_val_cv_pred = model.predict(X_val_cv)
            
        oofs_probs_all_models[val_idx,:] = y_val_cv_pred.reshape(-1,1)

    # Calculate score for the ensemble of all models
    best_score = log_loss(y_train, oofs_probs_all_models)
    best_models_list = models_list.copy()
    
    # Best score of the current step
    step_best_score = 0
    
    # Iterative loop
    while step_best_score <= best_score:
        
        scores = []
        for mdl in best_models_list:
            oofs_probs_model = np.zeros([len(y_train), 1])
        
            sel_models = [m for m in best_models_list if m!=mdl]
            X_train_sel = X_train[:,sel_models]
            
            for fold in CFG.folds_used:
            
                train_idx = folds!=fold
                val_idx = folds==fold
            
                X_train_cv = X_train_sel[train_idx,:]
                X_val_cv = X_train_sel[val_idx,:]
                y_train_cv = y_train[train_idx]
                y_val_cv = y_train[val_idx]
            
                model = OptimizeScore_MT()
                model.fit(X_train_cv, y_train_cv)
                y_val_cv_pred = model.predict(X_val_cv)
            
                oofs_probs_model[val_idx,:] = y_val_cv_pred.reshape(-1,1)

            # Calculate score for the current model
            scores.append(
                log_loss(y_train, oofs_probs_model)
            )
        
        step_best_score = np.min(scores)

        if step_best_score <= best_score:
            # Remove the model that improves score the most
            best_models_list.remove(best_models_list[np.argmin(scores)])
            best_score = step_best_score
        else:
            break
    
    # Final names of the selected models
    best_models_list_names = list(np.array(models_list_names)[best_models_list])
    logger.info(f"Selected models for the ensemble hill descent: {best_models_list_names}")
    logger.info(f"Final score for the ensemble hill descent: {best_score:.5f}")
    
    return best_models_list_names, best_score



# Create the ensemble with bagged folds
def create_ensemble(oof_dict, sub_dict, models_list_names, seed):
        
    # List of indices of the initial candidates for the ensemble
    models_list = list(range(len(models_list_names))) 
    folds = pd.read_csv(oof_dict[models_list_names[0]])["fold"].values

    oofs_matrix = []
    subs_matrix = []

    # Create matrix with the probabilities of the models
    for m in models_list_names:
        oofs_matrix.append(
            pd.read_csv(oof_dict[m])["preds"].values
        )
        subs_matrix.append(
            pd.read_csv(sub_dict[m])[CFG.target_col].values
        )

    y_train = pd.read_csv(oof_dict[models_list_names[0]])[CFG.target_col].values
    X_train = np.stack(oofs_matrix, axis=1)
    X_test = np.stack(subs_matrix, axis=1)
    
    oofs_probs = np.zeros([len(y_train), 1])
    sub_probs = np.zeros([len(X_test), 1])

    if CFG.ensemble_mode == "folds":
        for fold in CFG.folds_used:
        
            train_idx = folds!=fold
            val_idx = folds==fold
        
            X_train_cv = X_train[train_idx,:]
            X_val_cv = X_train[val_idx,:]
            y_train_cv = y_train[train_idx]
            y_val_cv = y_train[val_idx]
        
            model = OptimizeScore_MT()
            model.fit(X_train_cv, y_train_cv)
            y_val_cv_pred = model.predict(X_val_cv)
            y_test_pred = model.predict(X_test)
            
            oofs_probs[val_idx,:] = y_val_cv_pred.reshape(-1,1)
            sub_probs += y_test_pred.reshape(-1,1) / len(CFG.folds_used)
            
            joblib.dump(model, os.path.join(CFG.models_path, f"ens_model_fold{fold}_seed{seed}.pkl"))

        score = log_loss(y_train, oofs_probs)
        logger.info(f"Final score for the ensemble: {score:.5f}")
    
    
    elif CFG.ensemble_mode == "all":
        model = OptimizeScore_MT()
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        sub_probs = y_test_pred.reshape(-1,1)
        
        joblib.dump(model, os.path.join(CFG.models_path, f"ens_model_all_seed{seed}.pkl"))
    
    return oofs_probs, sub_probs

    



if __name__ == "__main__":

    
    # Create directories
    CFG.oofs_path = os.path.join(CFG.base_path, "outs", "oofs", CFG.sim_name); os.makedirs(CFG.oofs_path, exist_ok=True)
    CFG.models_path = os.path.join(CFG.base_path, "outs", "models", CFG.sim_name); os.makedirs(CFG.models_path, exist_ok=True)
    CFG.logs_path = os.path.join(CFG.base_path, "outs", "logs", CFG.sim_name); os.makedirs(CFG.logs_path, exist_ok=True)
    CFG.subs_path = os.path.join(CFG.base_path, "outs", "subs", CFG.sim_name); os.makedirs(CFG.subs_path, exist_ok=True)
    
    
    # Load data
    df_train = pd.read_csv(os.path.join(CFG.base_path, "data", "train.csv"))
    sub = pd.read_csv(os.path.join(CFG.base_path, "data", "sample_submission.csv"))
    df_test = sub.copy()
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    
    
    # Init logger
    logger = init_logger(CFG.logs_path)
    log_cfg(logger)
    
    
    # Dictionary with the candidate models
    oof_dict = {
        42: {
            "model_8": os.path.join(CFG.base_path, "outs", "oofs", "sim_8", "oofs.csv"),
            "model_9": os.path.join(CFG.base_path, "outs", "oofs", "sim_9", "oofs.csv")
        }
    }
    sub_dict = {
        42: {
            "model_8": os.path.join(CFG.base_path, "outs", "subs", "sim_8", "submission.csv"),
            "model_9": os.path.join(CFG.base_path, "outs", "subs", "sim_9", "submission.csv")
        }
    }
    
    
    # Select members for the ensemble using hill climbing
    if CFG.do_hill_climbing:
        for seed in CFG.seeds:
            best_models_list_names, best_score = select_models_hill_climbing(oof_dict[seed])
            logger.info(f"Selected models hill climbing seed {seed}: {best_models_list_names}, score {best_score:.5f}")
    
    
    # Select members for the ensemble using hill descent
    if CFG.do_hill_descent:
        for seed in CFG.seeds:
            best_models_list_names, best_score = select_models_hill_descent(oof_dict[seed])
            logger.info(f"Selected models hill descent seed {seed}: {best_models_list_names}, score {best_score:.5f}")
    
    # Create the ensemble
    if CFG.do_create_ensemble:
        oofs_probs = np.zeros([len(df_train), 1])
        sub_probs = np.zeros([len(df_test), 1])
        
        for seed in CFG.seeds:
            oofs_probs_seed, sub_probs_seed = create_ensemble(oof_dict[seed], sub_dict[seed], CFG.best_models_list_names[seed], seed)
            
            oofs_probs += oofs_probs_seed / len(CFG.seeds)
            sub_probs += sub_probs_seed / len(CFG.seeds)
        
        # Save oofs
        oofs_df = pd.read_csv(oof_dict[seed][CFG.best_models_list_names[seed][0]])
        oofs_df["preds"] = oofs_probs
        oofs_df.to_csv(os.path.join(CFG.oofs_path, "oofs.csv"), index=False)
        final_score = log_loss(oofs_df[CFG.target_col], oofs_df["preds"])
        if CFG.ensemble_mode == "folds": logger.info(f"Final score: {final_score:.5f}")
        
        # Save submissions
        sub[CFG.target_col] = sub_probs
        sub.to_csv(os.path.join(CFG.subs_path, "submission.csv"), index=False)