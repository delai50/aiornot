class CFG:
    base_path = "../aiornot"
    sim_name = "sim_ens_1"
    target_col = "label"
    folds_used = [0,1,2,3,4]
    seeds = [42]
    do_hill_climbing = False
    do_hill_descent = False
    do_create_ensemble = True
    ensemble_mode = "all" # "folds", "all"
    best_models_list_names = {
            42: ["model_1", "model_2"], 
            }