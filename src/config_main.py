# CONFIG FILE
class CFG:
    # General parameters
    webhook = "YOUR_WEBHOOK"
    base_path = "../aiornot"
    sim_name = "sim_1"
    target_col = "label"
    output_classes = 1
    seed = 42
    n_folds = 5
    folds_used = [0,1,2,3,4]
    debug = False
    parser = True
    opt_mode = "min"
    
    # Task mode
    do_hpoptimization = False
    hopt_timeout = 60*60*12
    do_crossvalidation = True
    oofs_mode = "last" # "best", "last"
    do_full_train = False
    do_inference = True
    infer_mode = "fold" # "all", "fold"
    do_pseudolabeling = False
    pseudo_prob = 0.
    draw_curves = True
    log = True
    
    # NN params
    model_name = "tf_efficientnetv2_b0"
    pretrained = True
    img_size = (224,224)
    
    # Training params
    optimizer = "AdamW"
    label_smoothing = 0.
    epochs = 10
    lr = 5e-4
    # lr_backbone = 3e-4 # $
    # lr_head = 3e-4 # $
    batch_size = 32
    wd = 1e-2 # 1e-3
    accumulate_grad_batches = 1
    # gradient_clip_val = gradient_clip_val
    val_check_interval = 1.
    es_patience = 100 # 5 * 1/val_check_interval
    monitor = "val_ove_metric"
    
    # Augmentations
    augmentation_group = "custom_soft" 
    mixup = False
    mixup_p = 0.5
    mixup_alpha = 0.5
    cutmix = True
    cutmix_p = 0.5
    cutmix_alpha = 0.5
    tta = False
    # Other parameters
    pin_memory = True
    num_workers = 8
    precision = 16
    accelerator = "gpu"
    gpus = [0]