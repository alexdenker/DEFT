import ml_collections


# We specify python file.py --config default_config.py:model_trained_on:dname:task
def get_config(config_str):
    extra_args = config_str.split(":")
    print("CONFIG STRING: ", config_str)
    if extra_args[0] == "training":
        assert len(extra_args) == 2
        task, dname = extra_args[0], extra_args[-1]
        model_trained_on = dname
        task = "training"
        inversion_task = None
    elif extra_args[0] == "finetuning":
        assert len(extra_args) == 3
        task, dname, inversion_task = extra_args[0], extra_args[1], extra_args[2]
        model_trained_on = dname
        task = "finetuning"
        inversion_task = inversion_task
    elif extra_args[0] == "cond_sampling":
        assert len(extra_args) == 3
        task, dname, inversion_task = extra_args[0], extra_args[1], extra_args[2]
        model_trained_on = dname
        task = "cond_sampling"
        inversion_task = inversion_task
    print(f"Task : {task}, Model trained on : {model_trained_on}, Dataset : {dname}")

    if task == "finetuning" or task == "cond_sampling":
        print(f"Inversion task: {inversion_task}")

    config = ml_collections.ConfigDict()

    config.model_trained_on = model_trained_on
    config.inversion_task = inversion_task
    config.device = "cuda"
    config.seed = 1

    # Set some eval config flags only used in eval
    config.run_id = ""

    # Set tasks-specific configs
    set_task_configs(config, task, model_trained_on, dname)

    config.forward_op = ml_collections.ConfigDict()

    if task == "finetuning" or task == "cond_sampling":
        config.forward_op.noise_std = 0.1
        if inversion_task == "ct":
            config.forward_op.num_angles = 60
        elif inversion_task == "sr":
            config.forward_op.scale = 8
        elif inversion_task == "inp":
            config.forward_op.min_mask_size = 60
            config.forward_op.max_mask_size = 120
            config.forward_op.mask_filename = (
                "imagenet_freeform_masks.npz"
            )
        elif inversion_task == "phase_retrieval":
            config.forward_op.oversample = 2.0
            config.forward_op.noise_std = 0.0
        elif inversion_task == "blur":
            config.forward_op.opt_yml_path = "./bkse/options/generate_blur/default.yml"
        elif inversion_task == "hdr":
            pass
        else:
            print("No inversion task found for finetuning!")

    # training configs
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 4
    training.epochs = 200
    training.log_freq = 1
    training.lr = 1e-4
    training.ema_decay = 0.999
    training.ema_warm_start_steps = (
        400  # only start updating ema after this amount of steps
    )
    training.save_model_every_n_epoch = 10
    training.lr_annealing = False

    training.loss_weight = "no"
    training.sigma_data_sq = 0.5

    # validation configs
    config.validation = validation = ml_collections.ConfigDict()
    validation.batch_size = 10
    validation.psnr_batch_size = 100
    validation.num_steps = 100
    validation.eta = 0.0
    validation.sample_freq = 5  # 0 = NO VALIDATION SAMPLES DURING TRAINING
    validation.psnr_sample_freq = -1
    validation.use_ema = False
    validation.perform_eval = False
    # data configs - specify in other configs
    config.data = ml_collections.ConfigDict()

    # Wandb Configs
    config.wandb = ml_collections.ConfigDict()
    config.wandb.log = False
    config.wandb.log_artifact = False
    config.wandb.project = "diff-models_" + task
    config.wandb.entity = ""
    config.wandb.code_dir = "./" 
    config.wandb.name = ""

    return config


def set_task_configs(config, task_name, model_trained_on, dname):
    
    if task_name == "finetuning":
        config.train_model_on = dname
        config.sde = "ddpm"
        config.model_type = "mlp"
        config.base_path = (
            "finetuning/"
            + config.inversion_task  
        )
        config.ckpt_path = ""
        
        config.image_path = "test"

        config.finetune = finetune = ml_collections.ConfigDict()

        config.finetune_model_config = finetune_model = ml_collections.ConfigDict()
        finetune_model.type = "openai"
        finetune_model.in_channels = 2
        finetune_model.out_channels = 3
        finetune_model.num_channels = 64
        finetune_model.num_heads = 4
        finetune_model.num_res_blocks = 1
        finetune_model.attention_resolutions = "32,16"
        finetune_model.channel_mult = ""
        finetune_model.dropout = 0.0
        finetune_model.resamp_with_conv = True
        finetune_model.learn_sigma = False
        finetune_model.use_scale_shift_norm = True
        finetune_model.use_fp16 = False
        finetune_model.resblock_updown = True
        finetune_model.num_heads_upsample = -1
        finetune_model.var_type = "fixedsmall"
        finetune_model.num_head_channels = 16
        finetune_model.image_size = 256
        finetune_model.use_new_attention_order = False

        finetune_model.cond_model = True
        finetune_model.use_residual = False
        finetune_model.epsilon_based = True
        finetune_model.init_scaling_bias = 0.01
        finetune_model.num_training_pts = 1000

        finetune_model.use_x0hat = True 
        finetune_model.use_loggrad = True

        config.dataset = dataset = ml_collections.ConfigDict()
        dataset.name = dname
        dataset.root = "/home/alexdenker/ImageNet"
        dataset.meta_root = "_data"
        dataset.subset_txt = (
            "datasets/data/imagenet_10k.txt"
        )
        dataset.val_subset_txt = "datasets/data/imagenet_1k_val.txt"
        dataset.use_default_loader = False

    if task_name == "cond_sampling":
        config.train_model_on = dname

        config.dataset = dataset = ml_collections.ConfigDict()
        dataset.name = dname
        dataset.root = "" 
        dataset.meta_root = "_data"
        dataset.subset_txt = (
            "datasets/data/imagenet_10k.txt" 
        )
        dataset.use_default_loader = False
