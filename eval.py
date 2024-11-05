import os

import hydra
import torch
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

import wandb
from eval.ca import main as ca_runner
from eval.fid import main as fid_runner
from eval.fid_stats import main as fid_stats_runner
from eval.psnr import main as psnr_runner
from utils.distributed import init_processes


def main(cfg: DictConfig):
    # Load hydra config from save_path
    with open(os.path.join(cfg.save_path, "config.yaml"), "r") as f:
        train_cfg = OmegaConf.load(f.name)

    # Load in the config that the samples were generated with
    # Easier than respecifying all the things again in eval config

    cfg.exp.root = train_cfg.exp.root
    cfg.exp.samples_root = train_cfg.exp.samples_root
    cfg.dataset.meta_root = train_cfg.dataset.meta_root
    cfg.dataset.split = "custom"
    cfg.dataset.subset_txt = train_cfg.dataset.subset_txt

    run_id = train_cfg.get("run_id", None)
    exp_name = f"{train_cfg.exp.name}_{run_id}" if run_id else train_cfg.exp.name

    original_save_path = cfg.save_path
    results_save_path = os.path.join(cfg.results_save_path, exp_name)

    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb_kwargs = {
        "project": train_cfg.wandb_config.project,
        "entity": train_cfg.wandb_config.entity,
        "config": wandb_config,
        "name": exp_name,
        "mode": "online" if train_cfg.wandb_config.log else "disabled",
        "settings": wandb.Settings(code_dir=train_cfg.wandb_config.code_dir),
        "dir": train_cfg.wandb_config.code_dir,
        "id": run_id,
        "resume": "allow",
    }
    with wandb.init(**wandb_kwargs) as _:
        ################################ FID ########################################
        # First create the source FID stats for actual dataset if file doesn't exist
        if not cfg.source_fid_stats_path:
            print("Creating source FID stats")
            cfg.dataset.root = train_cfg.dataset.root
            cfg.dataset.split = "val"
            cfg.mean_std_stats = True
            source_fid_stats_path = os.path.join(
                results_save_path,
                "fid_stats_original",
                "imagenet256_train_mean_std",
            )
            cfg.save_path = source_fid_stats_path
            fid_stats_runner(cfg)
            cfg.source_fid_stats_path = source_fid_stats_path

        # Now calculate the FID stats for the generated samples
        print("Calculating FID stats for generated samples")
        cfg.dataset.root = original_save_path
        cfg.dataset.split = "custom"
        cfg.mean_std_stats = True
        cfg.save_path = os.path.join(
            results_save_path, "fid_stats_generated", "fid_mean_std"
        )
        fid_stats_runner(cfg)

        # Now calculate the FID
        print("Calculating FID")
        # Load in the fid config
        fid_cfg = hydra.compose(config_name="fid")
        fid_cfg.path1 = os.path.join(
            results_save_path, "fid_stats_generated", "fid_mean_std.npz"
        )
        fid_cfg.path2 = cfg.source_fid_stats_path
        fid_cfg.results = os.path.join(results_save_path, "results")
        fid_cfg.exp.root = train_cfg.exp.root
        with open_dict(fid_cfg):
            fid_cfg.cwd = cfg.cwd

        fid = fid_runner(fid_cfg)
        wandb.log({"final_stats/fid": fid})

        ################# KID #########################
        # First create the source KID stats for actual dataset if file doesn't exist
        if not cfg.source_kid_stats_path:
            cfg.dataset.root = train_cfg.dataset.root
            cfg.dataset.split = "val"
            cfg.mean_std_stats = False
            source_kid_stats_path = os.path.join(
                results_save_path,
                "kid_stats_original",
                "imagenet256_val_dgp_top1k",
            )
            cfg.save_path = source_kid_stats_path
            fid_stats_runner(cfg)
            cfg.source_kid_stats_path = source_kid_stats_path

        # Now calculate the KID stats for the generated samples
        cfg.dataset.root = original_save_path
        cfg.mean_std_stats = False
        cfg.save_path = os.path.join(
            results_save_path, "kid_stats_generated", "kid_gen_dgp"
        )
        fid_stats_runner(cfg)

        # Now calculate the FID
        # create a new config for this
        kid_cfg = hydra.compose(config_name="fid")
        kid_cfg.path1 = os.path.join(
            results_save_path, "kid_stats_generated", "kid_gen_dgp.npy"
        )
        kid_cfg.path2 = cfg.source_kid_stats_path
        kid_cfg.results = os.path.join(results_save_path, "results")
        kid_cfg.exp.root = train_cfg.exp.root
        with open_dict(kid_cfg):
            kid_cfg.cwd = cfg.cwd
        kid = fid_runner(kid_cfg)
        wandb.log({"final_stats/kid": kid})

        ###################### PSNR, SSIM, LPIPS ##########################
        # Load in the psnr config
        psnr_cfg = hydra.compose(config_name="psnr")
        psnr_cfg.exp.root = train_cfg.exp.root
        psnr_cfg.dataset1.root = train_cfg.dataset.root
        psnr_cfg.dataset2.root = original_save_path
        psnr_cfg.dataset2.split = "custom"
        psnr_cfg.results = os.path.join(results_save_path, "results")
        with open_dict(psnr_cfg):
            psnr_cfg.cwd = cfg.cwd
        psnr, ssim, lpips = psnr_runner(psnr_cfg)
        wandb.log({"final_stats/psnr": psnr})
        wandb.log({"final_stats/ssim": ssim})
        wandb.log({"final_stats/lpips": lpips})

        ###################### Top 1 Accuracy ##########################
        ca_cfg = hydra.compose(config_name="ca")
        ca_cfg.dataset.transform = "ca_cropped"
        ca_cfg.dataset.root = original_save_path
        ca_cfg.dataset.split = "custom"
        ca_cfg.results = os.path.join(results_save_path, "results")
        with open_dict(ca_cfg):
            ca_cfg.cwd = cfg.cwd
        ca = ca_runner(ca_cfg)
        wandb.log({"final_stats/ca": ca})


@hydra.main(
    version_base="1.2", config_path="_configs", config_name="eval_imagenet256_uncond"
)
def main_dist(cfg: DictConfig):
    cwd = HydraConfig.get().runtime.output_dir

    if cfg.dist.num_processes_per_node < 0:
        size = torch.cuda.device_count()
        cfg.dist.num_processes_per_node = size
    else:
        size = cfg.dist.num_processes_per_node
    if size > 1:
        num_proc_node = cfg.dist.num_proc_node
        num_process_per_node = cfg.dist.num_processes_per_node
        world_size = num_proc_node * num_process_per_node
        mp.spawn(
            init_processes,
            args=(world_size, main, cfg, cwd),
            nprocs=world_size,
            join=True,
        )
    else:
        init_processes(0, size, main, cfg, cwd)

    # print('cfg', cfg)


if __name__ == "__main__":
    main_dist()
