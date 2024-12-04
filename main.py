# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

import datetime
import os
import shutil
import time

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


import wandb
from algos import build_algo
from datasets import build_loader
from htransform.likelihoods import get_likelihood, GaussianLikelihood

from models import build_model
from models.classifier_guidance_model import ClassifierGuidanceModel, HTransformModel
from models.diffusion import Diffusion
from utils.degredations import get_degreadation_image, build_degredation_model
from utils.distributed import common_init, get_logger, init_processes
from utils.functions import get_timesteps, postprocess, preprocess, strfdt
from utils.save import save_result

torch.set_printoptions(sci_mode=False)


# import sys
# sys.path.append('/lustre/fsw/nvresearch/mmardani/source/latent-diffusion-sampling/pgdm')
# print(sys.path)
# import pdb; pdb.set_trace()


def main(cfg):
    if cfg.htransform_model.ckpt_path is None:
        run_id = wandb.util.generate_id()
    else:
        id_str = cfg.htransform_model.ckpt_path.split("/")[-2]
        run_id = id_str.split("_")[-1]
    cfg.run_id = run_id
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb_kwargs = {
        "project": cfg.wandb_config.project,
        "entity": cfg.wandb_config.entity,
        "config": wandb_config,
        "name": f"{cfg.exp.name}_{run_id}",
        "mode": "online" if cfg.wandb_config.log else "disabled",
        "settings": wandb.Settings(code_dir=cfg.wandb_config.code_dir),
        "dir": cfg.wandb_config.code_dir,
        "id": run_id,
        "resume": "allow",
    }
    with wandb.init(**wandb_kwargs) as _:
        print("cfg.exp.seed", cfg.exp.seed)
        common_init(dist.get_rank(), seed=cfg.exp.seed)
        torch.cuda.set_device(dist.get_rank())

        logger = get_logger(name="main", cfg=cfg)
        if cfg.wandb_config.log:
            exp_name = f"{cfg.exp.name}_{run_id}"
        else:
            exp_name = cfg.exp.name
        logger.info(f"Experiment name is {exp_name}")
        exp_root = cfg.exp.root
        samples_root = cfg.exp.samples_root
        samples_root = os.path.join(exp_root, samples_root, exp_name)

        dataset_name = cfg.dataset.name
        if dist.get_rank() == 0:
            if cfg.exp.overwrite and cfg.htransform_model.ckpt_path is None:
                if os.path.exists(samples_root):
                    shutil.rmtree(samples_root)
                os.makedirs(samples_root)
            else:
                if not os.path.exists(samples_root):
                    os.makedirs(samples_root)

        # Save the hydra config
        with open(os.path.join(samples_root, "config.yaml"), "w") as f:
            OmegaConf.save(config=cfg, f=f.name)
        model, classifier, htransform_model = build_model(cfg)
        model.eval()
        if classifier is not None:
            classifier.eval()
        loader = build_loader(cfg)
        logger.info(f"Dataset size is {len(loader.dataset)}")

        diffusion = Diffusion(**cfg.diffusion)

        H = build_degredation_model(cfg)
        if "deft" in cfg.algo.name:
            likelihood = GaussianLikelihood(H=H, sigma_y=cfg.algo.sigma_y) 

            cg_model = HTransformModel(
                model, htransform_model, classifier, diffusion, likelihood, cfg
            )
        else:
            cg_model = ClassifierGuidanceModel(model, classifier, diffusion, cfg)

        algo = build_algo(cg_model, cfg)
        if (
            "ddrm" in cfg.algo.name
            or "mcg" in cfg.algo.name
            or "dps" in cfg.algo.name
            or "pgdm" in cfg.algo.name
            or "reddiff" in cfg.algo.name
            or "deft" in cfg.algo.name
        ):
            H = algo.H

        ########################## DO FINETUNING IF NEEDED ##########
        print("htransform checkpoint path: ", cfg.htransform_model.ckpt_path)
        if cfg.algo.name == "deft" and cfg.htransform_model.ckpt_path is None:
            algo.train()

        ########################## DO EVAL ##########################
        psnrs = []
        start_time = time.time()
        for it, (x, y, info) in tqdm(enumerate(loader)):
            if cfg.exp.smoke_test > 0 and it >= cfg.exp.smoke_test:
                break
            # Images are in [0, 1]
            # y here is the label of imagenet that class_cond models occasionally need.
            x, y = x.cuda(), y.cuda()

            # Convert from [0, 1] to [-1, 1]
            x = preprocess(x)
            ts = get_timesteps(cfg)

            kwargs = info
            # TODO: Can we combine the likelihood forward pass for all algorithms?
            if (
                "ddrm" in cfg.algo.name
                or "mcg" in cfg.algo.name
                or "dps" in cfg.algo.name
                or "pgdm" in cfg.algo.name
                or "reddiff" in cfg.algo.name
            ):
                idx = info["index"]
                if "inp" in cfg.algo.deg or "in2" in cfg.algo.deg:  # what is in2?
                    H.set_indices(idx)
                y_0 = H.H(x)

                # This is to account for scaling to [-1, 1]
                # y_0 is the degradation that we consider
                print("ALGO SIGMA Y : ", cfg.algo.sigma_y)
                y_0 = (
                    y_0 + torch.randn_like(y_0) * cfg.algo.sigma_y * 2
                )  # ?? what is it for???
                kwargs["y_0"] = y_0
            elif "deft" in cfg.algo.name:
                # TODO: Use algo.sigma_y instead of forward_op.noise_std
                # TODO: remove likelihood configs entirely, specify in algo.deg_args.
                if "inp" in cfg.algo.deg or "in2" in cfg.algo.deg:
                    y_0, masks = algo.model.likelihood.sample(
                        x,
                        deterministic_idx=torch.arange(0, x.shape[0])
                        .long()
                        .to(algo.device),
                    )
                else:
                    y_0 = algo.model.likelihood.sample(x)
                    masks = None
                kwargs["masks"] = masks
                kwargs["y_0"] = y_0
                kwargs["use_ema"] = cfg.algo.val_args.use_ema

            # pgdm
            if cfg.exp.save_evolution:
                if cfg.algo.name == "deft":
                    raise NotImplementedError("DEFT does not support evolution saving")
                xt_s, _, xt_vis, _, mu_fft_abs_s, mu_fft_ang_s = algo.sample(
                    x, y, ts, **kwargs
                )
            else:
                xt_s, _ = algo.sample(x, y, ts, **kwargs)

            # visualiztion of steps
            if cfg.exp.save_evolution:
                # Convert from [-1, 1] to [0, 1] for plotting
                xt_vis = postprocess(xt_vis).cpu()
                print("torch.max(mu_fft_abs_s)", torch.max(mu_fft_abs_s))
                print("torch.min(mu_fft_abs_s)", torch.min(mu_fft_abs_s))
                print("torch.max(mu_fft_ang_s)", torch.max(mu_fft_ang_s))
                print("torch.min(mu_fft_ang_s)", torch.min(mu_fft_ang_s))
                mu_fft_abs = torch.log(mu_fft_abs_s + 1)
                mu_fft_ang = mu_fft_ang_s  # torch.log10(mu_fft_abs_s+1)
                mu_fft_abs = (mu_fft_abs - torch.min(mu_fft_abs)) / (
                    torch.max(mu_fft_abs) - torch.min(mu_fft_abs)
                )
                mu_fft_ang = (mu_fft_ang - torch.min(mu_fft_ang)) / (
                    torch.max(mu_fft_ang) - torch.min(mu_fft_ang)
                )
                xx = torch.cat((xt_vis, mu_fft_abs, mu_fft_ang), dim=2)
                save_result(dataset_name, xx, y, info, samples_root, "evol")

            # timing
            # start_time_sample = time.time()
            # finish_time_sample = time.time() - start_time
            # print('cfg.loader.batch_size', cfg.loader.batch_size)
            # print('cfg.exp.num_steps', cfg.exp.num_steps)
            # time_per_sample = finish_time_sample/(cfg.exp.num_steps*cfg.loader.batch_size)
            # print('time_per_sample', time_per_sample)
            # import pdb; pdb.set_trace()

            if isinstance(xt_s, list):
                # Convert from [-1, 1] to [0, 1] for PSNR calculation
                xo = postprocess(xt_s[0], clamp=cfg.algo.name != "deft").cpu()
            else:
                xo = postprocess(xt_s, clamp=cfg.algo.name != "deft").cpu()

            save_result(dataset_name, xo, y, info, samples_root, "")

            # This definition of PSNR needs images in [0, 1]
            mse = torch.mean((xo - postprocess(x).cpu()) ** 2, dim=(1, 2, 3))
            psnr = 10 * torch.log10(1 / (mse + 1e-10))
            psnrs.append(psnr)

            if cfg.exp.save_deg:
                # TODO: x0 using DEFT likelihood
                xo = postprocess(get_degreadation_image(y_0, H, cfg))

                save_result(dataset_name, xo, y, info, samples_root, "deg")

            if cfg.exp.save_ori:
                xo = postprocess(x)
                save_result(dataset_name, xo, y, info, samples_root, "ori")

            if it % cfg.exp.logfreq == 0 or cfg.exp.smoke_test > 0 or it < 10:
                now = time.time() - start_time
                now_in_hours = strfdt(datetime.timedelta(seconds=now))
                future = (len(loader) - it - 1) / (it + 1) * now
                future_in_hours = strfdt(datetime.timedelta(seconds=future))
                logger.info(
                    f"Iter {it}: {now_in_hours} has passed, expect to finish in {future_in_hours}"
                )

        if len(loader) > 0:
            psnrs = torch.cat(psnrs, dim=0)
            logger.info(f"PSNR: {psnrs.mean().item()}")
            wandb.run.log({"psnr": psnrs.mean().item()})

        logger.info("Done.")
        now = time.time() - start_time
        now_in_hours = strfdt(datetime.timedelta(seconds=now))
        logger.info(f"Total time: {now_in_hours}")
        wandb.run.log({"total_time": now_in_hours})

        wandb.run.log_code(
            cfg.wandb_config.code_dir,
            include_fn=lambda path: path.endswith(".py")
            or path.endswith(".ipynb")
            or path.endswith(".yaml"),
        )


@hydra.main(version_base="1.2", config_path="_configs", config_name="deft")
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


if __name__ == "__main__":
    main_dist()
