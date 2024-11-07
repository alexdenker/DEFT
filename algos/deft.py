import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from ema_pytorch import EMA
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam
from tqdm import tqdm

import wandb
from htransform.likelihoods import (
    InPainting,
    NonLinearBlur,
    Radon,
    Superresolution,
    get_xi_condition,
)
from htransform.losses import epsilon_based_loss_fn_finetuning
from htransform.trainer import set_annealed_lr
from models.classifier_guidance_model import ClassifierGuidanceModel, HTransformModel
from models.utils import get_timesteps
from utils.degredations import build_degredation_model
from utils.functions import postprocess, preprocess

from .ddim import DDIM


class DEFT:
    def __init__(self, model: HTransformModel, cfg: DictConfig):
        self.model = model
        self.diffusion = model.diffusion
        self.H = build_degredation_model(cfg)
        self.cfg = cfg
        exp_root = cfg.exp.root
        exp_name = cfg.exp.name
        if cfg.wandb_config.log:
            self.log_dir = os.path.join(
                exp_root, "model_ckpts", f"{exp_name}_{cfg.run_id}"
            )
        else:
            self.log_dir = os.path.join(exp_root, "model_ckpts", exp_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        print(
            "number of parameters in pretrained model: ",
            sum([p.numel() for p in self.model.model.parameters()]),
        )
        print(
            "number of parameters in finetuned model: ",
            sum([p.numel() for p in self.model.htransform_model.parameters()]),
        )
        print(
            "Fraction: ",
            sum([p.numel() for p in self.model.htransform_model.parameters()])
            / sum([p.numel() for p in self.model.model.parameters()]),
        )
        self.device = self.model.model.device

    def train(self) -> None:
        optimizer = Adam(
            self.model.htransform_model.parameters(),
            lr=self.cfg.algo.finetune_args.lr,
        )

        self.ema = EMA(
            self.model.htransform_model,
            beta=self.cfg.algo.ema.beta,  # exponential moving average factor
            update_after_step=self.cfg.algo.ema.update_after_step,  # only after this number of .update() calls will it start updating
            update_every=self.cfg.algo.ema.update_every,  # how often to actually update, to save on compute (updates every 10th .update() call)
        )

        for epoch in tqdm(range(self.cfg.algo.finetune_args.epochs)):
            avg_loss, num_items = 0, 0
            self.model.htransform_model.train()

            # TODO: Does this shuffle dataset each epoch differently?
            for idx, batch in tqdm(
                enumerate(self.model.finetune_loader),
                total=len(self.model.finetune_loader),
            ):
                # Data coming here is in [0, 1]
                optimizer.zero_grad()
                x = batch[0]
                x = x.to(self.device)

                # Convert from [0, 1] to [-1, 1]
                x = preprocess(x)

                loss = epsilon_based_loss_fn_finetuning(
                    x=x,
                    model=self.model.htransform_model,
                    diffusion=self.diffusion,
                    pretrained_model=self.model.model,
                    likelihood=self.model.likelihood,
                    cfg=self.cfg,
                )
                loss.backward()

                if self.cfg.algo.finetune_args.get("lr_annealing", False):
                    current_iter = epoch * len(self.model.finetune_loader) + idx
                    total_iter = self.cfg.algo.finetune_args.epochs * len(
                        self.model.finetune_loader
                    )
                    set_annealed_lr(
                        optimizer,
                        self.cfg.algo.finetune_args.lr,
                        current_iter / total_iter,
                    )

                optimizer.step()
                self.ema.update()

                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]
                if idx % self.cfg.algo.finetune_args.log_freq == 0:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "step": epoch * len(self.model.finetune_loader) + idx,
                        }
                    )

            if (
                epoch % self.cfg.algo.finetune_args.save_model_every_n_epoch == 0
                or epoch == self.cfg.algo.finetune_args.epochs - 1
            ):
                if epoch == self.cfg.algo.finetune_args.epochs - 1:
                    torch.save(
                        self.model.htransform_model.module.state_dict(),
                        os.path.join(self.log_dir, "model.pt"),
                    )
                else:
                    torch.save(
                        self.model.htransform_model.module.state_dict(),
                        os.path.join(self.log_dir, f"model_{epoch}.pt"),
                    )
                # TODO: Remove
                # torch.save(
                #     self.model.htransform_model.module.state_dict(),
                #     os.path.join(self.log_dir, "model_tmp.pt"),
                # )
                # torch.save(
                #     self.ema.ema_model.module.state_dict(),
                #     os.path.join(self.log_dir, "ema_model_tmp.pt"),
                # )

            print("Average Loss: {:5f}".format(avg_loss / num_items))
            wandb.log(
                {"train/mean_loss_per_epoch": avg_loss / num_items, "step": epoch + 1}
            )

            if self.cfg.algo.val_args.sample_freq > 0:
                if epoch % self.cfg.algo.val_args.sample_freq == 0:
                    self.model.htransform_model.eval()
                    batch = next(iter(self.model.val_loader))
                    # x is in [0, 1]
                    x = batch[0]
                    y = batch[1]
                    x = x.to(self.device)
                    y = y.to(self.device)

                    # Convert from [0, 1] to [-1, 1]
                    x = preprocess(x)

                    if isinstance(self.model.likelihood, InPainting):
                        y_0, masks = self.model.likelihood.sample(
                            x,
                            deterministic_idx=torch.arange(0, x.shape[0])
                            .long()
                            .to(self.device),
                        )
                    else:
                        y_0 = self.model.likelihood.sample(x)
                        masks = None

                    ts = get_timesteps(
                        start_step=1000,
                        end_step=0,
                        num_steps=self.cfg.algo.val_args.num_steps,
                    )
                    self.sample(
                        x, y, ts, masks=masks, use_ema=True, y_0=y_0, log_images=True
                    )

        # always save last model
        torch.save(
            self.model.htransform_model.module.state_dict(),
            os.path.join(self.log_dir, "model.pt"),
        )
        torch.save(
            self.ema.ema_model.module.state_dict(),
            os.path.join(self.log_dir, "ema_model.pt"),
        )

    def sample(self, x, y, ts, **kwargs):
        """
        x: [batch_size, 3, 256, 256]
        y: [batch_size] labels of the image that we don't use, but fit reddiff API
        kwargs["y_0"]: [batch_size, 3, 256, 256] the actual degradation we consider
        """
        del y
        self.model.htransform_model.eval()
        masks = kwargs.get("masks", None)
        use_ema = kwargs.get("use_ema", False)
        y_0 = kwargs.get("y_0", None)
        log_images = kwargs.get("log_images", False)

        x = x.to(self.device)

        ATy = self.get_adjoint(x, y_0)
        with torch.no_grad():
            ts = (
                torch.arange(self.diffusion.num_diffusion_timesteps)
                .to(self.device)
                .unsqueeze(-1)
            )

            ts_scaling = self.model.htransform_model.module.get_time_scaling(ts)

        plt.figure()
        plt.plot(ts_scaling[:, 0].cpu().numpy())
        plt.title("Time scaling NN(t) * log_grad")
        wandb.log({"time_scaling": wandb.Image(plt)})
        plt.close()

        combined_model = self.create_combined_model(y_0, masks)

        if use_ema:
            combined_model_ema = self.create_combined_model(y_0, masks, use_ema=True)

        with torch.no_grad():
            guidance_model = ClassifierGuidanceModel(
                model=combined_model,
                classifier=None,
                diffusion=self.diffusion,
                cfg=None,
            )

            cfg_dict = {
                "algo": {
                    "eta": self.cfg.algo.val_args.eta,
                    "sdedit": False,
                    "cond_awd": False,
                }
            }
            conf = OmegaConf.create(cfg_dict)
            sampler = DDIM(model=guidance_model, cfg=conf)

            print("sampling w/ point estimate model")
            ts = get_timesteps(
                start_step=1000,
                end_step=0,
                num_steps=self.cfg.algo.val_args.num_steps,
            )

            sample = sampler.sample(
                x=torch.randn([y_0.shape[0], *x.shape[1:]], device=self.device),
                y=None,
                ts=ts,
            )
            sample = sample.to(self.device)

            # sample = sampler.sample(y)
            if use_ema:
                guidance_model_ema = ClassifierGuidanceModel(
                    model=combined_model_ema,
                    classifier=None,
                    diffusion=self.diffusion,
                    cfg=None,
                )
                sampler_ema = DDIM(model=guidance_model_ema, cfg=conf)
                print("sampling w/ ema")
                sample_ema = sampler_ema.sample(
                    x=torch.randn([y_0.shape[0], *x.shape[1:]], device=self.device),
                    y=None,
                    ts=ts,
                )
                sample_ema = sample_ema.to(self.device)
                sample_ema = postprocess(sample_ema)

        # Convert from [-1, 1] to [0, 1]
        sample = postprocess(sample)
        x = postprocess(x)
        y_0 = postprocess(y_0)

        if log_images:
            for i in range(sample.shape[0]):
                fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(16, 6))

                if x.shape[1] == 3:
                    # NOTE: Imshow needs images in [0, 1]
                    ax1.set_title("ground truth")
                    ax1.imshow(x[i, :, :, :].permute(1, 2, 0).cpu().detach().numpy())
                    ax1.axis("off")

                    ax2.set_title("y")
                    ax2.imshow(y_0[i, :, :, :].permute(1, 2, 0).cpu().numpy())
                    ax2.axis("off")

                    ax3.set_title("cheap_guidance")
                    ax3.imshow((ATy[i, :, :, :].permute(1, 2, 0).cpu().numpy() + 1) / 2)
                    ax3.axis("off")

                    ax4.set_title("sample")
                    ax4.imshow(sample[i, :, :, :].permute(1, 2, 0).cpu().numpy())
                    ax4.axis("off")
                    if use_ema:
                        ax5.set_title("sample_ema")
                        ax5.imshow(
                            sample_ema[i, :, :, :].permute(1, 2, 0).cpu().numpy()
                        )
                        ax5.axis("off")

                else:
                    ax1.set_title("ground truth")
                    ax1.imshow(x[i, 0, :, :].cpu().numpy(), cmap="gray")
                    ax1.axis("off")

                    ax2.set_title("y")
                    ax2.imshow(y_0[i, 0, :, :].cpu().numpy().T, cmap="gray")
                    ax2.axis("off")

                    ax3.set_title("A.T(y)")
                    ax3.imshow(ATy[i, 0, :, :].cpu().numpy(), cmap="gray")
                    ax3.axis("off")

                    ax4.set_title("sample")
                    ax4.imshow(sample[i, 0, :, :].cpu().numpy(), cmap="gray")
                    ax4.axis("off")
                    if use_ema:
                        ax5.set_title("sample ema")
                        ax5.imshow(sample_ema[i, 0, :, :].cpu().numpy(), cmap="gray")
                        ax5.axis("off")

                wandb.log({f"samples/{i}": wandb.Image(plt)})
                plt.close()

            # Calculate PSNR on validation batch, to have some metric to compare
            # NOTE: This definition of PSNR needs images in [0, 1]

            def _get_psnr(sample_, x_):
                mse = torch.mean(
                    (sample_ - x_) ** 2, dim=(1, 2, 3)
                )  # keep batch dim for now
                psnr_ = 10 * torch.log10(1 / (mse + 1e-10))

                return psnr_

            psnr = _get_psnr(sample, x)
            if use_ema:
                psnr_ema = _get_psnr(sample_ema, x)
                wandb.log(
                    {
                        "val/batch_psnr_ema": np.mean(psnr_ema.cpu().numpy()),
                    }
                )

            wandb.log(
                {
                    "val/batch_psnr": np.mean(psnr.cpu().numpy()),
                }
            )
        # We need to return the sample in the range [-1, 1]
        return preprocess(sample), None

    def get_adjoint(self, x, y):
        if isinstance(self.model.likelihood, Superresolution):
            ATy = self.model.likelihood.A_adjoint(y) * (self.model.likelihood.scale**2)
        if isinstance(self.model.likelihood, Radon):
            ATy = self.model.likelihood.fbp(y)
        if isinstance(self.model.likelihood, NonLinearBlur):
            ATy = self.model.likelihood.log_likelihood_grad(x, y)
        else:
            ATy = self.model.likelihood.A_adjoint(y)

        return ATy

    def create_combined_model(
        self,
        y,
        masks,
        use_ema=False,
    ):
        @torch.no_grad()
        def model_fn(x, t):
            eps1 = self.model.model(x, t)

            alpha_t = self.model.diffusion.alpha(t).view(-1, 1, 1, 1)

            x0hat = (x - eps1 * (1 - alpha_t).sqrt()) / alpha_t.sqrt()
            xi_condition = get_xi_condition(
                xi=x,
                x0hat=x0hat,
                y=y,
                likelihood=self.model.likelihood,
                masks=masks,
                cfg=self.cfg,
            )

            if use_ema:
                eps2 = self.ema.ema_model(xi_condition, t)
            else:
                eps2 = self.model.htransform_model(xi_condition, t)

            eps = eps1 + eps2

            return eps

        return model_fn
