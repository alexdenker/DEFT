import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import torch
from ema_pytorch import EMA
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from htransform.likelihoods import (
    NonLinearBlur,
    Radon,
    Superresolution,
    get_xi_condition,
)
from htransform.losses import epsilon_based_loss_fn_finetuning
from htransform.trainer import set_annealed_lr
from models.classifier_guidance_model import ClassifierGuidanceModel
from models.utils import create_model, get_timesteps
from utils.preprocessing import inverse_data_transform

from .ddim import DDIM


class DEFT:
    def __init__(
        self,
        model_config,
        htransform_model_config,
        likelihood,
        diffusion,
        train_config,
        log_dir,
        device,
    ):
        self.samplepretrained_model = create_model(**vars(model_config.model))
        self.pretrained_model.convert_to_fp32()
        self.pretrained_model.dtype = torch.float32
        self.pretrained_model.load_state_dict(torch.load(model_config.data.model_path))
        self.pretrained_model.eval()
        self.pretrained_model.to(device)

        self.htransform_model = create_model(**vars(htransform_model_config))
        self.htransform_model.convert_to_fp32()
        self.htransform_model.dtype = torch.float32
        self.htransform_model.to(device)

        self.device = device
        self.htransform_model_config = htransform_model_config

        print(
            "number of parameters in pretrained model: ",
            sum([p.numel() for p in self.pretrained_model.parameters()]),
        )
        print(
            "number of parameters in finetuned model: ",
            sum([p.numel() for p in self.htransform_model.parameters()]),
        )
        print(
            "Fraction: ",
            sum([p.numel() for p in self.htransform_model.parameters()])
            / sum([p.numel() for p in self.pretrained_model.parameters()]),
        )

        self.diffusion = diffusion
        self.likelihood = likelihood
        self.train_config = train_config

        self.log_dir = log_dir
        self.device = device

    def train(
        self,
        train_dl: DataLoader,
        psnr_val_dl: DataLoader,
        optim_kwargs: Dict,
        val_kwargs: Dict,
        val_dl: Optional[DataLoader] = None,
    ) -> None:
        optimizer = Adam(self.htransform_model.parameters(), lr=optim_kwargs["lr"])

        self.ema = EMA(
            self.htransform_model,
            beta=0.9999,  # exponential moving average factor
            update_after_step=500,  # only after this number of .update() calls will it start updating
            update_every=10,  # how often to actually update, to save on compute (updates every 10th .update() call)
        )

        for epoch in tqdm(range(optim_kwargs["epochs"])):
            avg_loss, num_items = 0, 0
            self.htransform_model.train()

            # TODO: Does this shuffle dataset each epoch differently?
            for idx, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
                optimizer.zero_grad()
                x = batch
                x = x.to(self.device)

                loss = epsilon_based_loss_fn_finetuning(
                    x=x,
                    model=self.htransform_model,
                    diffusion=self.diffusion,
                    pretrained_model=self.pretrained_model,
                    likelihood=self.likelihood,
                    cfg_model=self.cfg_model,
                )
                loss.backward()

                if optim_kwargs.get("lr_annealing", False):
                    current_iter = epoch * len(train_dl) + idx
                    total_iter = optim_kwargs["epochs"] * len(train_dl)
                    set_annealed_lr(
                        optimizer, optim_kwargs["lr"], current_iter / total_iter
                    )

                optimizer.step()
                self.ema.update()

                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]
                if idx % optim_kwargs["log_freq"] == 0:
                    wandb.log(
                        {"train/loss": loss.item(), "step": epoch * len(train_dl) + idx}
                    )

            if (epoch % optim_kwargs["save_model_every_n_epoch"]) == 0 or (
                epoch == optim_kwargs["epochs"] - 1
            ):
                if epoch == optim_kwargs["epochs"] - 1:
                    torch.save(
                        self.htransform_model.state_dict(),
                        os.path.join(self.log_dir, "model.pt"),
                    )
                else:
                    torch.save(
                        self.htransform_model.state_dict(),
                        os.path.join(self.log_dir, f"model_{epoch}.pt"),
                    )

                torch.save(
                    self.htransform_model.state_dict(),
                    os.path.join(self.log_dir, "model_tmp.pt"),
                )
                torch.save(
                    self.ema.ema_model.state_dict(),
                    os.path.join(self.log_dir, "ema_model_tmp.pt"),
                )

            print("Average Loss: {:5f}".format(avg_loss / num_items))
            wandb.log(
                {"train/mean_loss_per_epoch": avg_loss / num_items, "step": epoch + 1}
            )

            # if val_kwargs["sample_freq"] > 0:
            #     if epoch % val_kwargs["sample_freq"] == 0:
            #         self.sample(x, y, masks, cfg_model)

        # always save last model
        # torch.save(
        #     self.htransform_model.state_dict(), os.path.join(log_dir, "model.pt")
        # )
        # torch.save(
        #     self.ema.ema_model.state_dict(), os.path.join(log_dir, "ema_model.pt")
        # )

        return self.htransform_model, self.ema

    def sample(self, x, y, masks, val_kwargs):
        self.htransform_model.eval()

        # x = next(iter(val_dl))
        # x = x.to(device)

        # if isinstance(self.likelihood, InPainting):
        #     y, masks = self.likelihood.sample(
        #         x,
        #         deterministic_idx=torch.arange(0, x.shape[0]).long().to(device),
        #     )
        # else:
        #     y = likelihood.sample(x)
        #     masks = None
        ATy = self.get_adjoint(x, y)
        with torch.no_grad():
            ts = (
                torch.arange(self.diffusion.num_diffusion_timesteps)
                .to(self.device)
                .unsqueeze(-1)
            )

            ts_scaling = self.htransform_model.get_time_scaling(ts)

        plt.figure()
        plt.plot(ts_scaling[:, 0].cpu().numpy())
        plt.title("Time scaling NN(t) * log_grad")
        wandb.log({"time_scaling": wandb.Image(plt)})
        plt.close()

        combined_model = self.create_combined_model(y, masks)

        combined_model_ema = self.create_combined_model(y, masks, use_ema=True)

        with torch.no_grad():
            guidance_model = ClassifierGuidanceModel(
                model=combined_model,
                classifier=None,
                diffusion=self.diffusion,
                cfg=None,
            )

            guidance_model_ema = ClassifierGuidanceModel(
                model=combined_model_ema,
                classifier=None,
                diffusion=self.diffusion,
                cfg=None,
            )

            cfg_dict = {
                "algo": {
                    "eta": val_kwargs["eta"],
                    "sdedit": False,
                    "cond_awd": False,
                }
            }
            conf = OmegaConf.create(cfg_dict)
            sampler = DDIM(model=guidance_model, cfg=conf)
            sampler_ema = DDIM(model=guidance_model_ema, cfg=conf)

            print("sampling w/ point estimate model")
            ts = get_timesteps(
                start_step=1000,
                end_step=0,
                num_steps=val_kwargs["num_steps"],
            )

            sample = sampler.sample(
                x=torch.randn([y.shape[0], *x.shape[1:]], device=self.device),
                y=None,
                ts=ts,
            )
            sample = sample.to(self.device)

            # sample = sampler.sample(y)
            print("sampling w/ ema")
            sample_ema = sampler_ema.sample(
                x=torch.randn([y.shape[0], *x.shape[1:]], device=self.device),
                y=None,
                ts=ts,
            )
            sample_ema = sample_ema.to(self.device)
            # sample_ema = sampler_ema.sample(y)

        for i in range(sample.shape[0]):
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(16, 6))

            if x.shape[1] == 3:
                ax1.set_title("ground truth")
                ax1.imshow(
                    (x[i, :, :, :].permute(1, 2, 0).cpu().detach().numpy() + 1.0) / 2.0
                )
                ax1.axis("off")

                ax2.set_title("y")
                ax2.imshow((y[i, :, :, :].permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0)
                ax2.axis("off")

                ax3.set_title("cheap_guidance")
                ax3.imshow((ATy[i, :, :, :].permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0)
                ax3.axis("off")

                ax4.set_title("sample")
                ax4.imshow(
                    (sample[i, :, :, :].permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0
                )
                ax4.axis("off")

                ax5.set_title("sample_ema")
                ax5.imshow(
                    (sample_ema[i, :, :, :].permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0
                )
                ax5.axis("off")

            else:
                ax1.set_title("ground truth")
                ax1.imshow(x[i, 0, :, :].cpu().numpy(), cmap="gray")
                ax1.axis("off")

                ax2.set_title("y")
                ax2.imshow(y[i, 0, :, :].cpu().numpy().T, cmap="gray")
                ax2.axis("off")

                ax3.set_title("A.T(y)")
                ax3.imshow(ATy[i, 0, :, :].cpu().numpy(), cmap="gray")
                ax3.axis("off")

                ax4.set_title("sample")
                ax4.imshow(sample[i, 0, :, :].cpu().numpy(), cmap="gray")
                ax4.axis("off")

                ax5.set_title("sample ema")
                ax5.imshow(sample_ema[i, 0, :, :].cpu().numpy(), cmap="gray")
                ax5.axis("off")

            wandb.log({f"samples/{i}": wandb.Image(plt)})
            plt.close()

        # Calculate PSNR on validation batch, to have some metric to compare
        # PSNR needs images in [-1, 1]
        sample = inverse_data_transform(sample)
        sample_ema = inverse_data_transform(sample_ema)
        x = inverse_data_transform(x)

        def _get_psnr(sample_, x_):
            mse = torch.mean(
                (sample_ - x_) ** 2, dim=(1, 2, 3)
            )  # keep batch dim for now
            psnr_ = 10 * torch.log10(1 / (mse + 1e-10))

            return psnr_

        psnr = _get_psnr(sample, x)
        psnr_ema = _get_psnr(sample_ema, x)

        # wandb.log(
        #     {
        #         "val/batch_psnr": np.mean(psnr.cpu().numpy()),
        #         "step": epoch + 1,
        #     }
        # )
        # wandb.log(
        #     {
        #         "val/batch_psnr_ema": np.mean(psnr_ema.cpu().numpy()),
        #         "step": epoch + 1,
        #     }
        # )

    def get_adjoint(self, x, y):
        if isinstance(self.likelihood, Superresolution):
            ATy = self.likelihood.A_adjoint(y) * (self.likelihood.scale**2)
        if isinstance(self.likelihood, Radon):
            ATy = self.likelihood.fbp(y)
        if isinstance(self.likelihood, NonLinearBlur):
            ATy = self.likelihood.log_likelihood_grad(x, y)
        else:
            ATy = self.likelihood.A_adjoint(y)

        return ATy

    def create_combined_model(
        self,
        y,
        masks,
        use_ema=False,
    ):
        @torch.no_grad()
        def model_fn(x, t):
            eps1 = self.pretrained_model(x, t)

            alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)

            x0hat = (x - eps1 * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

            xi_condition = get_xi_condition(
                xi=x,
                x0hat=x0hat,
                y=y,
                likelihood=self.likelihood,
                masks=masks,
                cfg_model=self.htransform_model_config,
            )

            if use_ema:
                eps2 = self.ema.ema_model(xi_condition, t)
            else:
                eps2 = self.htransform_model(xi_condition, t)

            eps = eps1 + eps2

            return eps

        return model_fn
