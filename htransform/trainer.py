import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from ema_pytorch import EMA
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from algos.ddim import DDIM
from htransform.likelihoods import (
    InPainting,
    Likelihood,
    NonLinearBlur,
    Radon,
    Superresolution,
    get_xi_condition,
)
from htransform.losses import epsilon_based_loss_fn_finetuning
from models.classifier_guidance_model import ClassifierGuidanceModel
from models.utils import get_timesteps


def inverse_data_transform(x_, rescale=True):
    # Confirm that the image being passed in is in the range -1, 1
    # assert x_.min() >= -1.0 and x_.max() <= 1.0 and x_.min() < 0.0
    if rescale:
        x_ = (x_ + 1.0) / 2.0

    return x_


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def htransform_trainer(
    finetuned_score,
    pretrained_model,
    diffusion,
    train_dl: DataLoader,
    psnr_val_dl,
    optim_kwargs: Dict,
    val_kwargs: Dict,
    likelihood: Likelihood,
    cfg_train,
    cfg_model,
    val_dl: Optional[DataLoader] = None,
    device: Optional[Any] = None,
    log_dir: str = "./",
) -> None:
    """
    This class is responsible for fine-tuning h-transform models. It leverages a pretrained
    unconditional score model and fine-tunes it using a specified dataset and training parameters.

    Parameters:
    - finetuned_score (h-transform model): The new h-transform model to be finetuned.
    - pretrained_model (unconditional score model): The pretrained unconditional score model.
    - diffusion (forward sde): The forward SDE used for the diffusion process.
    - train_dl (DataLoader): DataLoader for the fine-tuning dataset.

    Optimization Arguments (optim_kwargs):
    - lr (float): The learning rate for the optimizer.
    - train_pretrain (bool): Flag indicating whether the unconditional model should be re-trained.
    - epochs (int): The total number of training epochs.
    - log_freq (int): The frequency (in epochs) at which to log the training loss to wandb.
    - save_model_every_n_epoch (int): The frequency (in epochs) at which to save the current
      h-transform model.

    Validation Arguments (val_kwargs):
    - sample_freq (int): The frequency (in epochs) at which to draw and save samples to wandb.
    - num_steps (int): The number of DDIM sampling steps.
    - eta (float): The eta value for DDIM sampling.
    """

    print(cfg_model)

    optimizer = Adam(finetuned_score.parameters(), lr=optim_kwargs["lr"])

    finetuned_score.to(device)

    ema = EMA(
        finetuned_score,
        beta=0.9999,  # exponential moving average factor
        update_after_step=500,  # only after this number of .update() calls will it start updating
        update_every=10,  # how often to actually update, to save on compute (updates every 10th .update() call)
    )

    pretrained_model.to(device)

    for epoch in tqdm(range(optim_kwargs["epochs"])):
        avg_loss, num_items = 0, 0
        finetuned_score.train()

        # TODO: Does this shuffle dataset each epoch differently?
        for idx, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
            optimizer.zero_grad()
            x = batch
            x = x.to(device)

            loss = epsilon_based_loss_fn_finetuning(
                x=x,
                model=finetuned_score,
                diffusion=diffusion,
                pretrained_model=pretrained_model,
                likelihood=likelihood,
                cfg_model=cfg_model,
            )
            loss.backward()

            if optim_kwargs.get("lr_annealing", False):
                current_iter = epoch * len(train_dl) + idx
                total_iter = optim_kwargs["epochs"] * len(train_dl)
                set_annealed_lr(
                    optimizer, optim_kwargs["lr"], current_iter / total_iter
                )

            optimizer.step()
            ema.update()

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
                    finetuned_score.state_dict(), os.path.join(log_dir, "model.pt")
                )
            else:
                torch.save(
                    finetuned_score.state_dict(),
                    os.path.join(log_dir, f"model_{epoch}.pt"),
                )

            torch.save(
                finetuned_score.state_dict(), os.path.join(log_dir, "model_tmp.pt")
            )
            torch.save(
                ema.ema_model.state_dict(), os.path.join(log_dir, "ema_model_tmp.pt")
            )

        print("Average Loss: {:5f}".format(avg_loss / num_items))
        wandb.log(
            {"train/mean_loss_per_epoch": avg_loss / num_items, "step": epoch + 1}
        )

        if val_kwargs["sample_freq"] > 0:
            if epoch % val_kwargs["sample_freq"] == 0:
                finetuned_score.eval()

                x = next(iter(val_dl))
                x = x.to(device)

                if isinstance(likelihood, InPainting):
                    y, masks = likelihood.sample(
                        x,
                        deterministic_idx=torch.arange(0, x.shape[0]).long().to(device),
                    )
                else:
                    y = likelihood.sample(x)
                    masks = None

                if isinstance(likelihood, Superresolution):
                    ATy = likelihood.A_adjoint(y) * (likelihood.scale**2)
                if isinstance(likelihood, Radon):
                    ATy = likelihood.fbp(y)
                if isinstance(likelihood, NonLinearBlur):
                    ATy = likelihood.log_likelihood_grad(x, y)
                else:
                    ATy = likelihood.A_adjoint(y)

                with torch.no_grad():
                    ts = (
                        torch.arange(diffusion.num_diffusion_timesteps)
                        .to(device)
                        .unsqueeze(-1)
                    )

                    ts_scaling = finetuned_score.get_time_scaling(ts)

                plt.figure()
                plt.plot(ts_scaling[:, 0].cpu().numpy())
                plt.title("Time scaling NN(t) * log_grad")
                wandb.log({"time_scaling": wandb.Image(plt)})
                plt.close()

                @torch.no_grad()
                def combined_model(x, t):
                    # print("t : ", t)
                    eps1 = pretrained_model(x, t)

                    alpha_t = diffusion.alpha(t).view(-1, 1, 1, 1)

                    x0hat = (x - eps1 * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

                    xi_condition = get_xi_condition(
                        xi=x,
                        x0hat=x0hat,
                        y=y,
                        likelihood=likelihood,
                        masks=masks,
                        cfg_model=cfg_model,
                    )
                    eps2 = finetuned_score(xi_condition, t)

                    eps = eps1 + eps2

                    return eps

                @torch.no_grad()
                def combined_model_ema(x, t):
                    eps1 = pretrained_model(x, t)

                    alpha_t = diffusion.alpha(t).view(-1, 1, 1, 1)

                    x0hat = (x - eps1 * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

                    xi_condition = get_xi_condition(
                        xi=x,
                        x0hat=x0hat,
                        y=y,
                        likelihood=likelihood,
                        masks=masks,
                        cfg_model=cfg_model,
                    )
                    eps2 = ema.ema_model(xi_condition, t)

                    eps = eps1 + eps2

                    return eps

                with torch.no_grad():
                    guidance_model = ClassifierGuidanceModel(
                        model=combined_model,
                        classifier=None,
                        diffusion=diffusion,
                        cfg=None,
                    )

                    guidance_model_ema = ClassifierGuidanceModel(
                        model=combined_model_ema,
                        classifier=None,
                        diffusion=diffusion,
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
                        start_step=1000, end_step=0, num_steps=val_kwargs["num_steps"]
                    )

                    sample = sampler.sample(
                        x=torch.randn([y.shape[0], *x.shape[1:]], device=device),
                        y=None,
                        ts=ts,
                    )
                    sample = sample.to(device)

                    # sample = sampler.sample(y)
                    print("sampling w/ ema")
                    sample_ema = sampler_ema.sample(
                        x=torch.randn([y.shape[0], *x.shape[1:]], device=device),
                        y=None,
                        ts=ts,
                    )
                    sample_ema = sample_ema.to(device)
                    # sample_ema = sampler_ema.sample(y)

                for i in range(sample.shape[0]):
                    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(16, 6))

                    if x.shape[1] == 3:
                        ax1.set_title("ground truth")
                        ax1.imshow(
                            (
                                x[i, :, :, :].permute(1, 2, 0).cpu().detach().numpy()
                                + 1.0
                            )
                            / 2.0
                        )
                        ax1.axis("off")

                        ax2.set_title("y")
                        ax2.imshow(
                            (y[i, :, :, :].permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0
                        )
                        ax2.axis("off")

                        ax3.set_title("cheap_guidance")
                        ax3.imshow(
                            (ATy[i, :, :, :].permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0
                        )
                        ax3.axis("off")

                        ax4.set_title("sample")
                        ax4.imshow(
                            (sample[i, :, :, :].permute(1, 2, 0).cpu().numpy() + 1.0)
                            / 2.0
                        )
                        ax4.axis("off")

                        ax5.set_title("sample_ema")
                        ax5.imshow(
                            (
                                sample_ema[i, :, :, :].permute(1, 2, 0).cpu().numpy()
                                + 1.0
                            )
                            / 2.0
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

                wandb.log(
                    {"val/batch_psnr": np.mean(psnr.cpu().numpy()), "step": epoch + 1}
                )
                wandb.log(
                    {
                        "val/batch_psnr_ema": np.mean(psnr_ema.cpu().numpy()),
                        "step": epoch + 1,
                    }
                )

    # always save last model
    torch.save(finetuned_score.state_dict(), os.path.join(log_dir, "model.pt"))
    torch.save(ema.ema_model.state_dict(), os.path.join(log_dir, "ema_model.pt"))

    return finetuned_score, ema
