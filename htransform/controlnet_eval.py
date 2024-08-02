import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.metrics import peak_signal_noise_ratio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

import wandb
from algos.ddim import DDIM
from htransform.likelihoods import InPainting, get_xi_condition
from models.classifier_guidance_model import ClassifierGuidanceModel
from models.diffusion import Diffusion
from models.utils import get_timesteps


def inverse_data_transform(x_, rescale=True):
    # Confirm that the image being passed in is in the range -1, 1
    # assert x_.min() >= -1.0 and x_.max() <= 1.0 and x_.min() < 0.0
    if rescale:
        x_ = (x_ + 1.0) / 2.0

    return x_

def correct_color_channels(x_, x_ref_):
    for c in range(x_.shape[0]):
        x_[c] = (x_[c] - x_[c].mean()) / x_[c].std()
        x_[c] = x_[c] * x_ref_[c].std() + x_ref_[c].mean()

    return x_


def plot_greyscale(x_sample, context, x_gt, index):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))
    ax1.imshow(x_sample.numpy(), cmap="gray")
    ax1.axis("off")
    ax1.set_title("Sample")

    ax2.imshow(context.cpu().numpy(), cmap="gray")
    ax2.axis("off")
    ax2.set_title("FBP")

    ax3.imshow(x_gt.cpu().numpy(), cmap="gray")
    ax3.axis("off")
    ax3.set_title("GT")

    wandb.log({f"cond samples/{index}": wandb.Image(plt)})
    plt.close()

    return x_sample, context, x_gt


def plot_color(x_sample, context, x_gt, task, index):
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))
    x_sample_plt = inverse_data_transform(x_sample)

    if task == "inpainting":
        c_plt = context[1:, :, :]
        c_plt = inverse_data_transform(c_plt) * context[0].std() + context[0].mean()
    else:
        c_plt = inverse_data_transform(context)

    # At each channel, correct the mean and std of x_mean_plt to match c_plt
    x_sample_plt = correct_color_channels(x_sample_plt, c_plt)
    x_sample_plt = inverse_data_transform(x_sample_plt).cpu()

    ax1.imshow(x_sample_plt.permute(1, 2, 0).detach().cpu().numpy())
    ax1.axis("off")
    ax1.set_title("Sample")

    ax2.imshow(c_plt.cpu().permute(1, 2, 0).numpy())
    ax2.axis("off")
    ax2.set_title("A^dagger(y)")

    x_plt = inverse_data_transform(x_gt)
    ax3.imshow(x_plt.cpu().permute(1, 2, 0).numpy())
    ax3.axis("off")
    ax3.set_title("GT")
    wandb.log({f"cond samples/{index}": wandb.Image(plt)})
    plt.close()

    return x_sample_plt, c_plt, x_plt


def calculate_total_psnr(
    pretrained_score,
    control_net,
    likelihood,
    val_dataloader,
    diffusion,
    device,
    val_kwargs,
    cfg_model,
    save_images=True,
    save_path=None,
    image_path="test",
    smpl_type="DDIM",
):
    control_net.eval()
    pretrained_score.eval()
    psnr_ = []
    lpips_ = []
    ssim_ = []
    psnr_skimage_ = []

    img_save_idx = 0

    ts = get_timesteps(start_step=1000, end_step=0, num_steps=val_kwargs["num_steps"])
    print("ts: ", ts)
    print(val_kwargs)

    diffusion = Diffusion()

    cfg_dict = {"algo": {"eta": val_kwargs["eta"], "sdedit": False, "cond_awd": False}}
    conf = OmegaConf.create(cfg_dict)

    for batch_idx, (x, info) in tqdm(
        enumerate(val_dataloader), total=len(val_dataloader)
    ):
        x = x.to(device)
        if isinstance(likelihood, InPainting):
            y, masks = likelihood.sample(
                x,
                deterministic_idx=torch.arange(batch_idx, batch_idx + x.shape[0])
                .long()
                .to(device),
            )
            cond = torch.cat([y, masks.unsqueeze(1)], dim=1)
        else:
            y = likelihood.sample(x)
            masks = None

            cond = y

        # y = torch.repeat_interleave(y, 10, dim=0)

        @torch.no_grad()
        def combined_model(x, t):

            control = control_net(x, t, cond)
            eps = pretrained_score(x, t,control=control)
            
            return eps

        guidance_model = ClassifierGuidanceModel(
            model=combined_model, classifier=None, diffusion=diffusion, cfg=None
        )

        sampler = DDIM(model=guidance_model, cfg=conf)

        sample = sampler.sample(
            x=torch.randn([y.shape[0], *x.shape[1:]], device=device), y=None, ts=ts
        )

        sample = sample.to(device)

        # First ensure all images in roughly [0, 1] instead of [-1, 1] for the PSNR function
        sample_01 = inverse_data_transform(sample, rescale=val_kwargs["rescale_image"])
        x_01 = inverse_data_transform(x, rescale=val_kwargs["rescale_image"])
        print("RANGE:", sample_01.min(), sample_01.max(), x_01.min(), x_01.max())
        mse = torch.mean(
            (sample_01 - x_01) ** 2, dim=(1, 2, 3)
        )  # keep batch dim for now
        psnr = 10 * torch.log10(1 / (mse + 1e-10))
        psnr_.extend(psnr.cpu().numpy())  # append entire batch separately here

        for k in range(sample.shape[0]):
            psnr_skimage_.append(
                peak_signal_noise_ratio(
                    x_01[k, :, :, :].cpu().numpy(),
                    sample_01[k, :, :, :].cpu().numpy(),
                    data_range=1,
                )
            )

        ssim = structural_similarity_index_measure(x, sample, reduction=None)
        ssim_.extend(ssim.cpu().numpy())

        if x.shape[1] == 3:
            with torch.no_grad():
                lpip = LearnedPerceptualImagePatchSimilarity().cuda()(
                    x, torch.clamp(sample, -1.0, 1.0)
                )
            lpips_.extend(np.tile(lpip.cpu().numpy(), reps=x.shape[0]))
        else:
            # dont use LPIPS if we have grayscale images
            lpips_.append(0)

        if save_images:
            for k in range(sample.shape[0]):

                import matplotlib.pyplot as plt

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                if x_01.shape[1] == 3:
                    ax1.imshow(x_01[0, :, :, :].permute(1,2,0).cpu().numpy())
                    ax2.imshow(
                    sample_01[0, :, :, :].permute(1,2,0).cpu().numpy(),
                    vmin=x_01.cpu().numpy().min(),
                    vmax=x_01.cpu().numpy().max(),
                    )
                else:
                    ax1.imshow(x_01.cpu().numpy()[0, 0, :, :], cmap="gray")
                    ax2.imshow(
                    sample_01.cpu().numpy()[0, 0, :, :],
                    cmap="gray",
                    vmin=x_01.cpu().numpy().min(),
                    vmax=x_01.cpu().numpy().max(),
                    )
                ax1.set_title("GT")
                
                ax2.set_title("Reconstruction")
                ax1.axis("off")
                ax2.axis("off")
                plt.savefig(os.path.join(save_path, f"res_{img_save_idx}.png"))
                plt.close()

                savedir = os.path.join(save_path, "imgs")
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                np.save(
                    os.path.join(savedir, "gt_{}.npy".format(img_save_idx)),
                    x_01.cpu().numpy(),
                )
                np.save(
                    os.path.join(savedir, "reco_{}.npy".format(img_save_idx)),
                    sample_01.cpu().numpy(),
                )

                img_save_idx += 1

   
        print(f"PSNR: {np.mean(psnr_)}")
        print(f"PSNR (skimage): {np.mean(psnr_skimage_)}")
        print(f"SSIM: {np.mean(ssim_)}")
        print(f"LPIPS: {np.mean(lpips_)}")
        wandb.log({"val/running_psnr": np.mean(psnr_)})
        wandb.log({"val/running_psnr_skimage": np.mean(psnr_skimage_)})
        wandb.log({"val/running_lpips": np.mean(lpips_)})
        wandb.log({"val/running_ssim": np.mean(ssim_)})

        

    wandb.log({"val/total_psnr": np.mean(psnr_)})
    wandb.log({"val/total_psnr_skimage": np.mean(psnr_skimage_)})
    wandb.log({"val/total_lpips": np.mean(lpips_)})
    wandb.log({"val/total_ssim": np.mean(ssim_)})
