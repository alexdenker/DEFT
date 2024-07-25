"""
This loss function works with the sde defined in models/diffusion.py

"""

import torch

from models.guided_diffusion.unet_cond import CondUNetModel
from models.diffusion import Diffusion

from htransform.likelihoods import Likelihood, InPainting, get_xi_condition

def epsilon_based_loss_fn_finetuning(
    x: torch.Tensor, 
    model: CondUNetModel, 
    diffusion: Diffusion, 
    pretrained_model: torch.nn.Module, 
    likelihood: Likelihood, 
    cfg_model 
):
    if isinstance(likelihood, InPainting):
        y, masks = likelihood.sample(x)
    else:
        y = likelihood.sample(x)
        masks = None
    device = x.device
    b = x.shape[0]
    i = torch.randint(0, diffusion.num_diffusion_timesteps, (b,), device=device).long()

    alpha_t = diffusion.alpha(i).view(-1, 1, 1, 1)

    z = torch.randn_like(x)

    xi = alpha_t.sqrt() * x + (1 - alpha_t).sqrt() * z

    with torch.no_grad():
        z2 = pretrained_model(xi, 1.0 * i)

        x0hat = (xi - z2 * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

        xi_condition = get_xi_condition(
            xi=xi, x0hat=x0hat, y=y, likelihood=likelihood, masks=masks, cfg_model=cfg_model
        )

    z1 = model(xi_condition, 1.0 * i)

    zhat = z1 + z2

    if zhat.ndim == 4:
        loss = torch.mean( torch.sum((z - zhat) ** 2, dim=(1, 2, 3)))
    elif zhat.ndim == 2:
        loss = torch.mean(torch.sum((z - zhat) ** 2, dim=(1,)))

    return loss
