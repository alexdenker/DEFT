"""
This loss function works with the sde defined in models/diffusion.py

"""

import torch

from htransform.likelihoods import InPainting, Likelihood, get_xi_condition
from models.diffusion import Diffusion
from models.guided_diffusion.unet_cond import CondUNetModel


def epsilon_based_loss_fn_finetuning(
    x: torch.Tensor,
    model: CondUNetModel,
    diffusion: Diffusion,
    pretrained_model: torch.nn.Module,
    likelihood: Likelihood,
    cfg,
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

    # z is unbounded
    z = torch.randn_like(x)

    # x is in [0, 1]
    xi = alpha_t.sqrt() * x + (1 - alpha_t).sqrt() * z

    with torch.no_grad():
        z2 = pretrained_model(xi, 1.0 * i)

        # z2 is unbounded in [-5, 4], x0hat is in [-a, b]
        x0hat = (xi - z2 * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

        xi_condition = get_xi_condition(
            xi=xi,
            x0hat=x0hat,
            y=y,
            likelihood=likelihood,
            masks=masks,
            cfg=cfg,
        )

    # z1 also unbounded
    z1 = model(xi_condition, 1.0 * i)

    # zhat is unbounded
    zhat = z1 + z2

    if zhat.ndim == 4:
        loss = torch.mean(torch.sum((z - zhat) ** 2, dim=(1, 2, 3)))
    elif zhat.ndim == 2:
        loss = torch.mean(torch.sum((z - zhat) ** 2, dim=(1,)))

    return loss
