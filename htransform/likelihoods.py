"""
This file includes the likelihood used for the imaging experiments in the paper. 
The likelihoods are implemented as sub-classes of *Likelihood*. 

These likelihood are wrappers for the forward operators defined in utils/degredations.py, 
but provide useful new functionalities, i.e., sampling and log_likelihood_grad.s

This file includes:
- Painting (Out/Inpainting)
- Superresolution
- HDR (high dynamic range, non-linear operator)
- NonLinearBlur (corrected)
- PhaseRetrieval (non-linear)
- Radon (based on https://github.com/deepinv/deepinv/blob/main/deepinv/physics/functional/radon.py)
- Blur (based on https://github.com/deepinv/deepinv/blob/main/deepinv/physics/functional/convolution.py)

Note that for some non-linear forward operator, we do not use the mathematical likelihood gradient, but 
rather cheaper approximations to facilitate faster training/sampling.

"""


import io
import math
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F

from radon.tomography import Tomography
from utils.degredations import HDR as HDR_old
from utils.degredations import NonlinearBlurOperator as NLB
from utils.degredations import PhaseRetrievalOperator as PR
from utils.degredations import SuperResolution as SR
from utils.fft_utils import fft2_m, ifft2_m


def get_xi_condition(xi, x0hat, y, likelihood, cfg, masks=None):
    """
    For the various inversion tasks, we have slightly different inputs to the model.

    Superresolution, Radon, HDR, NonLinearBlur:
        xi (noisy image at step i)
        x0hat (tweedie based on unconditional diffusion model)
        log_grad term ( A*(Ax0hat - y) )

    PhaseRetrieval:
        xi (noisy image at step i)
        x0hat (tweedie based on unconditional diffusion model)
        log_grad term ( A*(Ax0hat - y) )
        cheap_guidance ( magnitude(y) and phase(x0hat))
        simple inverse (few steps of Hybrid-Input-Output)

    InPainting:
        xi (noisy image at step i)
        x0hat (tweedie based on unconditional diffusion model)
        y (the masked measurements (masked elements = 0))
        masks (binary mask)
        log_grad term ( A*(Ax0hat - y) )
    """

    if (
        isinstance(likelihood, Superresolution)
        or isinstance(likelihood, Radon)
        or isinstance(likelihood, HDR)
        or isinstance(likelihood, NonLinearBlur)
    ):
        xi_condition = xi
        if cfg.algo.use_x0hat:
            xi_condition = torch.concat((xi_condition, x0hat), dim=1)

        if cfg.algo.use_loggrad:
            cheap_guidance = likelihood.log_likelihood_grad(x=x0hat, y=y)
            xi_condition = torch.concat((xi_condition, cheap_guidance), dim=1)
        else:
            # if we dont use the log_grad term we use A*y (makes sense for CT and superres as the measurements have a different shape)
            ATy = likelihood.A_adjoint(y)
            xi_condition = torch.concat((xi_condition, ATy), dim=1)

    elif isinstance(likelihood, PhaseRetrieval):
        cheap_guidance = likelihood.cheap_guidance(x0hat=x0hat, y=y)
        cheap_guidance_grad = likelihood.log_likelihood_grad(x=xi, y=y)
        simple_inverse = likelihood.simple_inverse(x0hat=x0hat, y=y)

        # TODO: Should we ensure all things here are in [-1, 1]?

        xi_condition = torch.concat(
            (xi, x0hat, cheap_guidance, simple_inverse, cheap_guidance_grad),
            dim=1,
        )  # concat across channels
    elif isinstance(likelihood, InPainting):
        cheap_guidance = likelihood.log_likelihood_grad(x=x0hat, y=y, masks=masks)

        xi_condition = torch.concat(
            (
                xi,  # noisy image
                x0hat,  # unconditional denoised image
                y,  # measurements
                masks[:, None, :, :].float(),  # mask
                cheap_guidance,
            ),
            dim=1,
        )  # concat across channels

    # xi_condition = x

    return xi_condition


class Likelihood:
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        samples = []
        for i in range(len(x)):
            samples.append(self._sample(x[i : i + 1]))
        return torch.concatenate(samples, dim=0)

    def _sample(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # TODO: What is this here, why are we redefining it?
    def sample(self, x: torch.Tensor) -> torch.Tensor:  # noqa: F811
        samples = []
        for k in range(x.shape[0]):
            e = self._sample(x[[k]])
            samples.append(e)
        return torch.cat(samples, dim=0)

    def none_like(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def plot_condition(self, x, y, ax):
        raise NotImplementedError


class Painting(Likelihood):
    def __init__(self, patch_size: int, pad_value: float):
        self.pad_value = pad_value
        self.patch_size = patch_size

    def get_random_patch(self, image_size):
        # don't sample to close to border.
        h = torch.randint(5, image_size - self.patch_size - 5, size=())
        w = torch.randint(5, image_size - self.patch_size - 5, size=())
        return h, w

    def none_like(self, x):
        return torch.ones_like(x) * self.pad_value, torch.zeros_like(x)[
            :, 0, :, :
        ].unsqueeze(1)

    def loss(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        x: [N, C, H, W]
        condition: [N, C, H, W]
        """
        x = torch.where(condition == self.pad_value, 0.0, x)
        condition = torch.where(condition == self.pad_value, 0.0, condition)
        loss = torch.sum((x - condition) ** 2, dim=(1, 2, 3))  # [N,]
        return loss

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        samples = []
        masks = []
        for i in range(len(x)):
            sample, mask = self._sample(x[i : i + 1])
            samples.append(sample)
            masks.append(mask)

        return torch.concat(samples, dim=0), torch.concat(masks, dim=0)

    def _sample(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n, c, h, 1, w, 1)
    out = out.view(n, c, scale * h, scale * w)
    return out


class Superresolution(Likelihood):
    def __init__(self, scale, sigma_y, device):
        self.scale = scale
        self.sigma_y = sigma_y
        self.device = device
        # scale = round(args.forward_op.scale)
        # self.AvgPool = torch.nn.AdaptiveAvgPool2d((256 // self.scale, 256 // self.scale))
        self.forward_op = SR(channels=3, img_dim=256, ratio=self.scale, device=device)

    def loss(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return None

    def log_likelihood_grad(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        res = self.A(x) - y

        if self.sigma_y == 0:
            return -self.A_adjoint(res)
        return -1 / self.sigma_y**2 * self.A_adjoint(res)

    def _sample(self, x: torch.Tensor) -> torch.Tensor:
        y = self.A(x)

        # TODO: Do we also scale sigma_y by 2 here?
        y_noise = y + self.sigma_y * torch.randn_like(y)

        return y_noise

    def A(self, x):
        return self.forward_op.H(x).reshape(
            x.shape[0], 3, 256 // self.scale, 256 // self.scale
        )

    def A_adjoint(self, y):
        return self.forward_op.Ht(y).reshape(
            y.shape[0], 3, 256, 256
        )  # MeanUpsample(y, self.scale) / (self.scale**2)


class HDR(Likelihood):
    def __init__(self, sigma_y, device):
        self.device = device
        self.sigma_y = sigma_y
        self.forward_op = HDR_old()

    def loss(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return None

    def log_likelihood_grad(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        res = self.A(x) - y

        if self.sigma_y == 0:
            return -self.A_adjoint(res)
        return -1 / self.sigma_y**2 * self.A_adjoint(res)

    def _sample(self, x: torch.Tensor) -> torch.Tensor:
        y = self.A(x)

        y_noise = y + self.sigma_y * torch.randn_like(y)

        return y_noise

    def A(self, x):
        return self.forward_op.H(x)

    def A_adjoint(self, y):
        return self.forward_op.H_pinv(y)


class NonLinearBlur(Likelihood):
    def __init__(self, opt_yml_path, current_dir, device):
        self.operator = NLB(opt_yml_path, current_dir, device)

    def _sample(self, x: torch.Tensor) -> torch.Tensor:
        y = self.A(x)

        y_noise = y

        return y_noise

    def A(self, x):
        return self.operator.H(x)

    def A_adjoint(self, y):
        return self.operator.H_pinv(y)

    def log_likelihood_grad(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        res = self.A(x) - y

        return -self.A_adjoint(res)


class PhaseRetrieval(Likelihood):
    def __init__(self, oversample, sigma_y, device):
        self.oversample = oversample
        self.device = device
        self.sigma_y = sigma_y
        # scale = round(args.forward_op.scale)
        # self.AvgPool = torch.nn.AdaptiveAvgPool2d((256 // self.scale, 256 // self.scale))
        self.forward_op = PR(oversample=self.oversample, device=device)
        self.pad = self.forward_op.pad

    def loss(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return None

    def log_likelihood_grad(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x.requires_grad_(True)

            x = (x + 1.0) / 2.0
            e_obs = y - self.A(x)
            loss_obs = -(e_obs**2).sum() / 2

            grad = torch.autograd.grad(outputs=loss_obs, inputs=x)[0]

            x.detach()

        # if self.sigma_y == 0:
        #    return -self.A_adjoint(res)
        # return -1 / self.sigma_y**2 * self.A_adjoint(res)

        return grad

    def _sample(self, x: torch.Tensor) -> torch.Tensor:
        y = self.A((x + 1) / 2)

        y_noise = y + self.sigma_y * torch.randn_like(y)

        return y_noise

    def A(self, x):
        return self.forward_op.H(x)

    def A_adjoint(self, y):
        return self.forward_op.H_pinv(y)

    def fft(self, data):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded)
        return amplitude

    def ifft(self, data):
        x = ifft2_m(data).abs()

        x = self.forward_op.undo_padding(x, self.pad, self.pad, self.pad, self.pad)
        return x

    def cheap_guidance(self, x0hat, y):
        x0hat_01 = (x0hat + 1.0) / 2.0

        x0hat_01 = torch.clamp(x0hat_01, 0.0, 1.0)
        x_fft = self.fft(x0hat_01)
        # print(x_fft.shape)

        phase = torch.angle(x_fft)

        # print(phase.shape)

        cheap_guidance = self.ifft(y * torch.exp(1j * phase))
        cheap_guidance = (2.0 * cheap_guidance) - 1.0

        return cheap_guidance

    def simple_inverse(self, x0hat, y):
        x_start = (x0hat + 1.0) / 2.0

        xi = F.pad(x_start, (self.pad, self.pad, self.pad, self.pad))
        beta = 0.95

        for k in range(7):
            for i in range(50):
                xi_fft = fft2_m(xi)
                g_fft = y * torch.exp(1j * torch.angle(xi_fft))
                g_ifft = ifft2_m(g_fft)

                if k % 2 == 0:
                    xi = F.pad(
                        g_ifft[:, :, self.pad : -self.pad, self.pad : -self.pad],
                        (self.pad, self.pad, self.pad, self.pad),
                    )
                else:
                    tmp = g_ifft[:, :, self.pad : -self.pad, self.pad : -self.pad]
                    xi = xi - beta * g_ifft
                    xi[:, :, self.pad : -self.pad, self.pad : -self.pad] = tmp

                # after three iterations: align color channels
                if k == 3:
                    if xi.shape[1] == 3:
                        xi[0, 1, :, :] = xi[0, 0, :, :]
                        xi[0, 2, :, :] = xi[0, 0, :, :]

        xi = (
            2.0
            * self.forward_op.undo_padding(
                xi.abs(), self.pad, self.pad, self.pad, self.pad
            )
            - 1.0
        )
        return xi


class Radon(Likelihood):
    def __init__(self, num_angles, sigma_y, image_size, device):
        self.num_angles = num_angles
        self.sigma_y = sigma_y

        # self.ray_trafo = SimpleTrafo(
        #    im_shape=[image_size, image_size],
        #    num_angles=self.num_angles,
        # )
        # self.ray_trafo = self.ray_trafo.to(device=device)
        self.physics = Tomography(
            img_width=image_size, angles=num_angles, device=device
        )

    def loss(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return None

    def log_likelihood_grad(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        res = self.A(x) - y

        return -1 / self.sigma_y**2 * self.fbp(res)

    def _sample(self, x: torch.Tensor) -> torch.Tensor:
        y = self.A(x)

        y_noise = y + self.sigma_y * torch.randn_like(y)

        return y_noise

    def A(self, x):
        return self.physics.A(x)

    def A_adjoint(self, y):
        return self.physics.A_adjoint(y)

    def A_adjoint_rescaled(self, y):
        return self.physics.A_adjoint(y) / (math.pi / (2 * self.num_angles))

    def fbp(self, y):
        return self.physics.A_dagger(y)


# w = torch.ones((1, 3, 3, 3)) / 9
class Blur(Likelihood):
    def __init__(self, filter, sigma_y, padding="circular", device="cpu"):
        self.padding = padding
        self.device = device
        self.sigma_y = sigma_y
        self.filter = torch.nn.Parameter(filter, requires_grad=False).to(device)

    def loss(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # """
        # x: [N, C, H, W]
        # condition: [N, C, H, W]
        # """
        # x = torch.where(condition == self.pad_value, 0.0, x)
        # condition = torch.where(condition == self.pad_value, 0.0, condition)
        # loss = torch.sum((x - condition)**2, dim=(1, 2, 3))  # [N,]
        # return None
        return None

    def log_likelihood_grad(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        res = self.A(x) - y

        return -1 / self.sigma_y**2 * self.A_adjoint(res)

    def _sample(self, x: torch.Tensor) -> torch.Tensor:
        y = self.A(x)

        y_noise = y + self.sigma_y * torch.randn_like(y)

        return y_noise

    def A(self, x):
        return conv(x, self.filter, self.padding)

    def A_adjoint(self, y):
        return conv_transpose(y, self.filter, self.padding)


class InPainting(Likelihood):
    """Condition is image with a missing patch."""

    def __init__(self, sigma_y, mask_filename, device):
        self.sigma_y = sigma_y
        self.device = device

        mask_filename = Path(mask_filename).resolve()
        with open(mask_filename, "rb") as f:
            data = f.read()

        data = dict(np.load(io.BytesIO(data)))

        for key in data:
            data[key] = (
                np.unpackbits(data[key], axis=None)[: np.prod([10000, 256, 256])]
                .reshape([10000, 256, 256])
                .astype(np.uint8)
            )

        self.dense_masks = torch.tensor(data["20-30% freeform"], device=device)

    def loss(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return None

    def log_likelihood_grad(
        self, x: torch.Tensor, y: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        grad_term = y - x * masks[:, None, :, :]

        if self.sigma_y == 0:
            return -grad_term
        return -1 / self.sigma_y**2 * grad_term

    def sample(self, x: torch.Tensor, deterministic_idx=None) -> torch.Tensor:
        # Randomly sample a batch of random idx from the dense_masks
        if deterministic_idx is not None:
            idx = deterministic_idx
        else:
            idx = torch.randint(
                0, len(self.dense_masks), (x.shape[0],), device=x.device
            )

        masks = 1 - self.dense_masks[idx]
        y = self.A(x, masks)

        y_noise = y + self.sigma_y * torch.randn_like(y)

        return y_noise, masks

    def A(self, x, masks):
        return (x * masks[:, None, :, :]).float()

    def A_adjoint(self, y):
        return y


class OutPainting(Painting):
    """
    The condition only contain a patch and everything else is masked out.
    """

    def _sample(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [N, C, H, W]
        """
        image_size = images.shape[-1]
        h, w = self.get_random_patch(image_size)
        slice_ = np.s_[:, :, h : h + self.patch_size, w : w + self.patch_size]
        condition = torch.ones_like(images) * self.pad_value
        condition[slice_] = images[slice_].detach().clone()

        mask = torch.zeros((1, 1, image_size, image_size), device=images.device)
        mask[slice_] = 1

        return condition, mask


def conv(x, filter, padding):
    r"""
    Convolution of x and filter. The transposed of this operation is conv_transpose(x, filter, padding)

    :param x: (torch.Tensor) Image of size (B,C,W,H).
    :param filter: (torcstring)h.Tensor) Filter of size (1,C,W,H) for colour filtering or (1,1,W,H) for filtering each channel with the same filter.
    :param padding: ( options = 'valid', 'circular', 'replicate', 'reflect'. If padding='valid' the blurred output is smaller than the image (no padding), otherwise the blurred output has the same size as the image.

    """
    b, c, h, w = x.shape

    filter = filter.flip(-1).flip(
        -2
    )  # In order to perform convolution and not correlation like Pytorch native conv

    filter = extend_filter(filter)

    ph = (filter.shape[2] - 1) / 2
    pw = (filter.shape[3] - 1) / 2

    if padding != "valid":
        pw = int(pw)
        ph = int(ph)
        x = F.pad(x, (pw, pw, ph, ph), mode=padding, value=0)

    if filter.shape[1] == 1:
        if filter.shape[0] == 1:
            filter = filter.repeat(c, 1, 1, 1)
        y = F.conv2d(x, filter, padding="valid", groups=c)
    else:
        y = F.conv2d(x, filter, padding="valid")

    return y


def conv_transpose(y, filter, padding):
    r"""
    Transposed convolution of x and filter. The transposed of this operation is conv(x, filter, padding)

    :param torch.Tensor x: Image of size (B,C,W,H).
    :param torch.Tensor filter: Filter of size (1,C,W,H) for colour filtering or (1,C,W,H) for filtering each channel with the same filter.
    :param str padding: options are ``'valid'``, ``'circular'``, ``'replicate'`` and ``'reflect'``.
        If ``padding='valid'`` the blurred output is smaller than the image (no padding)
        otherwise the blurred output has the same size as the image.
    """

    b, c, h, w = y.shape

    filter = filter.flip(-1).flip(
        -2
    )  # In order to perform convolution and not correlation like Pytorch native conv

    filter = extend_filter(filter)

    ph = (filter.shape[2] - 1) / 2
    pw = (filter.shape[3] - 1) / 2

    h_out = int(h + 2 * ph)
    w_out = int(w + 2 * pw)
    pw = int(pw)
    ph = int(ph)

    x = torch.zeros((b, c, h_out, w_out), device=y.device)
    if filter.shape[1] == 1:
        for i in range(b):
            if filter.shape[0] > 1:
                f = filter[i, :, :, :].unsqueeze(0)
            else:
                f = filter

            for j in range(c):
                x[i, j, :, :] = F.conv_transpose2d(
                    y[i, j, :, :].unsqueeze(0).unsqueeze(1), f
                )
    else:
        x = F.conv_transpose2d(y, filter)

    if padding == "valid":
        out = x
    elif padding == "zero":
        out = x[:, :, ph:-ph, pw:-pw]
    elif padding == "circular":
        out = x[:, :, ph:-ph, pw:-pw]
        # sides
        out[:, :, :ph, :] += x[:, :, -ph:, pw:-pw]
        out[:, :, -ph:, :] += x[:, :, :ph, pw:-pw]
        out[:, :, :, :pw] += x[:, :, ph:-ph, -pw:]
        out[:, :, :, -pw:] += x[:, :, ph:-ph, :pw]
        # corners
        out[:, :, :ph, :pw] += x[:, :, -ph:, -pw:]
        out[:, :, -ph:, -pw:] += x[:, :, :ph, :pw]
        out[:, :, :ph, -pw:] += x[:, :, -ph:, :pw]
        out[:, :, -ph:, :pw] += x[:, :, :ph, -pw:]

    elif padding == "reflect":
        out = x[:, :, ph:-ph, pw:-pw]
        # sides
        out[:, :, 1 : 1 + ph, :] += x[:, :, :ph, pw:-pw].flip(dims=(2,))
        out[:, :, -ph - 1 : -1, :] += x[:, :, -ph:, pw:-pw].flip(dims=(2,))
        out[:, :, :, 1 : 1 + pw] += x[:, :, ph:-ph, :pw].flip(dims=(3,))
        out[:, :, :, -pw - 1 : -1] += x[:, :, ph:-ph, -pw:].flip(dims=(3,))
        # corners
        out[:, :, 1 : 1 + ph, 1 : 1 + pw] += x[:, :, :ph, :pw].flip(dims=(2, 3))
        out[:, :, -ph - 1 : -1, -pw - 1 : -1] += x[:, :, -ph:, -pw:].flip(dims=(2, 3))
        out[:, :, -ph - 1 : -1, 1 : 1 + pw] += x[:, :, -ph:, :pw].flip(dims=(2, 3))
        out[:, :, 1 : 1 + ph, -pw - 1 : -1] += x[:, :, :ph, -pw:].flip(dims=(2, 3))

    elif padding == "replicate":
        out = x[:, :, ph:-ph, pw:-pw]
        # sides
        out[:, :, 0, :] += x[:, :, :ph, pw:-pw].sum(2)
        out[:, :, -1, :] += x[:, :, -ph:, pw:-pw].sum(2)
        out[:, :, :, 0] += x[:, :, ph:-ph, :pw].sum(3)
        out[:, :, :, -1] += x[:, :, ph:-ph, -pw:].sum(3)
        # corners
        out[:, :, 0, 0] += x[:, :, :ph, :pw].sum(3).sum(2)
        out[:, :, -1, -1] += x[:, :, -ph:, -pw:].sum(3).sum(2)
        out[:, :, -1, 0] += x[:, :, -ph:, :pw].sum(3).sum(2)
        out[:, :, 0, -1] += x[:, :, :ph, -pw:].sum(3).sum(2)
    return out


def extend_filter(filter):
    b, c, h, w = filter.shape
    w_new = w
    h_new = h

    offset_w = 0
    offset_h = 0

    if w == 1:
        w_new = 3
        offset_w = 1
    elif w % 2 == 0:
        w_new += 1

    if h == 1:
        h_new = 3
        offset_h = 1
    elif h % 2 == 0:
        h_new += 1

    out = torch.zeros((b, c, h_new, w_new), device=filter.device)
    out[:, :, offset_h : h + offset_h, offset_w : w + offset_w] = filter
    return out


def get_likelihood(cfg: DictConfig, device: str):
    # For a given algo.deg, get the corresponding yaml file from likelihoods/
    if cfg.algo.deg not in ["sr4", "blur", "ct", "phase_retrieval", "inp", "hdr"]:
        raise NotImplementedError
    if cfg.algo.deg != cfg.likelihood.name:
        warnings.warn(
            f"algo.deg and likelihood.name are not the same: {cfg.algo.deg} != {cfg.likelihood.name}. Loading likelihood_cfg from likelihood/{cfg.likelihood.name}.yaml"
        )
        likelihood_cfg = OmegaConf.load(
            f"DEFT/_configs/likelihood/{cfg.likelihood.name}.yaml"
        )
    else:
        likelihood_cfg = cfg.likelihood

    if cfg.algo.deg == "sr4":  # super resolution
        scale = round(likelihood_cfg.forward_op.scale)
        return Superresolution(
            scale=scale,
            sigma_y=likelihood_cfg.forward_op.noise_std,
            device=device,
        )

    elif cfg.algo.deg == "blur":
        return NonLinearBlur(
            opt_yml_path=likelihood_cfg.forward_op.opt_yml_path,
            current_dir=os.getcwd(),
            device=device,
        )

    elif cfg.algo.deg == "ct":
        return Radon(
            num_angles=likelihood_cfg.forward_op.num_angles,
            sigma_y=likelihood_cfg.forward_op.noise_std,
            image_size=cfg.data.image_size,
            device=device,
        )
    elif cfg.algo.deg == "phase_retrieval":
        return PhaseRetrieval(
            oversample=likelihood_cfg.forward_op.oversample,
            sigma_y=likelihood_cfg.forward_op.noise_std,
            device=device,
        )
    elif cfg.algo.deg == "inp":
        return InPainting(
            sigma_y=likelihood_cfg.forward_op.noise_std,
            mask_filename=likelihood_cfg.forward_op.mask_filename,
            device=device,
        )
    elif cfg.algo.deg == "hdr":
        return HDR(sigma_y=likelihood_cfg.forward_op.noise_std, device=device)
    else:
        raise NotImplementedError
