import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from datasets import build_one_dataset
from htransform.likelihoods import Likelihood, get_xi_condition
from utils.degredations import Inpainting2
from models.diffusion import Diffusion

class ClassifierGuidanceModel:
    def __init__(
        self,
        model: nn.Module,
        classifier: nn.Module,
        diffusion: Diffusion,
        cfg: DictConfig,
    ):
        self.model = model
        self.classifier = classifier
        self.diffusion = diffusion
        self.cfg = cfg

    def __call__(self, xt, y, t, scale=1.0):
        # Returns both the noise value (score function scaled) and the predicted x0.
        alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
        if self.classifier is None:
            et = self.model(xt, t)[:, :3]
        else:
            et = self.model(xt, t, y)[:, :3]
            et = et - (1 - alpha_t).sqrt() * self.cond_fn(xt, y, t, scale=scale)
        x0_pred = (xt - et * (1 - alpha_t).sqrt()) / alpha_t.sqrt()
        return et, x0_pred

    def cond_fn(self, xt, y, t, scale=1.0):
        with torch.enable_grad():
            x_in = xt.detach().requires_grad_(True)
            logits = self.classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]

            scale = scale * self.cfg.classifier.classifier_scale
            return (
                torch.autograd.grad(selected.sum(), x_in, create_graph=True)[0] * scale
            )


class HTransformModel(ClassifierGuidanceModel):
    def __init__(
        self,
        model: nn.Module,
        htransform_model: nn.Module,
        classifier: nn.Module,
        diffusion: Diffusion,
        likelihood: Likelihood,
        cfg: DictConfig,
    ):
        super().__init__(model, classifier, diffusion, cfg)
        self.htransform_model = htransform_model
        self.likelihood = likelihood

        self.init_dataloaders()

    def __call__(self, xt, y, t, scale=1.0):
        # Returns both the noise value (score function scaled) and the predicted x0.
        alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
        

        with torch.no_grad():
            eps1 = self.model(xt, t)[:, :3]

            x0hat = (xt - eps1 * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

            if isinstance(self.likelihood.H, Inpainting2):
                y, masks = y 
            else:
                masks = None 

            xi_condition = get_xi_condition(
                    xi=xt,
                    x0hat=x0hat,
                    y=y,
                    likelihood=self.likelihood,
                    masks=masks,
                    cfg=self.cfg,
                )

        eps2 = self.htransform_model(xi_condition, t)

        et = eps1 + eps2

        x0_pred = (xt - et * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

        return et, x0_pred

    def init_dataloaders(self):
        finetune_dset, val_dset = build_one_dataset(
            self.cfg, dataset_attr="finetune_dataset", return_splits=True
        )
        print(f"Finetune dataset size is {len(finetune_dset)}")
        print(f"Validation dataset size is {len(val_dset)}")
        self.finetune_loader = torch.utils.data.DataLoader(
            finetune_dset,
            batch_size=self.cfg.algo.finetune_args.batch_size,
            shuffle=self.cfg.algo.finetune_args.shuffle,
            drop_last=self.cfg.algo.finetune_args.drop_last,
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=self.cfg.algo.val_args.batch_size,
            shuffle=False,
            drop_last=True,
        )
        self.psnr_val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=self.cfg.algo.val_args.psnr_batch_size,
            shuffle=False,
            drop_last=True,
        )
