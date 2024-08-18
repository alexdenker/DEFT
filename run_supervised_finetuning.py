import os

import ml_collections.config_flags
import numpy as np
import torch
import torchvision
import yaml
from absl import app, flags
from omegaconf import OmegaConf
from PIL import Image

import wandb
from datasets.aapm import AAPMDataset
from datasets.imagenet import ImageNet
from datasets.lodopab import LoDoPabDatasetFromDival
from htransform.likelihoods import (
    HDR,
    InPainting,
    NonLinearBlur,
    PhaseRetrieval,
    Radon,
    Superresolution,
)
from htransform.trainer import htransform_trainer
from models.diffusion import Diffusion
from models.utils import create_model, dict2namespace, flatten_nested_dict

ml_collections.config_flags.DEFINE_config_file(
    "config",
    "configs/default_config.py",
    "Training configuration.",
    lock_config=True,
)

FLAGS = flags.FLAGS
# TODO: Do float64


def center_crop_arr(pil_image, image_size=256):
    # Imported from openai/guided-diffusion
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return (
        torch.from_numpy(
            arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
        )
        .permute(2, 0, 1)
        .float()
        / 255.0
    )


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    # Convert from [0, 1] to [-1, 1]
    if config.data.rescaled:
        X = 2 * X - 1.0

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)


def coordinator(args):
    run_id = wandb.util.generate_id()
    wandb_kwargs = {
        "project": args.wandb.project,
        "entity": args.wandb.entity,
        "config": flatten_nested_dict(args.to_dict()),
        "name": args.wandb.name if args.wandb.name else None,
        "mode": "online" if args.wandb.log else "disabled",
        "settings": wandb.Settings(code_dir=args.wandb.code_dir),
        "dir": args.wandb.code_dir,
        "id": run_id,
        "resume": "allow",
    }
    with wandb.init(**wandb_kwargs) as _:
        name_to_yaml = {
            "ellipses": "diskellipses.yml",
            "imagenet": "imagenet_256.yml",
            "lodopab": "lodopab.yml",
            "aapm": "aapm.yml",
        }
        assert args.model_trained_on in list(
            name_to_yaml.keys()
        ), "model configs not found for {}".format(args.model_trained_on)

        with open(
            os.path.join("configs", name_to_yaml[args.model_trained_on]), "r"
        ) as f:
            model_config = yaml.safe_load(f)
        model_config = dict2namespace(model_config)

        sde = Diffusion(
            num_diffusion_timesteps=model_config.diffusion.num_diffusion_timesteps
        )

        pretrained_model = create_model(**vars(model_config.model))
        pretrained_model.convert_to_fp32()
        pretrained_model.dtype = torch.float32
        pretrained_model.load_state_dict(torch.load(model_config.data.model_path))
        pretrained_model.eval()
        finetune_model_cfg = dict2namespace(args.finetune_model_config)
        htransform_model = create_model(**vars(finetune_model_cfg))

        finetune_model_cfg = dict2namespace(args.finetune_model_config)
        htransform_model = create_model(**vars(finetune_model_cfg))
        htransform_model.convert_to_fp32()
        htransform_model.dtype = torch.float32

        print(
            "number of parameters in pretrained model: ",
            sum([p.numel() for p in pretrained_model.parameters()]),
        )
        print(
            "number of parameters in finetuned model: ",
            sum([p.numel() for p in htransform_model.parameters()]),
        )
        print(
            "Fraction: ",
            sum([p.numel() for p in htransform_model.parameters()])
            / sum([p.numel() for p in pretrained_model.parameters()]),
        )
        base_path = args.base_path

        # log_dir = os.path.join(base_path, f'{time.strftime("%d-%m-%Y-%H-%M-%S")}')
        log_dir = os.path.join(base_path, run_id)
        print("save model to ", log_dir)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        with open(os.path.join(log_dir, "report.yaml"), "w") as file:
            try:
                yaml.dump(args, file)
            except AttributeError:
                yaml.dump(OmegaConf.to_container(args, resolve=True), file)

        with open(os.path.join(log_dir, "config.yaml"), "w") as file:
            yaml.dump(args.to_dict(), file)

        if args.model_trained_on == "imagenet":
            if args.dataset.use_default_loader:
                image_dataset = torchvision.datasets.ImageNet(
                    args.dataset.root,
                    split="val",
                    meta_root=args.dataset.meta_root,
                    transform=torchvision.transforms.Compose(
                        [
                            lambda x: center_crop_arr(x, image_size=256),
                            lambda x: data_transform(model_config, x),
                        ]
                    ),
                )
            else:
                image_dataset = ImageNet(
                    args.dataset.root,
                    split="val",
                    meta_root=args.dataset.meta_root,
                    transform=torchvision.transforms.Compose(
                        [
                            lambda x: center_crop_arr(x, image_size=256),
                            lambda x: data_transform(model_config, x),
                        ]
                    ),
                    subset_txt=args.dataset.subset_txt,
                    subset_type="old",
                )

            print(f"len total image_dataset: {len(image_dataset)}")

            val_subset = np.arange(1, len(image_dataset), 100)

            # We use default 1000 images
            train_subset = np.arange(0, len(image_dataset), 10)

            # Get all other ids that are not in val_subset or train_subset
            other_ids = np.setdiff1d(
                np.arange(len(image_dataset)),
                np.concatenate([val_subset, train_subset]),
            )

            # Shuffle the other_ids deterministically
            rng_1 = np.random.default_rng(seed=0)
            rng_2 = np.random.default_rng(seed=1)
            rng_1.shuffle(other_ids)
            rng_2.shuffle(train_subset)

            if args.finetune_model_config.num_training_pts > 1000:
                train_subset = np.concatenate(
                    [
                        train_subset,
                        other_ids[: args.finetune_model_config.num_training_pts - 1000],
                    ]
                )
            elif args.finetune_model_config.num_training_pts < 1000:
                train_subset = train_subset[
                    : args.finetune_model_config.num_training_pts
                ]

            train_image_dataset = torch.utils.data.Subset(image_dataset, train_subset)
            val_image_dataset = torch.utils.data.Subset(image_dataset, val_subset)

            class DatasetWrapper(torch.utils.data.Dataset):
                def __init__(self, image_dataset):
                    super().__init__()

                    self.image_dataset = image_dataset

                def __len__(self):
                    return len(self.image_dataset)

                def __getitem__(self, idx: int):
                    return self.image_dataset[idx][0]

            train_image_dataset = DatasetWrapper(train_image_dataset)
            val_image_dataset = DatasetWrapper(val_image_dataset)

            print(f"len fine-tuning image_dataset: {len(train_image_dataset)}")
            print(f"len val-image_dataset: {len(val_image_dataset)}")

        elif args.model_trained_on == "lodopab":
            dataset = LoDoPabDatasetFromDival(im_size=256)
            image_dataset = dataset.lodopab_val
            train_image_dataset = image_dataset
            val_image_dataset = torch.utils.data.Subset(
                image_dataset, np.arange(1, len(image_dataset), 200)
            )
            print("len train dataset: ", len(image_dataset))
            print("len val dataset: ", len(val_image_dataset))

        elif args.model_trained_on == "aapm":
            train_image_dataset = AAPMDataset(part="val", base_path=args.dataset.root)
            val_image_dataset = AAPMDataset(part="test", base_path=args.dataset.root)

            print("len train dataset: ", len(train_image_dataset))
            print("len val dataset: ", len(val_image_dataset))

        else:
            raise ValueError

        if args.inversion_task == "sr":  # super resolution
            scale = round(args.forward_op.scale)
            likelihood = Superresolution(
                scale=scale, sigma_y=args.forward_op.noise_std, device=args.device
            )

        elif args.inversion_task == "blur":
            likelihood = NonLinearBlur(
                opt_yml_path=args.forward_op.opt_yml_path,
                current_dir=os.getcwd(),
                device=args.device,
            )

        elif args.inversion_task == "ct":
            likelihood = Radon(
                num_angles=args.forward_op.num_angles,
                sigma_y=args.forward_op.noise_std,
                image_size=model_config.data.image_size,
                device=args.device,
            )
        elif args.inversion_task == "phase_retrieval":
            likelihood = PhaseRetrieval(
                oversample=args.forward_op.oversample,
                sigma_y=args.forward_op.noise_std,
                device=args.device,
            )
        elif args.inversion_task == "inp":
            likelihood = InPainting(
                sigma_y=args.forward_op.noise_std,
                mask_filename=args.forward_op.mask_filename,
                device=args.device,
            )
        elif args.inversion_task == "hdr":
            likelihood = HDR(sigma_y=args.forward_op.noise_std, device=args.device)
        else:
            raise NotImplementedError

        train_dl = torch.utils.data.DataLoader(
            train_image_dataset,
            batch_size=args.training.batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_dl = torch.utils.data.DataLoader(
            val_image_dataset,
            batch_size=args.validation.batch_size,
            shuffle=False,
            drop_last=True,
        )
        psnr_val_dl = torch.utils.data.DataLoader(
            val_image_dataset,
            batch_size=args.validation.psnr_batch_size,
            shuffle=False,
            drop_last=True,
        )

        finetuned_score, ema = htransform_trainer(
            finetuned_score=htransform_model.to(args.device),
            pretrained_model=pretrained_model.to(args.device),
            diffusion=sde,
            train_dl=train_dl,
            val_dl=val_dl,
            psnr_val_dl=psnr_val_dl,
            cfg_train=args.training,
            cfg_model=args.finetune_model_config,
            optim_kwargs={
                "epochs": args.training.epochs,
                "lr": float(args.training.lr),
                "log_freq": args.training.log_freq,
                "save_model_every_n_epoch": args.training.save_model_every_n_epoch,
                "lr_annealing": args.training.lr_annealing,
            },
            val_kwargs={
                "batch_size": args.validation.batch_size,
                "num_steps": args.validation.num_steps,
                "eta": args.validation.eta,
                "sample_freq": args.validation.sample_freq,
                "psnr_sample_freq": args.validation.psnr_sample_freq,
                "psnr_batch_size": args.validation.psnr_batch_size,
            },
            device=args.device,
            log_dir=log_dir,
            likelihood=likelihood,
        )

        # Save the wandb config so that it can be loaded again
        with open(os.path.join(log_dir, "wandb_config.yaml"), "w") as file:
            yaml.dump(wandb.config, file)


if __name__ == "__main__":

    def _main(argv):
        del argv
        config = FLAGS.config
        coordinator(config)

    app.run(_main)
