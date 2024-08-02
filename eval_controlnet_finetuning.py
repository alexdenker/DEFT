import argparse
import os
from collections.abc import MutableMapping

import ml_collections.config_flags
import numpy as np
import torch
import torchvision
import yaml
from absl import app, flags
from omegaconf import OmegaConf
from PIL import Image
import time 

import wandb
from datasets.imagenet import ImageNet
from datasets.lodopab import LoDoPabDatasetFromDival
from htransform.controlnet_eval import calculate_total_psnr
from htransform.likelihoods import (
    HDR,
    InPainting,
    NonLinearBlur,
    PhaseRetrieval,
    Radon,
    Superresolution,
)
from models.diffusion import Diffusion
from models.utils import create_model, dict2namespace, flatten_nested_dict, create_controlnet
from datasets.aapm import AAPMDataset

from utils.distributed import common_init

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
    elif config.data.logit_transform:
        X = logit_transform(X)

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
    run_id = args.run_id
    base_path = args.base_path

    image_path = args.image_path
    log_dir = os.path.join(base_path, run_id)

    common_init(0, seed=6)
    # torch.cuda.set_device(dist.get_rank())

    val_args = args.validation

    del args

    # Load it in from yaml to ml_collections
    with open(os.path.join(log_dir, "config.yaml"), "r") as file:
        args = OmegaConf.create(yaml.safe_load(file))

    print(f"val_args: {val_args}")
    for k, v in val_args.items():
        if k not in args.validation:
            print(f"adding {k} with value {v}")
            args.validation[k] = v
        else:
            if args.validation[k] != v:
                print(f"updating {k} from {args.validation[k]} to {v}")
                args.validation[k] = v

    args.wandb.name = f"eval of {run_id}"

    wandb_kwargs = {
        "project": args.wandb.project,
        "entity": args.wandb.entity,
        # TODO: Need to combine config here as well.
        # "config": flatten_nested_dict(args.to_dict()),
        "name": args.wandb.name if args.wandb.name else None,
        # "mode": "online" if args.wandb.log else "disabled",
        "mode": "disabled",
        "settings": wandb.Settings(code_dir=args.wandb.code_dir),
        "dir": args.wandb.code_dir,
        # "id": run_id,
        "resume": "allow",
    }
    with wandb.init(**wandb_kwargs) as run:
        if args.model_trained_on == "ellipses":
            with open(os.path.join("configs", "diskellipses.yml"), "r") as f:
                model_config = yaml.safe_load(f)
            model_config = dict2namespace(model_config)
        elif args.model_trained_on == "imagenet":
            with open(os.path.join("configs", "imagenet_256.yml"), "r") as f:
                model_config = yaml.safe_load(f)
            model_config = dict2namespace(model_config)
        elif args.model_trained_on == "lodopab":
            with open(os.path.join("configs", "lodopab.yml"), "r") as f:
                model_config = yaml.safe_load(f)
            model_config = dict2namespace(model_config)
        elif args.model_trained_on == "aapm":
            with open(os.path.join("configs", "aapm.yml"), "r") as f:
                model_config = yaml.safe_load(f)
            model_config = dict2namespace(model_config)
        
        sde = Diffusion(
             num_diffusion_timesteps=model_config.diffusion.num_diffusion_timesteps)


        pretrained_model = create_model(**vars(model_config.model), controlled=True)
        pretrained_model.convert_to_fp32()
        pretrained_model.dtype = torch.float32
        pretrained_model.load_state_dict(torch.load(model_config.data.model_path))
        pretrained_model.eval()

        control_net = create_controlnet(**vars(model_config.model), cond_channels=args.finetune_model_config.in_channels)
        control_net.to("cuda")

        print(
            "number of parameters in pretrained model: ",
            sum([p.numel() for p in pretrained_model.parameters()]),
        )
        print(
            "number of parameters in finetuned model: ",
            sum([p.numel() for p in control_net.parameters()]),
        )
        print(
            "Fraction: ",
            sum([p.numel() for p in control_net.parameters()])
            / sum([p.numel() for p in pretrained_model.parameters()]),
        )
        # base_path = args.base_path

        # log_dir = os.path.join(base_path, f'{time.strftime("%d-%m-%Y-%H-%M-%S")}')
        log_dir = os.path.join(base_path, run_id)
        print("load model from ", log_dir)
        if not os.path.exists(log_dir):
            raise ValueError(f"Path {log_dir} does not exist.")

        if args.validation.use_ema:
            control_net.load_state_dict(
                torch.load(os.path.join(log_dir, "ema_model_tmp.pt"))
            )
        else:
            control_net.load_state_dict(
                torch.load(os.path.join(log_dir, "model.pt"))
            )
        control_net.eval()

        if args.model_trained_on == "imagenet":
            print("loading val from ", args.dataset.val_subset_txt)
            psnr_val_image_dataset = ImageNet(
                args.dataset.root,
                split="val",
                meta_root=args.dataset.meta_root,
                transform=torchvision.transforms.Compose(
                    [
                        lambda x: center_crop_arr(x, image_size=256),
                        lambda x: data_transform(model_config, x),
                    ]
                ),
                subset_txt="datasets/data/dgp_top1k.txt",
            )

            # ONLY USE A SMALL SUBSET FOR A QUICK VALIDATION
            #print("WARNING: ONLY USE A SMALL SUBSET OF 20 IMAGES FOR A QUICK VALIDATION")
            #psnr_val_image_dataset = torch.utils.data.Subset(
            #    psnr_val_image_dataset, np.arange(0, 20)
            # )

            class DatasetWrapper(torch.utils.data.Dataset):
                def __init__(self, image_dataset):
                    super().__init__()

                    self.image_dataset = image_dataset

                def __len__(self):
                    return len(self.image_dataset)

                def __getitem__(self, idx: int):
                    return self.image_dataset[idx][0], self.image_dataset[idx][2]

            psnr_val_image_dataset = DatasetWrapper(psnr_val_image_dataset)

            print(f"len psnr val image_dataset: {len(psnr_val_image_dataset)}")
        elif args.model_trained_on == "lodopab":
            dataset = LoDoPabDatasetFromDival(im_size=256)

            image_dataset = dataset.lodopab_test  # lodopab_test  # use test set
            train_image_dataset = image_dataset
            val_image_dataset = torch.utils.data.Subset(
                image_dataset, np.arange(1, len(image_dataset), 20)
            )
            psnr_val_image_dataset = val_image_dataset
            print("len train dataset: ", len(image_dataset))
            print("len val dataset: ", len(val_image_dataset))
            print("len psnr_val_image_dataset dataset: ", len(psnr_val_image_dataset))

            class DatasetWrapper(torch.utils.data.Dataset):
                def __init__(self, image_dataset):
                    super().__init__()

                    self.image_dataset = image_dataset

                def __len__(self):
                    return len(self.image_dataset)

                def __getitem__(self, idx: int):
                    return self.image_dataset[idx], torch.tensor(0) 
            psnr_val_image_dataset = DatasetWrapper(psnr_val_image_dataset)

            #x = val_image_dataset[0]
            #print(x.min(), x.max())
        elif args.model_trained_on == "aapm":
     
            #val_image_dataset = AAPMDataset(
            #    part="test",
            #    base_path=args.dataset.root
            #    )
            #val_image_dataset = torch.utils.data.Subset(
            #    val_image_dataset, np.arange(1, len(val_image_dataset), 10)
            #)
            val_image_dataset = AAPMDataset()
            val_image_dataset = torch.utils.data.Subset(
                val_image_dataset, np.arange(1, len(val_image_dataset), 20)
            )
            psnr_val_image_dataset = val_image_dataset
            print("len val dataset: ", len(val_image_dataset))

        else:
            raise NotImplementedError

        if args.inversion_task == "sr":  # super resolution
            scale = round(args.forward_op.scale)

            likelihood = Superresolution(
                scale=scale, sigma_y=args.forward_op.noise_std, device=args.device
            )
        elif args.inversion_task == "blur":
            # kernel = torch.ones((1, 1, 3, 3)) / 9
            # likelihood = Blur(
            #     filter=kernel,
            #     sigma_y=args.forward_op.noise_std,
            #     device=args.device,
            # )
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

        # sampler = DistributedSampler(
        #     psnr_val_image_dataset, shuffle=False, drop_last=False
        # )
        psnr_val_dl = torch.utils.data.DataLoader(
            psnr_val_image_dataset,
            pin_memory=True,
            batch_size=args.validation.psnr_batch_size,
            shuffle=False,
            drop_last=False,
        )

        log_dir = f"benchmark/{args.inversion_task}/{args.model_trained_on}/test/controlnet"
        log_dir = os.path.join(log_dir, f'{time.strftime("%d-%m-%Y-%H-%M-%S")}')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        print("----------------------------\n")
        print("RESULTS WILL BE LOGGED TO: ", log_dir)

        with open(os.path.join(log_dir, "report.yaml"), "w") as file:
            try:
                yaml.dump(args, file)
            except AttributeError:
                yaml.dump(OmegaConf.to_container(args, resolve=True), file)


        calculate_total_psnr(
            pretrained_score=pretrained_model.to(args.device),
            control_net=control_net.to(args.device),
            likelihood=likelihood,
            val_dataloader=psnr_val_dl,
            diffusion=sde,
            device=args.device,
            cfg_model=args.finetune_model_config,
            val_kwargs={
                "batch_size": args.validation.batch_size,
                "num_steps": args.validation.num_steps,
                "eta": args.validation.eta,
                "sample_freq": args.validation.sample_freq,
                "psnr_sample_freq": args.validation.psnr_sample_freq,
                "psnr_batch_size": args.validation.psnr_batch_size,
                "rescale_image": model_config.data.rescaled,
            },
            save_images=False,
            save_path=log_dir,
            image_path=image_path,
        )



if __name__ == "__main__":
    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        coordinator(config)

    app.run(_main)
