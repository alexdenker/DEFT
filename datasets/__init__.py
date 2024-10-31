import os

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from datasets.aapm import AAPMDataset

# SELENE
from datasets.ffhq import get_ffhq_dataset, get_ffhq_loader
from datasets.imagenet import get_imagenet_dataset, get_imagenet_loader
from datasets.lodopab import LoDoPabDatasetFromDival
from utils.distributed import get_logger

# LOCAL MACHINE
# from pgdm.datasets.ffhq import get_ffhq_dataset, get_ffhq_loader
# from pgdm.datasets.imagenet import get_imagenet_dataset, get_imagenet_loader
# from pgdm.utils.distributed import get_logger


class ZipDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        assert all(len(dataset) == len(datasets[0]) for dataset in datasets)

    def __getitem__(self, index):
        return [dataset[index] for dataset in self.datasets]

    def __len__(self):
        return len(self.datasets[0])


def build_one_dataset(cfg, dataset_attr="dataset", return_splits=False):
    logger = get_logger("dataset", cfg)
    exp_root = cfg.exp.root
    cfg_dataset = getattr(cfg, dataset_attr)
    try:
        samples_root = cfg.exp.samples_root
        exp_name = cfg.exp.name
        samples_root = os.path.join(exp_root, samples_root, exp_name)
    except Exception:
        samples_root = ""
        logger.info("Does not attempt to prune existing samples (overwrite=False).")
    if "ImageNet" in cfg_dataset.name:
        overwrite = getattr(cfg.exp, "overwrite", True)
        dset = get_imagenet_dataset(
            overwrite=overwrite, samples_root=samples_root, **cfg_dataset
        )
        dist.barrier()

        if return_splits:
            # TODO: Add the code to select less than or more than 1000 images to finetune on
            # Take 100 images to calculate validation metrics on
            val_dset = torch.utils.data.Subset(dset, np.arange(1, len(dset), 100))
            # Take 1000 images to finetune on
            finetune_dset = torch.utils.data.Subset(dset, np.arange(0, len(dset), 10))
            return finetune_dset, val_dset
    if "FFHQ" in cfg_dataset.name:
        dset = get_ffhq_dataset(**cfg_dataset)

        if return_splits:
            raise ValueError("FFHQ dataset has no finetune/val splits by default.")
    if "LoDoPab" in cfg_dataset.name:
        dset = LoDoPabDatasetFromDival(im_size=256)
        dset = dset.lodopab_val

        if return_splits:
            val_dset = torch.utils.data.Subset(dset, np.arange(1, len(dset), 200))
            return dset, val_dset
    if "AAPM" in cfg_dataset.name:
        if not return_splits:
            raise ValueError("AAPM dataset has finetune/val splits by default.")
        else:
            finetune_dset = AAPMDataset(part="val", base_path=cfg_dataset.root)
            val_dset = AAPMDataset(part="test", base_path=cfg_dataset.root)
            return finetune_dset, val_dset
    return dset


def build_loader(cfg, dataset_attr="dataset"):
    if isinstance(dataset_attr, list):
        dsets = []
        for da in dataset_attr:
            cfg_dataset = getattr(cfg, da)
            dset = build_one_dataset(cfg, dataset_attr=da)
            dsets.append(dset)
        dsets = ZipDataset(dsets)
        if "ImageNet" in cfg_dataset.name:
            loader = get_imagenet_loader(dsets, **cfg.loader)
        elif "FFHQ" in cfg_dataset.name:
            loader = get_ffhq_loader(dsets, **cfg.loader)
    else:
        cfg_dataset = getattr(cfg, dataset_attr)
        dset = build_one_dataset(cfg, dataset_attr=dataset_attr)
        if "ImageNet" in cfg_dataset.name:
            loader = get_imagenet_loader(dset, **cfg.loader)
        elif "FFHQ" in cfg_dataset.name:
            loader = get_ffhq_loader(dset, **cfg.loader)

    return loader
