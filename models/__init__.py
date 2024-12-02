import torch
import torch.distributed as dist
from hydra.utils import call
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.checkpoints import ckpt_path_adm
from utils.distributed import get_logger


def build_model(cfg):
    logger = get_logger("model", cfg)
    model = call(cfg.model)
    map_location = {"cuda:0": f"cuda:{dist.get_rank()}"}
    model_ckpt = ckpt_path_adm(cfg.model.ckpt, cfg)
    logger.info(f"Loading model from {model_ckpt}")
    model.load_state_dict(torch.load(model_ckpt, map_location=map_location, weights_only=True))
    classifier = call(cfg.classifier)

    if getattr(cfg.classifier, "ckpt", None):
        classifier_ckpt = ckpt_path_adm(cfg.classifier.ckpt, cfg)
        logger.info(f"Loading classifier from {classifier_ckpt}..")
        classifier.load_state_dict(
            torch.load(classifier_ckpt, map_location=map_location, weights_only=True)
        )
    if classifier is not None:
        classifier.cuda(dist.get_rank())
        classifier = DDP(
            classifier,
            device_ids=[dist.get_rank()],
            output_device=[dist.get_rank()],
        )
    if getattr(cfg, "htransform_model", None):
        htransform_model = call(cfg.htransform_model)
        htransform_model.cuda(dist.get_rank())
        if cfg.htransform_model.ckpt_path is not None:
            htransform_model.load_state_dict(torch.load(cfg.htransform_model.ckpt_path))
        htransform_model = DDP(
            htransform_model,
            device_ids=[dist.get_rank()],
            output_device=[dist.get_rank()],
        )
    else:
        htransform_model = None
    model.cuda(dist.get_rank())

    model = DDP(model, device_ids=[dist.get_rank()], output_device=[dist.get_rank()])
    return model, classifier, htransform_model
