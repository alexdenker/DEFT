# Commands to replicate paper results

## Superresolution (x4)

### REDDIFF

```bash
uv run main.py \
algo=reddiff \
algo.deg=sr4 \
exp.name=sr4_reddiff \
algo.awd=True \
algo.eta=1.0 \
algo.grad_term_weight=0.25 \
algo.lr=0.25 \
algo.sigma_y=0.0 \
exp.num_steps=100 \
exp.overwrite=True \
exp.samples_root=samples \
exp.save_deg=True \
exp.save_evolution=True \
exp.save_ori=True \
exp.seed=3 \
exp.smoke_test=-1 \
loader.batch_size=10 \
loader=imagenet256_ddrmpp \
dist.num_processes_per_node=1
```

### DEFT

Replicating https://wandb.ai/shreyaspadhy/diff-models_finetuning/runs/kdbbaeqk?nw=nwusershreyaspadhy

```bash
uv run main.py \
algo=deft \
algo.deg=sr4 \
exp.name=sr4_deft \
algo.finetune_args.batch_size=16 \
algo.finetune_args.epochs=200 \
algo.finetune_args.lr=0.0005 \
algo.val_args.batch_size=10 \
algo.val_args.eta=1 \
algo.val_args.sample_freq=5 \
exp.overwrite=True \
exp.samples_root=samples \
exp.save_deg=True \
exp.save_evolution=False \
exp.save_ori=True \
exp.seed=3 \
exp.smoke_test=-1 \
htransform_model.in_channels=9 \
htransform_model.num_channels=64 \
htransform_model.num_head_channels=16 \
htransform_model.out_channels=3 \
likelihood.forward_op.noise_std=0.0 \
likelihood.forward_op.scale=4.0 \
loader=imagenet256_ddrmpp \
loader.batch_size=10 \
dist.num_processes_per_node=1 \
wandb_config.log=True \
htransform_model.ckpt_path=/home/sp2058/DEFT/outputs/model_ckpts/sr4_deft_8fi2wuj3/model_old.pt
```

# TODO: Do sampling and eval with ema model.

```bash
uv run eval.py \
save_path=outputs/samples/sr4_deft_8fi2wuj3/ \
dist.num_processes_per_node=1 \
dist.port=12356 \
wandb_config.log=True
```
