# [DEFT: Efficient Finetuning of Conditional Diffusion Models by Learning the Generalised h-transform](https://arxiv.org/pdf/2406.01781)

Alexander Denker*, Francisco Vargas*, Shreyas Padhy*, Kieran Didi*, Simon Mathis*, Vincent Dutordoir, Riccardo Barbano, Emile Mathieu, Urszula Julia Komorowska, Pietro Lio 

**This repository will contain the implementation of DEFT as a fork of RED-diff.**



![deft](images/DEFT_Sampling.png)

### Abstract
Generative modelling paradigms based on denoising diffusion processes have emerged as a leading candidate for conditional sampling in inverse problems. In many real-world applications, we often have access to large, expensively trained unconditional diffusion models, which we aim to exploit for improving conditional sampling. Most recent approaches are motivated heuristically and lack a unifying framework, obscuring connections between them. Further, they often suffer from issues such as being very sensitive to hyperparameters, being expensive to train or needing access to weights hidden behind a closed API. In this work, we unify conditional training and sampling using the mathematically well-understood Doob’s h-transform. This new perspective allows us to unify many existing methods under a common umbrella. Under this framework, we propose DEFT (Doob’s h-transform Efficient FineTuning), a new approach for conditional generation that
simply fine-tunes a very small network to quickly learn the conditional h-transform, while keeping the larger unconditional network unchanged. DEFT is much faster
than existing baselines while achieving state-of-the-art performance across a variety of linear and non-linear benchmarks. On image reconstruction tasks, we achieve
speedups of up to 1.6x, while having the best perceptual quality on natural images and reconstruction performance on medical images.

## Getting Started

First, we install `uv` to manage all dependencies. See the [uv documentation](https://docs.astral.sh/uv/getting-started/) for more details. As an example on macOS and Linux, you can run 
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Once `uv` is installed, you can run `uv sync` to install all dependencies, which will create a `.venv` in the directory. Now, for running any python script, you can run `uv run <script> <args>`.

### Super-resolution Example 

```bash
uv run main.py \
algo=deft \
algo.deg=sr4 \
exp.name=sr4_deft \
algo.finetune_args.batch_size=16 \
algo.finetune_args.epochs=200 \
algo.finetune_args.lr=0.0005 \
algo.val_args.batch_size=10 \
algo.val_args.sample_freq=5 \
exp.overwrite=True \
exp.samples_root=samples \
exp.save_deg=True \
exp.save_evolution=True \
exp.save_ori=True \
exp.seed=3 \
exp.smoke_test=-1 \
htransform_model.in_channels=9 \
htransform_model.num_channels=32 \
htransform_model.num_head_channels=16 \
htransform_model.out_channels=3 \
likelihood.forward_op.noise_std=0.0 \
likelihood.forward_op.scale=4.0 \
loader=imagenet256_ddrmpp \
loader.batch_size=10 \
dist.num_processes_per_node=1
```

### Inpainting Example 

```
python python run_supervised_finetuning.py --config configs/default_config.py:finetuning:imagenet:inp   --config.finetune_model_config.in_channels 13 --config.forward_op.noise_std 0.0 --config.wandb.log --config.wandb.name "finetuning, inp" --config.finetune_model_config.use_residual --config.training.lr_annealing  --config.dataset.root "path/to/imagenet

```


## Citation

If you find this work helpful please cite:

``` 
@article{denker2024deft,
  title={DEFT: Efficient Finetuning of Conditional Diffusion Models by Learning the Generalised $h$-transform},
  author={Denker, Alexander and Vargas, Francisco and Padhy, Shreyas and Didi, Kieran and Mathis, Simon and Dutordoir, Vincent and Barbano, Riccardo and Mathieu, Emile and Komorowska, Urszula Julia and Lio, Pietro},
  journal={arXiv preprint arXiv:2406.01781},
  year={2024}
}
```