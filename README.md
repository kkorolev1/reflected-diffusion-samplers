# Numerical Integration on Bounded Domains via Learned Reflected Diffusions

Official implementation of **Numerical Integration on Bounded Domains via Learned Reflected Diffusions**.

Kirill Korolev, Artur Goldman, Timofei Gritsaev, Tigran Ramazyan, Nikita Morozov

[**Paper ↗**](https://arxiv.org/abs/2610.40499) · [Installation](#installation) · [Usage](#usage) · [Configuration](#configuration)

## Abstract

We study high-dimensional integration on bounded convex domains. Given a non-negative integrand $`r`$, the goal is to estimate

$$
Z = \int_{\mathcal{D}} r(x) \mathrm{d}x,
\qquad \mathcal{D} \subset \mathbb{R}^d.
$$

Standard Quasi-Monte Carlo does not adapt its evaluation points to the integrand, while effective importance sampling requires a proposal that captures its concentration. Learned diffusion samplers can supply such a proposal, but existing methods are formulated on $`\mathbb{R}^d`$ and do not respect the domain constraint $`x \in \mathcal{D}`$.

We instead build the proposal from reflected dynamics that stay in $`\mathcal{D}`$ by construction. A reflected diffusion progressively transforms the target distribution toward the uniform distribution on $`\mathcal{D}`$, and a neural score model is trained to reverse this process. A change-of-measure formula for reflected diffusions yields a training objective without additional boundary terms and an unbiased importance-sampling estimator of $`Z`$.

For non-smooth integrands, where reparameterization gradients may be undefined or uninformative, we derive a REINFORCE estimator of the training gradient using Gaussian proposal densities before reflection, without evaluating reflected transition densities or differentiating $`r`$. We evaluate on numerous synthetic targets and real-world problems, demonstrating strong performance of the proposed method, as well as favorable scaling with the dimension of the problem and the number of points used for the Monte Carlo estimate.

## Installation

**Environment:** Python 3.10.14 · JAX 0.6.2

Run the following commands from the repository root. We recommend an isolated Conda or Mamba environment.

```bash
conda create -n rds python=3.10.14
conda activate rds
```

Install JAX for your hardware. For CUDA 12:

```bash
python -m pip install -U "jax[cuda12]==0.6.2"
```

Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

Choose a training gradient estimator:

| Method | Configuration | Gradient estimator |
| --- | --- | --- |
| **RDS-RP** | `algorithm.loss=kl_reparam` | Reparameterization trick |
| **RDS-RF** | `algorithm.loss=kl_reinforce` | REINFORCE |

For example, train **RDS-RP** on the Genz continuous target:

```bash
python run.py algorithm=rds target=genz_continuous \
  algorithm.loss=kl_reparam logger.use_comet=false
```

By default results of the experiment will be stored locally in `logs` directory. To use Comet, configure `logger.comet` in [configs/setup.yaml](configs/setup.yaml) and set `logger.use_comet=true`.

### Configuration

Experiments use Hydra configuration overrides in the form `key=value`. Select a target using a filename from [configs/target](configs/target), without the `.yaml` extension.

For example, change the dimension, training iterations, and random seed:

```bash
python run.py algorithm=rds target=genz_continuous \
  algorithm.loss=kl_reparam target.dim=10 algorithm.iters=4000 \
  seed=42 logger.use_comet=false
```

| Configuration | Location |
| --- | --- |
| Targets and target-specific hyperparameters | [configs/target](configs/target) |
| RDS settings and default loss | [configs/algorithm/rds.yaml](configs/algorithm/rds.yaml) |
| Evaluation settings, random seed, and output path | [configs/base_conf.yaml](configs/base_conf.yaml) |
| Logging backends and Hydra settings | [configs/setup.yaml](configs/setup.yaml) |

## Acknowledgments

This repository builds on the code of Blessing et al. (2024) for *Beyond ELBOs: A Large-Scale Evaluation of Variational Methods for Sampling*.
