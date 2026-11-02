"""NICE TARGET"""

import functools
import math
import os
import pickle
from typing import List, Optional

import chex
import distrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import linen as nn
from flax.training import checkpoints

from algorithms.common.bounded import rejection_sample_domain
from targets.base_target import Target
from utils.mode_classifier.classifier_model import CNN, FashionMnistCNN
from utils.path_utils import project_path
from targets.cube import Cube
from targets.nice import load_model_nice, classify
from eval.discrepancies import compute_sd

Array = jax.Array


def logit(x, eps=1e-6):
    x = jnp.clip(x, eps, 1.0 - eps)
    return jnp.log(x) - jnp.log1p(-x)


def log_jac_logit(x, eps=1e-6):
    x = jnp.clip(x, eps, 1.0 - eps)
    log_jac = -jnp.log(x) - jnp.log1p(-x)
    return jnp.sum(log_jac, axis=-1)


class NiceCube(Cube):
    def __init__(self, dim, dataset="mnist") -> None:
        self.data_ndim = dim
        self.im_size = int(np.sqrt(dim))

        self.logpx_fn_without_rng, _, self.sample_fn_clean = load_model_nice(
            dataset, dim
        )

        state = checkpoints.restore_checkpoint(
            ckpt_dir=project_path("utils/mode_classifier"),
            target=None,
            prefix="{}_{}x{}_classifier_checkpoint".format(
                dataset, int(math.sqrt(dim)), int(math.sqrt(dim))
            ),
        )

        if dataset == "mnist":
            classifier = CNN()
        elif dataset == "fashion_mnist":
            classifier = FashionMnistCNN()
        else:
            raise NotImplementedError

        self.classify = jax.jit(
            functools.partial(
                classify, cnn=classifier, params=state["params"], im_size=self.im_size
            )
        )
        self._compute_true_logZ = lambda *args: 0.0
        super().__init__(
            dim,
            log_prob_fn=self._log_prob,
        )

    def _log_prob(self, x: chex.Array) -> chex.Array:
        return self.logpx_fn_without_rng(logit(x)) + log_jac_logit(x)

    def sample(
        self, seed: chex.PRNGKey, sample_shape: chex.Shape, constrained=True
    ) -> chex.Array:
        # ignore constrained param for this target
        if len(sample_shape) == 0:
            n_samples = 1
            x = self.sample_fn_clean(seed, n_samples)[0]
        else:
            n_samples = sample_shape[0]
            x = self.sample_fn_clean(seed, n_samples)
        return x

    def compute_sd(self, target_samples, samples, cfg):
        return compute_sd(target_samples, logit(samples), cfg)

    def visualise(
        self,
        samples: chex.Array = None,
        show: bool = False,
        constrained: bool = True,
        map_samples: bool = True,
        grid_n_points: int = 100,
        samples_num: int = 300,
        samples_alpha: float = 0.5,
    ) -> None:
        plt.close()
        n = 64
        if constrained:
            samples = logit(samples)
        x = np.array(samples[:n].reshape(-1, self.im_size, self.im_size))

        n_rows = int(np.sqrt(n))
        fig, ax = plt.subplots(n_rows, n_rows, figsize=(8, 8))

        # Plot each image
        for i in range(n_rows):
            for j in range(n_rows):
                ax[i, j].imshow(x[i * n_rows + j], cmap="gray")
                ax[i, j].axis("off")

        # plt.savefig(os.path.join(project_path('./figures/'), f"{prefix}nice.pdf"), bbox_inches='tight', pad_inches=0.1)
        # Log into wandb
        wb = {"figures/vis": [fig]}
        if show:
            plt.show()

        return wb


if __name__ == "__main__":

    def check(dist):
        N = 10**3
        key = jax.random.PRNGKey(0)
        samples = jax.random.uniform(key, shape=(N, dist.dim))
        log_probs = dist.log_prob(samples, constrained=True)
        print(
            f"\t Min log_prob: {jnp.min(log_probs)}, Max log_prob: {jnp.max(log_probs)}"
        )
        print("\t", dist.Z, dist.log_Z)

    key = jax.random.PRNGKey(0)
    cube = NiceCube(dim=196)
    print(cube.log_Z)
    print(cube.Z)

    check(cube)
    samples = cube.sample(jax.random.PRNGKey(0), (100,), constrained=True)
    print(samples.min(), samples.max())
    cube.visualise(samples, map_samples=False, show=True)
