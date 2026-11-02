import os
from typing import List

import chex
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import jax.random as random
import numpy as np
from functools import partial
import distrax
from matplotlib import pyplot as plt
from scipy.stats import wishart
import matplotlib.patches as patches
from itertools import product

from targets.base_target import Target
from utils.path_utils import project_path
from algorithms.common.bounded import rejection_sample_domain
from matplotlib.colors import LogNorm


class Cube(Target):
    def __init__(
        self,
        dim,
        log_prob_fn=None,
        min_coord=0.0,
        max_coord=1.0,
        mask_outside=True,  # whether to set p(x)=0 outside the domain
        sample_fn=None,
        corrector=0.0,
        is_non_negative=True,
    ) -> None:
        self._dim = dim
        self.min_coord = min_coord
        self.max_coord = max_coord
        self.log_prob_fn = log_prob_fn
        self.mask_outside = mask_outside
        self.sample_fn = sample_fn

        log_Z = self._compute_true_logZ()
        super().__init__(
            dim,
            log_Z,
            can_sample=sample_fn is not None,
            corrector=corrector,
            is_non_negative=is_non_negative,
        )

    @property
    def volume(self) -> int:
        return (self.max_coord - self.min_coord) ** self.dim

    def is_inside(self, x: chex.Array) -> bool:
        return jnp.all((x >= self.min_coord) & (x <= self.max_coord), axis=-1)

    def project(self, x: chex.Array) -> chex.Array:
        return jnp.clip(x, min=self.min_coord, max=self.max_coord)

    def reflect(self, x: chex.Array) -> chex.Array:
        # xm2 = x % 2
        # xm2 = jnp.where(xm2 > 1, 2 - xm2, xm2)
        # return xm2
        length = self.max_coord - self.min_coord
        phase = jnp.mod(x - self.min_coord, 2.0 * length)
        return self.min_coord + jnp.where(
            phase > length,
            2.0 * length - phase,
            phase,
        )

    def rd_to_cube(self, x: chex.Array):
        return (self.max_coord - self.min_coord) * jax.nn.sigmoid(x) + self.min_coord

    def cube_to_rd(self, x: chex.Array):
        x = (x - self.min_coord) / (self.max_coord - self.min_coord)
        eps = jnp.finfo(x.dtype).eps
        x = jnp.clip(x, min=eps, max=1 - eps)
        return jnp.log(x) - jnp.log1p(-x)

    def _add_offset(self, log_reward, offset):
        if offset == 0.0:
            return log_reward
        return jnp.logaddexp(log_reward, jnp.log(offset))

    def sample(
        self, seed: chex.PRNGKey, sample_shape: chex.Shape, constrained: bool = False
    ) -> chex.Array:
        # constrained = True - defined in [a,b]^d
        # constrained = False - defined in R^d via change of variables
        if not constrained:
            samples = self.sample(seed, sample_shape, constrained=True)
            return self.cube_to_rd(samples)

        return rejection_sample_domain(
            seed, sample_shape, self.sample_fn, self.is_inside, self.dim
        )

    def log_prob(self, x: chex.Array, constrained: bool = False) -> chex.Array:
        # constrained = True - defined in [a,b]^d
        # constrained = False - defined in R^d via change of variables
        if not constrained:
            log_jacobian = jnp.sum(
                jax.nn.log_sigmoid(x) + jax.nn.log_sigmoid(-x), axis=-1
            ) + x.shape[-1] * jnp.log(self.max_coord - self.min_coord)
            return self.log_prob(self.rd_to_cube(x), constrained=True) + log_jacobian

        batched = x.ndim == 2
        if not batched:
            x = x[None]

        log_prob = self.log_prob_fn(x)

        if self.mask_outside:
            log_prob = jnp.where(self.is_inside(x), log_prob, -jnp.inf)

        if not batched:
            log_prob = jnp.squeeze(log_prob, axis=0)
        return log_prob

    def _compute_true_logZ(
        self, num_samples: int = 1000000, seed: chex.PRNGKey = None
    ) -> float:
        if seed is None:
            seed = jax.random.PRNGKey(42)
        shape = (num_samples, self.dim)
        seed, subkey = random.split(seed)
        samples = jax.random.uniform(
            subkey, shape=shape, minval=self.min_coord, maxval=self.max_coord
        )
        log_probs = self.log_prob(samples, constrained=True)
        log_volume = self.dim * jnp.log(float(self.max_coord - self.min_coord))
        log_integral = logsumexp(log_probs) - jnp.log(log_probs.shape[0]) + log_volume
        return float(log_integral)

    def _add_cube_border(self, ax: plt.Axes) -> None:
        rect = patches.Rectangle(
            (self.min_coord, self.min_coord),
            self.max_coord - self.min_coord,
            self.max_coord - self.min_coord,
            linewidth=1,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(rect)

    def log_prob_marginal_pair(self, x_2d, i, j, constrained=True):
        x = jnp.zeros((x_2d.shape[0], self.dim))
        x = x.at[:, i].set(x_2d[:, 0])
        x = x.at[:, j].set(x_2d[:, 1])
        return self.log_prob(x, constrained=constrained)

    def visualise(
        self,
        samples: chex.Array = None,
        show: bool = False,
        constrained: bool = True,
        map_samples: bool = True,
        grid_n_points: int = 100,
        samples_num: int = 300,
        samples_alpha: float = 0.5,
        log_scale: bool = False,
    ) -> None:
        # samples are mapped to cube if constrained=True and map_samples=True
        plt.close()
        if constrained:
            border = [self.min_coord - 0.1, self.max_coord + 0.1]
        else:
            border = [
                self.cube_to_rd(jnp.array(self.min_coord + 1e-3)),
                self.cube_to_rd(jnp.array(self.max_coord - 1e-3)),
            ]
        if samples is not None and constrained and map_samples:
            samples = self.rd_to_cube(samples)
        if self.dim == 2:
            fig = plt.figure()
            ax = fig.add_subplot()

            x, y = jnp.meshgrid(
                jnp.linspace(border[0], border[1], grid_n_points),
                jnp.linspace(border[0], border[1], grid_n_points),
            )
            grid = jnp.c_[x.ravel(), y.ravel()]
            log_probs = self.log_prob(grid, constrained=constrained)
            if self.needs_correction:
                log_probs = jnp.logaddexp(log_probs, jnp.log(self.corrector))
            log_probs = jnp.reshape(log_probs, x.shape)
            z = log_probs if log_scale else jnp.exp(log_probs)
            ax.contourf(x, y, z, levels=20, cmap="viridis")
            # mesh = ax.pcolormesh(x, y, pdf_values, cmap="viridis", shading="auto")
            # fig.colorbar(mesh, ax=ax)
            if samples is not None:
                ax.scatter(
                    samples[:samples_num, 0],
                    samples[:samples_num, 1],
                    c="r",
                    alpha=samples_alpha,
                    marker="x",
                )

            if constrained:
                self._add_cube_border(ax)
            ax.set_aspect("equal", adjustable="box")
            wb = {"figures/vis": [fig]}
            if show:
                plt.show()
            return wb

        fig = plt.figure(figsize=(8, 8))
        n_rows = 2
        axs = fig.subplots(n_rows, n_rows, sharex="row", sharey="row")
        for i, j in product(range(n_rows), range(n_rows)):
            x, y = jnp.meshgrid(
                jnp.linspace(border[0], border[1], grid_n_points),
                jnp.linspace(border[0], border[1], grid_n_points),
            )
            grid = jnp.c_[x.ravel(), y.ravel()]
            log_probs = self.log_prob_marginal_pair(
                grid, i, j + 2, constrained=constrained
            )
            if self.needs_correction:
                log_probs = jnp.logaddexp(log_probs, jnp.log(self.corrector))
            log_probs = jnp.reshape(log_probs, x.shape)
            z = log_probs if log_scale else jnp.exp(log_probs)
            axs[i, j].contourf(x, y, z, levels=20, cmap="viridis")
            # mesh = axs[i, j].pcolormesh(
            #     x, y, pdf_values, cmap="viridis", shading="auto"
            # )
            # fig.colorbar(mesh, ax=axs[i, j])
            if samples is not None:
                axs[i, j].scatter(
                    samples[:samples_num, i],
                    samples[:samples_num, j + 2],
                    c="r",
                    alpha=samples_alpha,
                    marker="x",
                )
            if constrained:
                self._add_cube_border(axs[i, j])
            if j == 0:
                axs[i, j].set_ylabel(f"$x_{i + 1}$")
            if i == 1:
                axs[i, j].set_xlabel(f"$x_{j + 1 + 2}$")
            axs[i, j].set_aspect("equal", adjustable="box")
        wb = {"figures/vis": [fig]}
        if show:
            plt.show()
        else:
            plt.close()
        return wb
