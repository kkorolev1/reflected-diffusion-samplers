from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

PyTree = Any


@dataclass
class AdamState:
    count: jax.Array
    first_moment: PyTree
    second_moment: PyTree


def adam_init(params: PyTree) -> AdamState:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return AdamState(
        count=jnp.asarray(0, dtype=jnp.int32),
        first_moment=zeros,
        second_moment=zeros,
    )


def global_norm(tree: PyTree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


def clip_by_global_norm(tree: PyTree, max_norm: float) -> tuple[PyTree, jax.Array]:
    norm = global_norm(tree)
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-12))
    return jax.tree_util.tree_map(lambda x: x * scale, tree), norm


def adam_update(
    params: PyTree,
    grads: PyTree,
    state: AdamState,
    *,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    max_grad_norm: float | None = None,
) -> tuple[PyTree, AdamState, jax.Array]:
    if max_grad_norm is not None:
        grads, grad_norm = clip_by_global_norm(grads, max_grad_norm)
    else:
        grad_norm = global_norm(grads)

    count = state.count + 1
    first = jax.tree_util.tree_map(
        lambda m, g: beta1 * m + (1.0 - beta1) * g,
        state.first_moment,
        grads,
    )
    second = jax.tree_util.tree_map(
        lambda v, g: beta2 * v + (1.0 - beta2) * jnp.square(g),
        state.second_moment,
        grads,
    )
    correction1 = 1.0 - beta1**count
    correction2 = 1.0 - beta2**count
    updates = jax.tree_util.tree_map(
        lambda m, v: learning_rate * (m / correction1) / (
            jnp.sqrt(v / correction2) + eps
        ),
        first,
        second,
    )
    new_params = jax.tree_util.tree_map(lambda p, u: p - u, params, updates)
    return new_params, AdamState(count, first, second), grad_norm
