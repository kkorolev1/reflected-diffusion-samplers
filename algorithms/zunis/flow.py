from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp

Array = jax.Array
MLPParams = tuple[tuple[Array, Array], ...]
FlowParams = tuple[MLPParams, ...]


@dataclass(frozen=True)
class FlowSpec:
    """Static architecture information for a bounded spline coupling flow."""

    dim: int
    num_bins: int
    masks: tuple[Array, ...]
    min_derivative: float = 1e-3
    eps: float = 1e-6


def _init_mlp(
    key: Array,
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
) -> MLPParams:
    """Initialize an MLP whose final layer is zero, yielding an identity flow."""
    sizes = (input_dim, *tuple(int(v) for v in hidden_sizes), output_dim)
    keys = jax.random.split(key, len(sizes) - 1)
    layers: list[tuple[Array, Array]] = []

    for i, (fan_in, fan_out, layer_key) in enumerate(
        zip(sizes[:-1], sizes[1:], keys)
    ):
        if i == len(sizes) - 2:
            weight = jnp.zeros((fan_in, fan_out), dtype=jnp.float32)
            bias = jnp.zeros((fan_out,), dtype=jnp.float32)
        else:
            scale = jnp.sqrt(2.0 / max(fan_in, 1))
            weight = scale * jax.random.normal(
                layer_key, (fan_in, fan_out), dtype=jnp.float32
            )
            bias = jnp.zeros((fan_out,), dtype=jnp.float32)
        layers.append((weight, bias))

    return tuple(layers)


def _mlp_apply(params: MLPParams, x: Array) -> Array:
    h = x
    for i, (weight, bias) in enumerate(params):
        h = h @ weight + bias
        if i < len(params) - 1:
            h = jax.nn.relu(h)
    return h


def init_bounded_spline_flow(
    key: Array,
    *,
    dim: int,
    num_layers: int = 8,
    num_bins: int = 10,
    hidden_sizes: Sequence[int] = (256, 256),
    min_derivative: float = 1e-3,
    eps: float = 1e-6,
) -> tuple[FlowParams, FlowSpec]:
    """Create an identity-initialized piecewise-quadratic coupling flow.

    Each scalar map has a piecewise-linear positive derivative. Integrating
    that derivative gives a monotone piecewise-quadratic bijection of [0, 1].
    This is the bounded spline construction used by the ZüNIS/NIS family.
    """
    if dim <= 0:
        raise ValueError("dim must be positive")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    if min_derivative <= 0:
        raise ValueError("min_derivative must be positive")

    layer_keys = jax.random.split(key, num_layers)
    masks: list[Array] = []
    params: list[MLPParams] = []

    for layer_index, layer_key in enumerate(layer_keys):
        if dim == 1:
            mask = jnp.zeros((1,), dtype=jnp.float32)
        else:
            mask = ((jnp.arange(dim) + layer_index) % 2 == 0).astype(jnp.float32)
        masks.append(mask)
        params.append(
            _init_mlp(
                layer_key,
                input_dim=dim,
                hidden_sizes=hidden_sizes,
                output_dim=dim * (num_bins + 1),
            )
        )

    spec = FlowSpec(
        dim=dim,
        num_bins=num_bins,
        masks=tuple(masks),
        min_derivative=float(min_derivative),
        eps=float(eps),
    )
    return tuple(params), spec


def _normalized_heights(raw: Array, spec: FlowSpec) -> Array:
    """Convert unconstrained outputs into positive derivatives of unit area."""
    heights = jax.nn.softplus(raw) + spec.min_derivative
    bin_areas = 0.5 * (heights[..., :-1] + heights[..., 1:]) / spec.num_bins
    total_area = jnp.sum(bin_areas, axis=-1, keepdims=True)
    return heights / total_area


def _spline_parameters(
    layer_params: MLPParams,
    mask: Array,
    x: Array,
    spec: FlowSpec,
) -> Array:
    conditioned = x * mask
    raw = _mlp_apply(layer_params, conditioned)
    raw = raw.reshape(x.shape[:-1] + (spec.dim, spec.num_bins + 1))
    return _normalized_heights(raw, spec)


def _gather_last(x: Array, index: Array) -> Array:
    return jnp.take_along_axis(x, index[..., None], axis=-1)[..., 0]


def _piecewise_quadratic_forward(
    x: Array,
    heights: Array,
    spec: FlowSpec,
) -> tuple[Array, Array]:
    """Apply the monotone spline coordinatewise and return log |dy/dx|."""
    x = jnp.clip(x, spec.eps, 1.0 - spec.eps)
    scaled = x * spec.num_bins
    index = jnp.minimum(jnp.floor(scaled).astype(jnp.int32), spec.num_bins - 1)
    local = scaled - index

    h0 = _gather_last(heights[..., :-1], index)
    h1 = _gather_last(heights[..., 1:], index)
    delta = h1 - h0

    bin_areas = 0.5 * (heights[..., :-1] + heights[..., 1:]) / spec.num_bins
    y_knots = jnp.concatenate(
        [jnp.zeros_like(bin_areas[..., :1]), jnp.cumsum(bin_areas, axis=-1)],
        axis=-1,
    )
    y0 = _gather_last(y_knots[..., :-1], index)

    y = y0 + (h0 * local + 0.5 * delta * local**2) / spec.num_bins
    derivative = h0 + delta * local
    return jnp.clip(y, spec.eps, 1.0 - spec.eps), jnp.log(derivative)


def _piecewise_quadratic_inverse(
    y: Array,
    heights: Array,
    spec: FlowSpec,
) -> tuple[Array, Array]:
    """Invert the monotone spline and return log |dx/dy|."""
    y = jnp.clip(y, spec.eps, 1.0 - spec.eps)
    bin_areas = 0.5 * (heights[..., :-1] + heights[..., 1:]) / spec.num_bins
    y_knots = jnp.concatenate(
        [jnp.zeros_like(bin_areas[..., :1]), jnp.cumsum(bin_areas, axis=-1)],
        axis=-1,
    )

    index = jnp.sum(y[..., None] >= y_knots[..., 1:], axis=-1)
    index = jnp.clip(index, 0, spec.num_bins - 1).astype(jnp.int32)

    y0 = _gather_last(y_knots[..., :-1], index)
    h0 = _gather_last(heights[..., :-1], index)
    h1 = _gather_last(heights[..., 1:], index)
    delta = h1 - h0
    target = jnp.maximum(y - y0, 0.0)

    discriminant = jnp.maximum(
        h0**2 + 2.0 * delta * spec.num_bins * target,
        0.0,
    )
    sqrt_discriminant = jnp.sqrt(discriminant)
    nonlinear_local = (
        2.0 * spec.num_bins * target / jnp.maximum(h0 + sqrt_discriminant, 1e-12)
    )
    linear_local = spec.num_bins * target / h0
    local = jnp.where(jnp.abs(delta) > 1e-6, nonlinear_local, linear_local)
    local = jnp.clip(local, 0.0, 1.0)

    x = (index.astype(y.dtype) + local) / spec.num_bins
    derivative = h0 + delta * local
    return jnp.clip(x, spec.eps, 1.0 - spec.eps), -jnp.log(derivative)


def flow_forward(
    params: FlowParams,
    spec: FlowSpec,
    u: Array,
) -> tuple[Array, Array]:
    """Map base-uniform points to target space and return log |dx/du|."""
    x = jnp.asarray(u)
    log_det = jnp.zeros(x.shape[:-1], dtype=x.dtype)

    for layer_params, mask in zip(params, spec.masks):
        heights = _spline_parameters(layer_params, mask, x, spec)
        transformed, coordinate_log_det = _piecewise_quadratic_forward(x, heights, spec)
        transform_mask = 1.0 - mask
        x = mask * x + transform_mask * transformed
        log_det = log_det + jnp.sum(transform_mask * coordinate_log_det, axis=-1)

    return x, log_det


def flow_inverse(
    params: FlowParams,
    spec: FlowSpec,
    x: Array,
) -> tuple[Array, Array]:
    """Map cube points to the base and return log |du/dx|."""
    u = jnp.asarray(x)
    log_det = jnp.zeros(u.shape[:-1], dtype=u.dtype)

    for layer_params, mask in reversed(tuple(zip(params, spec.masks))):
        # The conditioning coordinates are unchanged by this coupling layer.
        heights = _spline_parameters(layer_params, mask, u, spec)
        transformed, coordinate_log_det = _piecewise_quadratic_inverse(u, heights, spec)
        transform_mask = 1.0 - mask
        u = mask * u + transform_mask * transformed
        log_det = log_det + jnp.sum(transform_mask * coordinate_log_det, axis=-1)

    return u, log_det


def flow_log_prob(params: FlowParams, spec: FlowSpec, x: Array) -> Array:
    """Evaluate the normalized proposal density on [0,1]^d."""
    x = jnp.asarray(x)
    inside = jnp.all((x > 0.0) & (x < 1.0), axis=-1)
    _, inverse_log_det = flow_inverse(params, spec, x)
    return jnp.where(inside, inverse_log_det, -jnp.inf)


def sample_and_log_prob(
    key: Array,
    params: FlowParams,
    spec: FlowSpec,
    num_samples: int,
) -> tuple[Array, Array]:
    """Draw proposal samples and return their log density."""
    u = jax.random.uniform(
        key,
        shape=(num_samples, spec.dim),
        minval=spec.eps,
        maxval=1.0 - spec.eps,
    )
    x, forward_log_det = flow_forward(params, spec, u)
    return x, -forward_log_det
