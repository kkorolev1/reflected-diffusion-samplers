from __future__ import annotations

import jax
import jax.numpy as jnp

from algorithms.zunis.flow import FlowParams, FlowSpec, flow_log_prob

Array = jax.Array


def estimate_log_z(log_weights: Array) -> Array:
    """Stable log of the arithmetic mean of importance weights."""
    return jax.scipy.special.logsumexp(log_weights) - jnp.log(log_weights.shape[0])


def dkl_backward_loss(
    params: FlowParams,
    spec: FlowSpec,
    x: Array,
    log_r: Array,
    log_survey: Array,
) -> Array:
    """Backward estimator of the distributional-KL objective.

    Up to constants independent of the flow, the objective is

        -E_{x~s}[(r(x)/s(x)) log q_theta(x)].

    We normalize the batch weights to mean one. This only multiplies the
    stochastic objective by a positive, theta-independent batch constant and
    substantially improves numerical conditioning.
    """
    log_importance = log_r - log_survey
    log_normalizer = estimate_log_z(log_importance)
    weights = jnp.exp(log_importance - log_normalizer)
    weights = jax.lax.stop_gradient(jnp.where(jnp.isfinite(weights), weights, 0.0))
    log_q = flow_log_prob(params, spec, x)
    loss = -jnp.mean(weights * log_q)
    has_mass = jnp.any(jnp.isfinite(log_importance))
    return jnp.where(has_mass, loss, jnp.asarray(jnp.inf, dtype=loss.dtype))


def variance_backward_loss(
    params: FlowParams,
    spec: FlowSpec,
    x: Array,
    log_r: Array,
    log_survey: Array,
) -> Array:
    """Log second-moment loss from backward/survey samples.

        log E_{x~s}[r(x)^2 / (s(x) q_theta(x))].

    The logarithm is monotone, so it has the same minimizer as the raw second
    moment while avoiding overflow for sharply peaked integrands.
    """
    log_q = flow_log_prob(params, spec, x)
    log_terms = 2.0 * log_r - log_survey - log_q
    loss = estimate_log_z(log_terms)
    return jnp.where(jnp.isfinite(loss), loss, jnp.asarray(jnp.inf, dtype=loss.dtype))


def importance_diagnostics(log_weights: Array) -> dict[str, Array]:
    """Return stable diagnostics for an importance-weight batch."""
    log_sum_w = jax.scipy.special.logsumexp(log_weights)
    log_sum_w2 = jax.scipy.special.logsumexp(2.0 * log_weights)
    ess = jnp.exp(2.0 * log_sum_w - log_sum_w2)
    finite = jnp.isfinite(log_weights)
    safe = jnp.where(finite, log_weights, jnp.nan)
    return {
        "ess": ess,
        "ess_fraction": ess / log_weights.shape[0],
        "log_weight_mean": jnp.nanmean(safe),
        "log_weight_std": jnp.nanstd(safe),
        "log_weight_max": jnp.nanmax(safe),
    }
