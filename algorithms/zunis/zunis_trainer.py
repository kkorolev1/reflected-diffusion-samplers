from __future__ import annotations

import math
from functools import partial
from time import time
from typing import Callable

import jax
import jax.numpy as jnp

from algorithms.zunis.flow import (
    FlowParams,
    FlowSpec,
    init_bounded_spline_flow,
    sample_and_log_prob,
)
from algorithms.zunis.losses import (
    dkl_backward_loss,
    importance_diagnostics,
    variance_backward_loss,
)
from algorithms.zunis.optim import adam_init, adam_update
from eval.discrepancies import compute_sd
from eval.utils import (
    compute_statistics_over_runs,
    extract_last_entry,
    save_samples,
    update_train_estimate,
)
from utils.logger import log
from utils.print_utils import print_results

Array = jax.Array


def make_loss(loss_name: str) -> Callable[..., Array]:
    if loss_name == "dkl":
        return dkl_backward_loss
    if loss_name == "variance":
        return variance_backward_loss
    raise ValueError(f"Unknown ZüNIS loss '{loss_name}'. Expected 'dkl' or 'variance'.")


def sample_survey(key, params, spec, batch_size, strategy, step):
    if strategy == "flat" or (strategy == "forward_flat_init" and step == 0):
        samples = jax.random.uniform(
            key, (batch_size, spec.dim), minval=spec.eps, maxval=1.0 - spec.eps
        )
        return samples, jnp.zeros((batch_size,), dtype=samples.dtype)

    if strategy in ("forward", "forward_flat_init"):
        return sample_and_log_prob(key, params, spec, batch_size)

    raise ValueError(
        "survey_strategy must be one of {'flat', 'forward', 'forward_flat_init'}"
    )


def compute_estimates_from_weights(log_is_weights, target):
    log_Z = jax.scipy.special.logsumexp(log_is_weights) - jnp.log(
        log_is_weights.shape[0]
    )

    if target.needs_correction:
        log_Z, sign = jax.scipy.special.logsumexp(
            jnp.array([log_Z, jnp.log(target.corrector * target.volume)]),
            b=jnp.array([1.0, -1.0]),
            return_sign=True,
        )
        Z = sign * jnp.exp(log_Z)
        log_Z = jnp.where(sign >= 0, log_Z, jnp.nan)
    else:
        Z = jnp.exp(log_Z)

    return log_Z, Z


def rnd_zunis(key, params, *, spec, target, batch_size):
    samples, log_q = sample_and_log_prob(key, params, spec, batch_size)
    log_is_weights = target.log_prob(samples) - log_q
    log_Z, Z = compute_estimates_from_weights(log_is_weights, target)
    return samples, log_Z, Z


def get_eval_fn(rnd, target, cfg):
    n_runs = cfg.n_runs
    rnd_eval = jax.jit(rnd, static_argnames=("batch_size",))

    logger = {
        "logZ/value": [],
        "logZ/delta": [],
        "Z/value": [],
        "Z/delta": [],
        "sd/value": [],
        "stats/step": [],
        "stats/wallclock": [],
        "stats/nfe": [],
    }

    for metric_name in ["logZ/value", "logZ/delta", "Z/value", "Z/delta", "sd/value"]:
        logger[f"{metric_name}/mean"] = []
        logger[f"{metric_name}/std"] = []

    def _short_eval(params, key, log_media=False, iter_step=0):
        eval_samples = (iter_step + 1) * cfg.algorithm.batch_size
        samples, log_Z, Z = rnd_eval(key, params, batch_size=eval_samples)
        logger["logZ/value"].append(log_Z)
        logger["Z/value"].append(Z)

        if target.log_Z is not None:
            logger["logZ/delta"].append(jnp.abs(log_Z - target.log_Z))
        if target.Z is not None:
            logger["Z/delta"].append(jnp.abs(Z - target.Z))

        if log_media:
            logger.update(target.visualise(samples=samples))

        if cfg.compute_sd:
            key, key_gen = jax.random.split(key)
            target_samples = target.sample(key_gen, (eval_samples,))
            if hasattr(target, "compute_sd"):
                logger["sd/value"].append(
                    target.compute_sd(target_samples, samples, cfg)
                )
            else:
                logger["sd/value"].append(compute_sd(target_samples, samples, cfg))

        if cfg.save_samples and log_media:
            save_samples(cfg, logger, samples, iter_step)

        return logger

    def short_eval(params, key, iter_step=0):
        keys = jax.random.split(key, n_runs)
        logger = _short_eval(params, keys[0], log_media=True, iter_step=iter_step)
        for k in keys[1:]:
            logger = _short_eval(params, k, iter_step=iter_step)

        compute_statistics_over_runs(logger, "logZ/value", n_runs)
        if target.log_Z is not None:
            compute_statistics_over_runs(logger, "logZ/delta", n_runs)
        compute_statistics_over_runs(logger, "Z/value", n_runs)
        if target.Z is not None:
            compute_statistics_over_runs(logger, "Z/delta", n_runs)
        if cfg.compute_sd:
            compute_statistics_over_runs(logger, "sd/value", n_runs)

        return logger

    return short_eval, logger


def zunis_trainer(cfg, target):
    assert target.is_non_negative, "ZüNIS currently supports nonnegative targets only"

    key_gen = jax.random.PRNGKey(cfg.seed)
    alg_cfg = cfg.algorithm
    batch_size = int(alg_cfg.batch_size)
    iters = int(alg_cfg.iters)
    reuse_epochs = int(alg_cfg.reuse_epochs)
    survey_strategy = str(alg_cfg.survey_strategy)

    target.visualise = partial(target.visualise, constrained=True, map_samples=False)
    target.sample = partial(target.sample, constrained=True)
    target.log_prob = partial(target.log_prob, constrained=True)

    print(f"True log_Z: {target.log_Z}")
    print(f"True Z: {target.Z}")
    print(f"Needs correction: {target.needs_correction}, corrector: {target.corrector}")

    key, key_gen = jax.random.split(key_gen)
    params, spec = init_bounded_spline_flow(
        key,
        dim=target.dim,
        num_layers=int(alg_cfg.num_layers),
        num_bins=int(alg_cfg.num_bins),
        hidden_sizes=tuple(alg_cfg.hidden_sizes),
        min_derivative=float(alg_cfg.min_derivative),
    )

    optimizer_state = adam_init(params)
    loss_fn = make_loss(str(alg_cfg.loss))
    learning_rate = float(alg_cfg.lr)
    max_grad_norm = float(alg_cfg.max_grad_norm)

    value_and_grad = jax.jit(
        jax.value_and_grad(lambda p, x, log_r, log_s: loss_fn(p, spec, x, log_r, log_s))
    )

    rnd_eval = partial(rnd_zunis, spec=spec, target=target)
    eval_fn, logger = get_eval_fn(rnd_eval, target, cfg)

    eval_freq = max(iters // cfg.n_evals, 1)
    timer = 0.0
    running_metrics = {"Z_train/value": None}

    for step in range(iters):
        iter_time = time()
        key, key_gen = jax.random.split(key_gen)

        samples, log_survey = sample_survey(
            key, params, spec, batch_size, survey_strategy, step
        )
        samples = jax.lax.stop_gradient(samples)
        log_survey = jax.lax.stop_gradient(log_survey)
        log_r = jax.lax.stop_gradient(target.log_prob(samples))
        log_is_weights = log_r - log_survey

        _, Z = compute_estimates_from_weights(log_is_weights, target)
        running_metrics["Z_train/value"] = update_train_estimate(
            Z, running_metrics["Z_train/value"], batch_size, step * batch_size
        )

        finite_mass = bool(jnp.any(jnp.isfinite(log_r)))
        last_loss = jnp.asarray(jnp.nan)
        last_grad_norm = jnp.asarray(jnp.nan)
        survey_ess_fraction = jnp.asarray(0.0)
        survey_log_weight_std = jnp.asarray(jnp.nan)

        if finite_mass:
            survey_diag = importance_diagnostics(log_is_weights)
            survey_ess_fraction = survey_diag["ess_fraction"]
            survey_log_weight_std = survey_diag["log_weight_std"]

            for _ in range(reuse_epochs):
                last_loss, grads = value_and_grad(params, samples, log_r, log_survey)
                params, optimizer_state, last_grad_norm = adam_update(
                    params,
                    grads,
                    optimizer_state,
                    learning_rate=learning_rate,
                    max_grad_norm=max_grad_norm,
                )

        timer += time() - iter_time

        if (step % eval_freq == 0) or (step == iters - 1):
            log(
                {
                    "loss": last_loss,
                    "grad_norm": last_grad_norm,
                    "survey_ess_fraction": survey_ess_fraction,
                    "survey_log_weight_std": survey_log_weight_std,
                },
                step=step,
            )

            key, key_gen = jax.random.split(key_gen)
            logger["stats/step"].append(step)
            logger["stats/wallclock"].append(timer)
            logger["stats/nfe"].append((step + 1) * batch_size)

            logger.update(eval_fn(params, key, step))
            print_results(step, logger, cfg)

            if cfg.use_logger:
                log(extract_last_entry(logger), step=step)
                Z_train = running_metrics["Z_train/value"]
                payload = {"Z_train/value": Z_train}
                if target.Z is not None:
                    payload["Z_train/delta"] = jnp.abs(Z_train - target.Z)
                log(payload, step=step)

    return logger
