from functools import partial

import jax
import jax.numpy as jnp

from eval.utils import (
    save_samples,
    compute_statistics_over_runs,
)
from eval.discrepancies import compute_sd
from utils.plot_utils import visualise_trajectories


def compute_estimates_from_weights(log_is_weights, target):
    # fmt: off
    log_Z = jax.scipy.special.logsumexp(log_is_weights) - jnp.log(log_is_weights.shape[0])
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


def get_eval_fn(rnd, target, cfg):
    n_runs = cfg.n_runs
    rnd_reverse = jax.jit(
        partial(rnd, prior_to_target=True),
        static_argnames=("batch_size",),
    )

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

    def _short_eval(model_state, key, log_media=False, iter_step=0):
        if isinstance(model_state, tuple):
            model_state1, model_state2 = model_state
            params = (model_state1.params, model_state2.params)
        else:
            params = (model_state.params,)
        eval_samples = (
            cfg.eval_samples
            if cfg.fix_eval_budget
            else (iter_step + 1) * cfg.algorithm.batch_size
        )
        samples, running_costs, stochastic_costs, terminal_costs, *aux = rnd_reverse(
            key, model_state, *params, batch_size=eval_samples
        )
        trajectories = aux[0] if len(aux) == 1 else None
        if trajectories is not None and log_media:
            logger.update(
                visualise_trajectories(
                    trajectories[:10],
                    target,
                    dims=(0, 1),
                    device=samples.device,
                    prefix="trajectories_fwd",
                )
            )

        log_is_weights = -(running_costs + stochastic_costs + terminal_costs)
        log_Z, Z = compute_estimates_from_weights(log_is_weights, target)
        logger["logZ/value"].append(log_Z)
        if target.log_Z is not None:
            delta_log_Z = jnp.abs(log_Z - target.log_Z)
            logger["logZ/delta"].append(delta_log_Z)
        logger["Z/value"].append(Z)
        if target.Z is not None:
            delta_Z = jnp.abs(Z - target.Z)
            logger["Z/delta"].append(delta_Z)

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

    def short_eval(model_state, key, iter_step=0):
        keys = jax.random.split(key, n_runs)
        logger = _short_eval(model_state, keys[0], log_media=True, iter_step=iter_step)
        for k in keys[1:]:
            logger = _short_eval(model_state, k, iter_step=iter_step)
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
