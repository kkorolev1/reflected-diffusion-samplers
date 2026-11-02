import jax
import jax.numpy as jnp

from eval.utils import compute_statistics_over_runs


def get_eval_fn(rnd, cfg, target):
    n_runs = cfg.n_runs

    logger = {
        "logZ/value": [],
        "logZ/delta": [],
        "Z/value": [],
        "Z/delta": [],
        "stats/step": [],
        "stats/wallclock": [],
        "stats/nfe": [],
    }

    for metric_name in ["logZ/value", "logZ/delta", "Z/value", "Z/delta", "sd/value"]:
        logger[f"{metric_name}/mean"] = []
        logger[f"{metric_name}/std"] = []

    def _short_eval(key, iter_step=0):
        log_Z, Z = rnd(key, num_samples=2**iter_step)
        if target.is_non_negative:
            logger["logZ/value"].append(log_Z)
            if target.log_Z is not None:
                delta_log_Z = jnp.abs(log_Z - target.log_Z)
                logger["logZ/delta"].append(delta_log_Z)
        logger["Z/value"].append(Z)
        if target.Z is not None:
            delta_Z = jnp.abs(Z - target.Z)
            logger["Z/delta"].append(delta_Z)
        return logger

    def short_eval(key, iter_step=0):
        keys = jax.random.split(key, n_runs)
        logger = _short_eval(keys[0], iter_step=iter_step)
        for k in keys[1:]:
            logger = _short_eval(k, iter_step=iter_step)
        compute_statistics_over_runs(logger, "logZ/value", n_runs)
        if target.log_Z is not None:
            compute_statistics_over_runs(logger, "logZ/delta", n_runs)
        compute_statistics_over_runs(logger, "Z/value", n_runs)
        if target.Z is not None:
            compute_statistics_over_runs(logger, "Z/delta", n_runs)
        return logger

    return short_eval, logger
