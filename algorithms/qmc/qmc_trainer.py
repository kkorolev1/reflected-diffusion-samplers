import jax
import numpy as np
import jax.numpy as jnp
import math
from functools import partial
from time import time
from scipy.stats import qmc
from eval.utils import extract_last_entry
from utils.logger import log
from utils.print_utils import print_results
from algorithms.common.eval_methods.tractable_density_methods import (
    get_eval_fn,
)


def compute_estimates_from_weights(samples, log_is_weights, target):
    # if not target.is_non_negative:
    #     signs = target.sign(samples)
    # else:
    #     signs = jnp.ones_like(log_is_weights)
    # log_abs_sum, sign_Z = jax.scipy.special.logsumexp(
    #     log_is_weights,
    #     b=signs,
    #     return_sign=True,
    # )
    # log_Z = log_abs_sum - jnp.log(log_is_weights.shape[0])
    # Z = sign_Z * jnp.exp(log_Z)
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


def rnd_qmc(key, target, num_samples):
    # fmt: off
    np_seed = int(jax.random.randint(key, (), minval=0, maxval=1_000_000, dtype=jnp.uint32))
    sampler = qmc.Sobol(d=target.dim, scramble=True, seed=np_seed)
    samples = sampler.random_base2(m=int(math.ceil(math.log2(num_samples))))
    # currently we test qmc only in a unit cube, so we don't account for volume of the domain in log_is_weights
    log_is_weights = target.log_prob(samples)
    if target.needs_correction:
        log_is_weights = jnp.logaddexp(log_is_weights, jnp.log(target.corrector))
    return compute_estimates_from_weights(samples, log_is_weights, target)


def qmc_trainer(cfg, target):
    assert target.min_coord == 0.0 and target.max_coord == 1.0
    key_gen = jax.random.PRNGKey(cfg.seed)
    alg_cfg = cfg.algorithm

    target.visualise = partial(target.visualise, constrained=True, map_samples=False)
    target.sample = partial(target.sample, constrained=True)
    target.log_prob = partial(target.log_prob, constrained=True)

    print(f"True log_Z: {target.log_Z}")
    print(f"True Z: {target.Z}")
    print(f"Needs correction: {target.needs_correction}, corrector: {target.corrector}")

    eval_fn, logger = get_eval_fn(partial(rnd_qmc, target=target), cfg, target)

    for step in range(alg_cfg.iters):
        key, key_gen = jax.random.split(key_gen)
        logger["stats/step"].append(step)
        logger["stats/nfe"].append(2**step)

        logger.update(eval_fn(key, step))

        print_results(step, logger, cfg)

        if cfg.use_logger:
            log(extract_last_entry(logger), step=step)
