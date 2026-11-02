"""
Code for Denoising Diffusion Sampler (DDS)
For further details see https://arxiv.org/abs/2302.13834
Code based on https://github.com/franciscovargas/denoising_diffusion_samplers
"""

from functools import partial
from time import time

import distrax
import jax
import jax.numpy as jnp
import math

# import wandb  # Replaced by unified logger

from algorithms.common.diffusion_related.init_model import init_model_rds
from algorithms.common.eval_methods.stochastic_oc_methods import (
    get_eval_fn,
    compute_estimates_from_weights,
)
from algorithms.rds.rds_rnd import kl_loss_reparam_fn, kl_loss_reinforce_fn, rnd
from eval.utils import extract_last_entry, update_train_estimate
from utils.logger import log
from utils.print_utils import print_results
from utils.helper import save_model_checkpoint

import pickle
from flax import serialization


def load_trained_model(cfg, target, ckpt_path):
    key = jax.random.PRNGKey(cfg.seed)
    dim = target.dim

    # Recreate the model state with the same architecture/config
    model_state = init_model_rds(key, dim, cfg.algorithm)

    with open(ckpt_path, "rb") as f:
        payload = pickle.load(f)

    params = serialization.from_state_dict(model_state.params, payload["params"])
    model_state = model_state.replace(params=params)

    print(f"Loaded checkpoint from iter {payload.get('iter', 'unknown')}")

    return model_state


def rds_trainer(cfg, target):
    assert target.is_non_negative, "rds cannot be run for signed targets, use rds_cond"
    key_gen = jax.random.PRNGKey(cfg.seed)
    dim = target.dim
    alg_cfg = cfg.algorithm
    batch_size = alg_cfg.batch_size

    target.visualise = partial(target.visualise, constrained=True, map_samples=False)
    target.sample = partial(target.sample, constrained=True)
    target.log_prob = partial(target.log_prob, constrained=True)

    print(f"True log_Z: {target.log_Z}")
    print(f"True Z: {target.Z}")
    print(f"Needs correction: {target.needs_correction}, corrector: {target.corrector}")

    # Define initial and target density
    # fmt: off
    initial_dist = distrax.Independent(
        distrax.Uniform(low=target.min_coord * jnp.ones(dim), high=target.max_coord * jnp.ones(dim)), 1
    )
    aux_tuple = (
        alg_cfg.noise_schedule,
    )

    # Initialize the model
    key, key_gen = jax.random.split(key_gen)
    model_state = init_model_rds(key, dim, alg_cfg)

    if cfg.algorithm.loss == "kl_reparam":
        loss_fn = kl_loss_reparam_fn
        stop_grad = False
    elif cfg.algorithm.loss == "kl_reinforce":
        loss_fn = kl_loss_reinforce_fn
        stop_grad = True
    else:
        raise ValueError(f"Unknown loss {cfg.algorithm.loss}")

    rnd_partial_base = partial(
        rnd,
        aux_tuple=aux_tuple,
        initial_dist=initial_dist,
        target=target,
        num_steps=cfg.algorithm.num_steps,
        use_lp=alg_cfg.model.use_lp,
        stop_grad=stop_grad,
    )

    # Define the function to be JIT-ed for FWD pass
    @partial(jax.jit)
    @partial(jax.grad, argnums=2, has_aux=True)
    def loss_fwd_grad_fn(key, model_state, params):
        rnd_p = partial(
            rnd_partial_base,
            batch_size=batch_size,
            prior_to_target=True,
            accumulate_costs=False,
        )
        return loss_fn(key, model_state, params, rnd_p)

    eval_fn, logger = get_eval_fn(
        partial(rnd_partial_base, accumulate_costs=True), target, cfg
    )

    eval_freq = max(alg_cfg.iters // cfg.n_evals, 1)
    timer = 0
    running_metrics = {"Z_train/value": None}
    
    for step in range(alg_cfg.iters):
        iter_time = time()
        key, key_gen = jax.random.split(key_gen)
        grads, (losses, log_is_weights) = loss_fwd_grad_fn(
            key,
            model_state,
            model_state.params,
        )
        timer += time() - iter_time
        model_state = model_state.apply_gradients(grads=grads)
        
        _, Z = compute_estimates_from_weights(log_is_weights, target)
        running_metrics["Z_train/value"] = update_train_estimate(Z, running_metrics["Z_train/value"], batch_size, step * batch_size)

        if (step % eval_freq == 0) or (step == alg_cfg.iters - 1):
            log({"loss": jnp.mean(losses)}, step=step)

            key, key_gen = jax.random.split(key_gen)
            logger["stats/step"].append(step)
            logger["stats/wallclock"].append(timer)
            logger["stats/nfe"].append((step + 1) * batch_size)

            logger.update(eval_fn(model_state, key, step))

            print_results(step, logger, cfg)
            save_model_checkpoint(cfg, model_state, iter=step)

            if cfg.use_logger:
                log(extract_last_entry(logger), step=step)
                # fmt: off
                log({"Z_train/value": running_metrics["Z_train/value"], 
                    "Z_train/delta": jnp.abs(running_metrics["Z_train/value"] - target.Z)}, step=step)
