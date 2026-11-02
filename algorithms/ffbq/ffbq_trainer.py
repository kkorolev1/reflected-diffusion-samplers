import sys
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from tensorflow_probability.substrates.jax import distributions as tfd

from eval.utils import extract_last_entry
from utils.logger import log
from utils.print_utils import print_results
from algorithms.common.eval_methods.tractable_density_methods import get_eval_fn

_FFBQ_ROOT = Path(__file__).resolve().parent
if str(_FFBQ_ROOT) not in sys.path:
    sys.path.insert(0, str(_FFBQ_ROOT))

from ffbq.conv import FFTConv, GaussianConv, MCConv, QMCConv, SVGPConv  # noqa: E402
from ffbq.models import (  # noqa: E402
    build_train_gp,
    build_train_lrgp,
    build_train_lrrbf,
    build_train_rff_gp,
    build_train_svgp,
)
from ffbq.sparse_conv import SparseFFTConv, SparseMCConv  # noqa: E402


def cfg_to_dict(cfg):
    return OmegaConf.to_container(cfg, resolve=True) if cfg is not None else {}


def sample_uniform(key, n, bounds):
    lb, ub = bounds
    return jax.random.uniform(key, (n, bounds.shape[1]), minval=lb, maxval=ub)


def target_values(target, x):
    log_abs_y = target.log_prob(x)
    if hasattr(target, "sign"):
        return target.sign(x) * jnp.exp(log_abs_y)
    return jnp.exp(log_abs_y)


def build_measure(name, bounds, dim, alg_cfg):
    if name == "uniform":
        base = tfd.Uniform(low=bounds[0], high=bounds[1])
        return tfd.Independent(base, reinterpreted_batch_ndims=1)

    if name in ("mvn", "gaussian"):
        mean = alg_cfg.get("measure_mean", None)
        if mean is None:
            mean = (bounds[0] + bounds[1]) / 2
        scale = alg_cfg.get("measure_scale", None)
        if scale is None:
            scale = jnp.ones(dim)
        mean = jnp.asarray(mean, dtype=jnp.float32)
        scale = jnp.asarray(scale, dtype=jnp.float32)
        return tfd.MultivariateNormalDiag(loc=mean, scale_diag=scale)

    raise ValueError(f"Unknown FFBQ measure '{name}'.")


def build_gp(key, x_train, y_train, alg_cfg):
    model = alg_cfg.model
    params = cfg_to_dict(alg_cfg.get(model, None))
    params.setdefault("diag", alg_cfg.diag)
    params.setdefault("epochs", alg_cfg.epochs)
    params.setdefault("lr", alg_cfg.lr)
    params.setdefault("init_ls", alg_cfg.init_ls)

    if model == "gp":
        params.setdefault("kernel", alg_cfg.kernel)
        return build_train_gp(key, x_train, y_train, **params)
    if model == "rff":
        params.setdefault("R", alg_cfg.num_features)
        params.setdefault(
            "kernel", alg_cfg.kernel if alg_cfg.kernel != "rbf" else "rff"
        )
        return build_train_rff_gp(key, x_train, y_train, **params)
    if model == "lrgp":
        params.setdefault("R", alg_cfg.num_features)
        return build_train_lrgp(key, x_train, y_train, **params)
    if model == "lrrbf":
        params.setdefault("R", alg_cfg.num_features)
        return build_train_lrrbf(key, x_train, y_train, **params)
    if model == "svgp":
        params.setdefault("R", alg_cfg.num_features)
        params.setdefault("kernel", alg_cfg.kernel)
        return build_train_svgp(key, x_train, y_train, **params)

    raise ValueError(f"Unknown FFBQ model '{model}'.")


def update_gp_inputs(gp, x_eval):
    return eqx.tree_at(lambda t: t.X, gp, x_eval)


def run_bq(key, gp, y_eval, measure, bounds, alg_cfg):
    operator = alg_cfg.operator
    kwargs = cfg_to_dict(alg_cfg.get("operator_kwargs", None))
    diag = alg_cfg.get("bq_diag", alg_cfg.diag)

    if operator == "gaussian":
        conv = GaussianConv(measure, bounds=bounds)
        kwargs.setdefault("R", alg_cfg.num_conv_samples)
        kwargs.setdefault("key", key)
    elif operator == "mc":
        conv = MCConv(measure, bounds=bounds)
        kwargs.setdefault("R", alg_cfg.num_conv_samples)
        kwargs.setdefault("key", key)
    elif operator == "qmc":
        conv = QMCConv(measure, bounds=bounds)
        kwargs.setdefault("R", alg_cfg.num_conv_samples)
        kwargs.setdefault("key", key)
    elif operator == "fft":
        conv = FFTConv(measure, bounds=bounds)
        kwargs.setdefault("sr", alg_cfg.sr)
    elif operator == "svgp":
        conv = SVGPConv(measure, bounds=bounds)
        kwargs.setdefault("R", alg_cfg.num_conv_samples)
        kwargs.setdefault("key", key)
    elif operator in ("sparse_mc", "sparse_qmc"):
        conv = SparseMCConv(measure, bounds=bounds)
        kwargs.setdefault("R", alg_cfg.num_conv_samples)
        kwargs.setdefault("key", key)
        kwargs.setdefault("qmc", operator != "sparse_mc")
    elif operator == "sparse_fft":
        conv = SparseFFTConv(measure, bounds=bounds)
        kwargs.setdefault("sr", alg_cfg.sr)
    else:
        raise ValueError(f"Unknown FFBQ operator '{operator}'.")

    try:
        return conv(gp, y_eval, diag=diag, **kwargs)
    except TypeError:
        return conv(gp, y_eval, **kwargs)


def train_gp(key, cfg, target, bounds):
    alg_cfg = cfg.algorithm
    train_key, eval_key, gp_key = jax.random.split(key, 3)
    x_train = sample_uniform(train_key, alg_cfg.n_train, bounds)
    y_train = target_values(target, x_train)
    gp, gp_loss = build_gp(gp_key, x_train, y_train, alg_cfg)
    if alg_cfg.use_eval_split:
        x_eval = sample_uniform(eval_key, alg_cfg.n_eval, bounds)
        y_eval = target_values(target, x_eval)
        gp = update_gp_inputs(gp, x_eval)
    else:
        y_eval = y_train
    return gp, y_eval


def rnd_ffbq(key, cfg, gp, y_eval, measure, bounds, num_samples):
    alg_cfg = cfg.algorithm
    alg_cfg.num_conv_samples = num_samples
    z_estimate, bq_var = run_bq(key, gp, y_eval, measure, bounds, alg_cfg)
    return jnp.log(z_estimate), z_estimate


def ffbq_trainer(cfg, target):
    key_gen = jax.random.PRNGKey(cfg.seed)
    dim = target.dim
    alg_cfg = cfg.algorithm

    print(f"True log_Z: {target.log_Z}")
    print(f"True Z: {target.Z}")
    print(f"Needs correction: {target.needs_correction}, corrector: {target.corrector}")

    target.visualise = partial(target.visualise, constrained=True, map_samples=False)
    target.sample = partial(target.sample, constrained=True)
    target.log_prob = partial(target.log_prob, constrained=True)

    bounds = jnp.stack([jnp.zeros((dim,)), jnp.ones((dim,))])
    measure = build_measure(alg_cfg.measure, bounds, dim, alg_cfg)

    key, key_gen = jax.random.split(key_gen)
    gp, y_eval = train_gp(key, cfg, target, bounds)

    eval_fn, logger = get_eval_fn(
        partial(
            rnd_ffbq, cfg=cfg, gp=gp, y_eval=y_eval, measure=measure, bounds=bounds
        ),
        cfg,
        target,
    )

    for step in range(7, alg_cfg.iters):
        key, key_gen = jax.random.split(key_gen)
        logger["stats/step"].append(step)
        logger["stats/nfe"].append(2**step)

        logger.update(eval_fn(key, step))

        print_results(step, logger, cfg)

        if cfg.use_logger:
            log(extract_last_entry(logger), step=step)

    return logger
