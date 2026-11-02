# ---------------------------------------------------------------------------------------- #
#                                   PROBABILITY MEASURES                                   #
# ---------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd
from tinygp.kernels import ExpSquared as RBF

from .gp.transforms import ARD
from .gp.sampling import halton_samples
# from .utils import grid, X_to_freq, scale_conv


# ---------------------------------- TOP LEVEL FUNCTIONS --------------------------------- #
def sample(key, m, N, bounds=None, qmc=True):
    mc_dict_key = "qmc" if qmc else "sample"

    if isinstance(m, tfd.Independent):
        sample_func = MEASURES[m.distribution.__class__][mc_dict_key]
    else:
        sample_func = MEASURES[m.__class__][mc_dict_key]
    return sample_func(key, m, N, bounds)


def cf(m, X):
    """Characteristic function (FT) of a measure"""
    m_cf = MEASURES[m.__class__]["cf"]


def truncation_term(m, bounds):
    if isinstance(m, tfd.Independent):
        trunc_fn = MEASURES[m.distribution.__class__]["trunc"]
        trunc = trunc_fn(m.distribution, bounds)
        return jnp.prod(trunc)
    else:
        trunc_fn = MEASURES[m.__class__]["trunc"]
        return trunc_fn(m, bounds)


# ---------------------------------------- UNIFORM --------------------------------------- #
def sample_uniform(key, m, N, bounds=None):
    d = m.event_shape[0]
    samples = jax.random.uniform(key, (N, d))
    if bounds is not None:
        samples = bounds[0, :] + samples * (bounds[1, :] - bounds[0, :])
    return samples


def qmc_uniform(key, m, N, bounds=None):
    d = m.event_shape[0]
    samples = halton_samples(key, N, d)
    if bounds is not None:
        samples = bounds[0, :] + samples * (bounds[1, :] - bounds[0, :])
    return samples


def uniform_trunc(m, bounds):
    if m.batch_shape[0] > 1:
        return jnp.ones(m.batch_shape[0])
    else:
        return 1.


# ----------------------------------- STANDARD GAUSSIAN ---------------------------------- #
def sample_gaussian(key, m, N, bounds=None):
    d = m.event_shape[0]
    samples = jax.random.normal(key, (N, d))
    if bounds is not None:
        lb_cdf, ub_cdf = jax.scipy.stats.norm.cdf(bounds, m.mean(), m.stddev())
        diff = ub_cdf - lb_cdf
        samples = lb_cdf + samples * diff
        samples = jax.scipy.stats.norm.ppf(samples, m.mean(), m.stddev())
    return samples


def qmc_gaussian(key, m, N, bounds=None):
    d = m.event_shape[0]
    samples = halton_samples(key, N, d)
    if bounds is not None:
        lb_cdf, ub_cdf = jax.scipy.stats.norm.cdf(bounds, m.mean(), m.stddev())
        diff = ub_cdf - lb_cdf
        samples = lb_cdf + samples * diff
        samples = jax.scipy.stats.norm.ppf(samples, m.mean(), m.stddev())
    return samples


def cf_gaussian(m, X):
    d = m.event_shape[0]
    nconst = (2 * jnp.pi)**(d/2) * jnp.sqrt(jnp.linalg.det(m.stddev()**2))
    return m.prob(X) * nconst


@jax.jit
def gaussian_trunc(m, bounds):
    mu = m.loc
    sigma = m.scale
    cdf_vals = jax.scipy.stats.norm.cdf(bounds, mu, sigma)
    cdf_diffs = jnp.abs(cdf_vals[1, :] - cdf_vals[0, :])
    return cdf_diffs


# -------------------------------------- MV GAUSSIAN ------------------------------------- #
def diag_mvgaussian_quantile(m, X):
    return jax.scipy.stats.norm.ppf(X, m.mean(), m.stddev())


def sample_mvg(key, m, N, bounds=None):
    d = m.event_shape[0]
    if bounds is None:
        samples = jax.random.uniform(key, (N, d))
        samples = diag_mvgaussian_quantile(m, samples)
    else:
        lb_cdf, ub_cdf = jax.scipy.stats.norm.cdf(bounds, m.mean(), m.stddev())
        diff = ub_cdf - lb_cdf
        samples = samples = jax.random.uniform(key, (N, d))
        samples = lb_cdf + samples * diff
        samples = diag_mvgaussian_quantile(m, samples)
    return samples


def qmc_mvg(key, m, N, bounds=None):
    d = m.event_shape[0]
    if bounds is None:
        samples = halton_samples(key, N, d)
        samples = diag_mvgaussian_quantile(m, samples)
    else:
        lb_cdf, ub_cdf = jax.scipy.stats.norm.cdf(bounds, m.mean(), m.stddev())
        diff = ub_cdf - lb_cdf
        samples = halton_samples(key, N, d)
        samples = lb_cdf + samples * diff
        samples = diag_mvgaussian_quantile(m, samples)
    return samples


@jax.jit
def mvg_trunc(m, bounds):
    mu = m.loc
    sigma = m.scale.diag
    cdf_vals = jax.scipy.stats.norm.cdf(bounds, mu, sigma)
    cdf_diffs = jnp.abs(cdf_vals[1, :] - cdf_vals[0, :])
    return jnp.prod(cdf_diffs)


# dictionary mapping measures to their various sampling methods
MEASURES = {
    tfd.MultivariateNormalDiag: {
        "sample": sample_mvg,
        "qmc": qmc_mvg,
        "trunc": mvg_trunc,
    },
    tfd.Uniform: {
        "sample": sample_uniform,
        "qmc": qmc_uniform,
        "trunc": uniform_trunc,
    },
    tfd.Normal: {
        "sample": sample_gaussian,
        "qmc": qmc_gaussian,
        "trunc": gaussian_trunc,
    },
}


# ----------------------------------- MEASURE TRAINING ----------------------------------- #
def train_measure(m, gp, bq, X, y, method, bounds=None, diag=None, **kwargs):
    pass
    
    # z, Z = kernel_mean(gp, m, X, bounds=bounds, method=method, var=True, **kwargs)
    # zK, zKz = bq.zK_z(X, z, diag=diag, method=method)
    # mu = (y @ zK).squeeze()
    
    # # variance
    # variance = (Z - zKz).squeeze() 
    # if bounds is not None:
    #     variance /= area(bounds)**2

    # return jnp.array([mu, variance]).reshape(-1)


# ----------------------------------------- UTILS ---------------------------------------- #
# def sample_truncated_measure(m, N, bounds, key):
#     d = m.event_shape[0]
#     if isinstance(m, tfd.MultivariateNormalDiag):
#         lb_cdf, ub_cdf = jax.scipy.stats.norm.cdf(bounds, m.mean(), m.stddev())
#         samples = jax.random.uniform(key, (N, d), minval=lb_cdf, maxval=ub_cdf)
#         samples = diag_mvgaussian_quantile(m, samples)
#         return samples

#     else:
#         raise NotImplementedError("QMC sampling only implemented for multivariate normal")

    # # sample points
    # samples = m.sample(N, key)
    # lb, ub = bounds[0, :], bounds[1,:]
    # idx = jnp.where((samples >= lb).all(axis=1) & ((samples <= ub)).all(axis=1))[0]
    # samples = samples[idx, :]
    
    # # loop if not enough samples
    # sample_power = 0
    # while samples.shape[0] < N:
    #     sample_power += 1
    #     # if N * 2**sample_power * len(lb) > 1000000:
    #     #     return samples[0:N, :]
        
    #     samples = m.sample(N * 2**sample_power, key)
    #     idx = jnp.where((samples >= lb).all(axis=1) & ((samples <= ub)).all(axis=1))[0]
    #     samples = samples[idx, :]

    # samples = samples[0:N, :]
    # return samples


# ----------------------------- MEASURE/KERNEL FOURIER DUALS ----------------------------- #
# def measure_to_kernel_grid(m, bounds, sr, pad=True):
#     # find apprpropriate kernel for measure
#     if isinstance(m, tfd.MultivariateNormalDiag):
#         kernel = ARD(m.stddev(), RBF())
#     else:
#         raise NotImplementedError("Only implemented for multivariate normal")
    
#     trunc = diag_mvgaussian_trunc_term(m.mean(), m.stddev(), bounds)

#     # make evaluation grid
#     d = bounds.shape[-1]
#     if pad:
#         Xg = grid(bounds * 2, sr, flatten=False)
#     else:
#         Xg = grid(bounds, sr, flatten=False)
#     Xg_flat = Xg.reshape(-1, d)
#     grid_dims = Xg.shape[:-1]

#     # transform original inputs to frequencies
#     freqs_flat = X_to_freq(Xg_flat, sr, shift=False)

#     # evaluate the kernel at the frequencies
#     Kf = kernel(jnp.zeros(d), freqs_flat).reshape(grid_dims)
#     Kf /= trunc

#     # scale to fft
#     Kf = scale_conv(Xg_flat, Kf, inverse=True, single=True, pad=pad)
#     return Kf


