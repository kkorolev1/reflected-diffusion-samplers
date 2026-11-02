# ---------------------------------------------------------------------------------------- #
#                                   CONVOLUTION OPERATORS                                  #
# ---------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
from jax import vmap, jit
from tensorflow_probability.substrates.jax import distributions as tfd
import scipy
from jax.scipy.sparse.linalg import cg
import equinox as eqx
from tinygp.helpers import JAXArray
from jaxtyping import Float, Bool
from scipy.interpolate import interpn

from ffbq.measures import truncation_term, mvg_trunc, sample
from ffbq.utils import area, grid, smart_conv


__all__ = [
    "GaussianConv", 
    "MCConv", 
    "QMCConv",
    "FFTConv",
    "SVGPConv"
]


# ------------------------------------ TOP LEVEL CONV ------------------------------------ #
class KernelMean(eqx.Module):
    m: tfd.Distribution
    bounds: JAXArray | None
    bounded: Bool
    trunc: Float

    def __init__(self, m, bounds=None,  **kwargs):
        self.m = m
        self.bounds = bounds

        if bounds is not None:
            self.trunc = truncation_term(m, bounds)
            self.bounded = True
        else:
            self.trunc = 1.
            self.bounded = False

    def kmu_bounded(self, gp, **kwargs):  # override in subclass
        raise NotImplementedError
    
    def kmu_unbounded(self, gp, **kwargs):  # override in subclass
        raise NotImplementedError

    def kmu(self, gp, **kwargs):
        if self.bounded:
            return self.kmu_bounded(gp, **kwargs)
        else:
            return self.kmu_unbounded(gp, **kwargs)

    def zK_z(self, gp, diag=None, **kwargs):
        X = gp.X
        N_x, d = X.shape
        
        # calulate kernel mean
        z, Z = self.kmu(gp, **kwargs)

        # calculate kernel
        K = gp.kernel(X, X)
        if diag is None:
            # diag = gp._diag
            diag = jnp.trace(K) * 0.001
        K = K + jnp.eye(N_x) * diag
        K = jnp.clip(K, 0, None)

        # solve through conjugate gradients
        zK = cg(K, z)[0]
        zKz = z @ zK

        return z, Z, zK, zKz

    def bq(self, gp, y, **kwargs):
        _, Z, zK, zKz = self.zK_z(gp, **kwargs)
        mu = (y @ zK).squeeze()
        
        # variance
        variance = (Z - zKz).squeeze()
        if self.bounded:
            variance /= area(self.bounds)**2
        return (mu, variance)

    def __call__(self, gp, y, **kwargs):
        return self.bq(gp, y, **kwargs)


# ---------------------------------- GAUSSIAN ANALYTICAL --------------------------------- #
class GaussianConv(KernelMean):
    def kmu_bounded(self, gp, key, R=10000):
        # extact parameters
        m = self.m
        k = gp.kernel
        scale = k.transform._scale
        bounds = self.bounds
        X = gp.X

        d = m.event_shape[0]
        Σ_ls = jnp.diag(scale**2)
        Σ_x = jnp.diag(m.stddev()**2)

        # posterior term of multiple of two gaussians
        C = jnp.linalg.inv(jnp.linalg.inv(Σ_ls) + jnp.linalg.inv(Σ_x))
        A_inv_a = (jnp.linalg.inv(Σ_ls) @ X.T).T
        B_inv_b = jnp.linalg.inv(Σ_x) @ m.mean()
        AB = A_inv_a + B_inv_b
        c = (C @ AB.T).T
        
        # normalizing constant
        normalizer = tfd.MultivariateNormalDiag(
            m.mean(), jnp.sqrt(jnp.diag(Σ_ls + Σ_x))
        )
        norm_vals = normalizer.prob(X)

        # truncation terms
        p_C = tfd.MultivariateNormalDiag(loc=c, scale_diag=jnp.sqrt(jnp.diag(C)))
        posterior_truncs = jax.vmap(
            lambda i: mvg_trunc(p_C[i], bounds)
        )(jnp.arange(p_C.batch_shape[0]))
        p_x_trunc_term = self.trunc

        # calculate output
        z = norm_vals * posterior_truncs
        z /= p_x_trunc_term
        z *= area(bounds)
        # if normalize_k:
        z *= (2 * jnp.pi)**(d/2) * jnp.sqrt(jnp.linalg.det(Σ_ls))
        
        #### variance calculation
        x_samples = sample(key, m, R, bounds, qmc=True)
        Z = k(x_samples, x_samples).mean().squeeze() * area(bounds)**2
        return z, Z

    @jax.jit
    def kmu_unbounded(self, gp):
        m = self.m
        k = gp.kernel
        scale = k.transform._scale
        X = gp.X

        # # helpers
        d = m.event_shape[0]
        Σ_ls = jnp.diag(scale**2)
        Σ_x = jnp.diag(m.stddev()**2)

        # # normalizing constant
        normalizer = tfd.MultivariateNormalDiag(
            m.mean(), jnp.sqrt(jnp.diag(Σ_ls + Σ_x))
        )
        norm_vals = normalizer.prob(X)

        # # calculate output
        z = norm_vals * ((2 * jnp.pi)**(d/2) * jnp.sqrt(jnp.linalg.det(Σ_ls)))

        # # variance calculation
        Z = jnp.sqrt(scale**2 / (scale**2 + 2 * m.stddev()**2)).prod()

        return z, Z


# -------------------------------------- MONTE CARLO ------------------------------------- #
class MCConv(KernelMean):
    def kmu_bounded(self, gp, key, R):
        m = self.m
        k = gp.kernel
        bounds = self.bounds
        X = gp.X

        # sample points
        samples = sample(key, m, R, bounds, qmc=False)
        var_samples = sample(key, m, min(R, 10000), bounds, qmc=False)

        # run kernel matrix for mean and variance
        ab = area(bounds)
        f_z = jit(lambda x1, x2: k(x1, x2).mean(axis=1))
        z = f_z(X, samples) * ab
        Z = f_z(samples, var_samples).mean().squeeze() * area(bounds)**2

        return z, Z

    def kmu_unbounded(self, gp, key, R):
        m = self.m
        k = gp.kernel
        X = gp.X

        samples = sample(key, m, R, qmc=False)
        var_samples = sample(key, m, min(R, 10000), qmc=False)

        f_z = jit(lambda x1, x2: k(x1, x2).mean(axis=1))
        z = f_z(X, samples)
        Z = f_z(samples, var_samples).mean().squeeze()

        return z, Z


# ---------------------------------------- QMC ------------------------------------------- #
class QMCConv(KernelMean):
    def kmu_bounded(self, gp, key, R):
        m = self.m
        k = gp.kernel
        bounds = self.bounds
        X = gp.X

        # sample points
        samples = sample(key, m, R, bounds, qmc=True)
        var_samples = sample(key, m, min(R, 10000), bounds, qmc=True)

        # run kernel matrix for mean and variance
        ab = area(bounds)
        f_z = jit(lambda x1, x2: k(x1, x2).mean(axis=1))
        z = f_z(X, samples) * ab
        Z = f_z(samples, var_samples).mean().squeeze() * area(bounds)**2

        return z, Z

    def kmu_unbounded(self, gp, key, R):
        m = self.m
        k = gp.kernel
        X = gp.X

        samples = sample(key, m, R, qmc=True)
        var_samples = sample(key, m, min(R, 10000), qmc=True)

        f_z = jit(lambda x1, x2: k(x1, x2).mean(axis=1))
        z = f_z(X, samples)
        Z = f_z(samples, var_samples).mean().squeeze()

        return z, Z


# # ------------------------------------------ FFT ----------------------------------------- #
class FFTConv(KernelMean):
    def kmu_bounded(self, gp, sr, bounds=None, pad=True, center=True, pow2=True, energy_bound=0.5):
        m = self.m
        k = gp.kernel
        bounds = self.bounds if bounds is None else bounds
        X = gp.X
        d = X.shape[-1]

        # make evaluation grid
        t, t_axes = grid(bounds, sr=sr, center=center, flatten=False, axes=True)
        t_flat = t.reshape(-1, d)
        grid_dims = t.shape[:-1]
        grid_size = jnp.prod(jnp.asarray(grid_dims))

        # eval kernel and measure
        k_mid = jnp.zeros((1, d)) + jnp.median(bounds, axis=0)  # center kernel
        kt = k(k_mid, t_flat).reshape(grid_dims)
        pt = jnp.apply_along_axis(m.prob, -1, t)
        if self.bounded:
            pt = pt / self.trunc

        #### MEAN
        # convolve
        zG = smart_conv(kt, pt, pad=pad, pow2=pow2, energy_bound=energy_bound)  # will come out unpadded and shifted
        zG_scaled = zG.real * area(bounds) / grid_size

        # interpolate
        z = jnp.asarray(interpn(
            t_axes, zG_scaled, X, 
            method="cubic", bounds_error=False, fill_value=None
        )).reshape(-1)

        #### VARIANCE
        ZG = smart_conv(zG, pt, pad=pad, pow2=pow2, energy_bound=energy_bound)
        ZG_scaled = ZG.real * (area(bounds) / grid_size)**2
        if self.bounded:
            ZG_scaled *= area(bounds)**2

        # interpolate
        Z = jnp.asarray(interpn(
            t_axes, ZG_scaled, jnp.zeros((1, d)), 
            method="cubic", bounds_error=False, fill_value=None
        )).reshape(-1).squeeze()

        return z, Z

    def kmu_unbounded(self, gp, sr, bounds, pad=True, center=True, pow2=True, energy_bound=0.5):
        return self.kmu_bounded(gp, sr, bounds, pad, center)


# --------------------------------------- SVGP CONV -------------------------------------- #
class SVGPConv(KernelMean):
    def kmu_bounded(self, gp, key, R):
        m = self.m
        k = gp.kernel
        bounds = self.bounds
        X_m = gp.m

        # sample points
        samples = sample(key, m, R, bounds, qmc=True)
        var_samples = sample(key, m, min(R, 10000), bounds, qmc=True)

        # run kernel matrix for mean and variance
        ab = area(bounds)
        f_z = jit(lambda x1, x2: k(x1, x2).mean(axis=1))
        z = f_z(X_m, samples) * ab
        Z = f_z(samples, var_samples).mean().squeeze() * area(bounds)**2

        return z, Z

    def kmu_unbounded(self, gp, key, R):
        m = self.m
        k = gp.kernel
        X = gp.X
        X_m = gp.m

        samples = sample(key, m, R, qmc=True)
        var_samples = sample(key, m, min(R, 10000), qmc=True)

        f_z = jit(lambda x1, x2: k(x1, x2).mean(axis=1))
        z = f_z(X_m, samples)
        Z = f_z(samples, var_samples).mean().squeeze()

        return z, Z
    
    def zK_z(self, gp, diag=None, **kwargs):
        X = gp.X
        X_m = gp.m
        R = X_m.shape[0]
        
        # calulate kernel mean
        z, Z = self.kmu(gp, **kwargs)

        # calculate kernel
        K = gp.kernel(X_m, X_m)
        if diag is None:
            diag = jnp.trace(K) * 0.0001
            if jnp.isnan(diag) or diag < 0:
                diag = 1e-3
        K = K + jnp.eye(R) * diag
        K = jnp.clip(K, 0, None)

        # solve through conjugate gradients
        zK = cg(K, z)[0]
        zKz = z @ zK

        return z, Z, zK, zKz

    def bq(self, gp, y, **kwargs):
        _, Z, zK, zKz = self.zK_z(gp, **kwargs)
        y_m = gp.phi_opt(y)[0]
        mu = (y_m @ zK).squeeze()
        
        # variance
        variance = (Z - zKz).squeeze()
        if self.bounded:
            variance /= area(self.bounds)**2
        return (mu, variance)

    def __call__(self, gp, y, **kwargs):
        return self.bq(gp, y, **kwargs)
