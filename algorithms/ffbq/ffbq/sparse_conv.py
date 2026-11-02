# ---------------------------------------------------------------------------------------- #
#                                 RFF CONVOLUTION OPERATORS                                #
# ---------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
from jax import vmap, jit
from tensorflow_probability.substrates.jax import distributions as tfd
from .utils import grid, fftn_sep, fftfreqn, sr_to_N, area
from scipy.interpolate import interpn
from jax.scipy.sparse.linalg import cg
import equinox as eqx
from tinygp.helpers import JAXArray
from jaxtyping import Float, Bool, Array

from ffbq.measures import truncation_term, mvg_trunc, sample
from ffbq.conv import KernelMean
from ffbq.utils import unpad, fftpad

__all__ = ["SparseMCConv", "SparseFFTConv"]


# ------------------------------------ TOP LEVEL CONV ------------------------------------ #
class SparseKernelMean(KernelMean):
    def mu_phi(self, gp, **kwargs):
        if self.bounded:
            return self.mu_phi_bounded(gp, **kwargs)
        else:
            return self.mu_phi_unbounded(gp, **kwargs)
    
    def mu_phi_bounded(self, gp, **kwargs):
        raise NotImplementedError

    def mu_phi_unbounded(self, gp, **kwargs):
        raise NotImplementedError

    def kmu(self, gp, **kwargs):  # low rank approximation to full rank kernel mean
        phiW = self.mu_phi(gp, **kwargs)
        phiX = gp.kernel.phi(gp.X)
        z = phiX @ phiW.T
        Z = phiW.T @ phiW
        return z, Z

    def __call__(self, gp, y, diag=None, sparse=True, **kwargs):
        if not sparse:
            return self.bq(gp, y, **kwargs)

        # basis function means
        R = gp.kernel.kernel.w.shape[0]
        phiW = self.mu_phi(gp, **kwargs)
        phiX = gp.kernel.phi(gp.X)

        # calculate low-rank integral mean and variance
        A = phiX.T @ phiX
        if diag is None:
            diag = jnp.trace(A) * 0.0001
            if jnp.isnan(diag) or diag < 0:
                diag = 1e-3
        A = A + jnp.eye(2 * R) * diag
        Akmu = cg(A, phiW)[0]
        # Akmu = jnp.linalg.inv(A) @ phiW.T

        # mean
        mu = (y @ phiX @ Akmu).squeeze()

        # variance
        variance = (phiW @ Akmu).squeeze()
        if self.bounded:
            variance /= area(self.bounds)**2

        return (mu, variance)


# ------------------------------------------ MC ------------------------------------------ #
class SparseMCConv(SparseKernelMean):
    def mu_phi_bounded(self, gp, key, R=None, sr=None, qmc=True):
        m = self.m
        k = gp.kernel
        bounds = self.bounds
        
        # get sample count
        if sr is not None:
            R = jnp.prod(jnp.asarray(sr_to_N(sr, self.bounds)))
        elif R is None:
            raise ValueError("Either `sr` or `R` must be provided.")
        
        # sample points
        samples = sample(key, m, R, bounds, qmc=qmc)
        phiW = k.phi(samples).mean(axis=0)
        phiW *= area(bounds)

        return phiW

    def mu_phi_unbounded(self, gp, key, R, qmc=True):
        m = self.m
        k = gp.kernel
           
        # sample points
        samples = sample(key, m, R, bounds=None, qmc=qmc)
        phiW = k.phi(samples).mean(axis=0)

        return phiW

 
# ------------------------------------------ FFT ----------------------------------------- #
class SparseFFTConv(SparseKernelMean):
    trunc: Float[Array, "d"]

    def mu_phi_bounded(self, gp, sr, bounds=None, pad=True, center=True):
        # extract vars
        m = self.m
        k = gp.kernel
        bounds = self.bounds if bounds is None else bounds
        w = gp.kernel.kernel.w
        ls = gp.kernel.transform._scale
        w_ls = w / ls
        
        if isinstance(m, tfd.Uniform):
            midpts = jnp.median(bounds, axis=0)
            w_mod_cos = jnp.dot(w / (jnp.pi / 2), midpts)
            mu_w_cos = jnp.sinc(w_mod_cos) * area(bounds)

            mult = jnp.pi / 2
            w_mod_sin = jnp.dot(w * mult, midpts - jnp.pi / 2)
            mu_w_sin = jnp.sinc(w_mod_sin) * area(bounds)

        elif isinstance(m, tfd.Normal):
            # make data
            t, axes = grid(bounds, sr=sr, center=center, flatten=False, axes=True)

            # apply the measure to the grid
            pts = m.prob(axes.T).T
            if self.bounded:
                pts = (pts / jnp.atleast_2d(self.trunc))
            
            # apply the fft (and padding)
            delta = axes[:, 1] - axes[:, 0]
            if pad:
                pts = jax.vmap(lambda pt: fftpad(pt, center=False))(pts)
                pad_shape = (len(pt) for pt in pts)
                _, freqs = fftfreqn(pad_shape, delta, axes=True)
            else:
                _, freqs = fftfreqn(t.shape[:-1], delta, axes=True)
            Fpt = jnp.fft.fftshift(fftn_sep(pts))
            # Fpt = jnp.abs(Fpt.real)
            Fpt_real = Fpt.real * area(bounds)**2 / jnp.prod(jnp.asarray(t.shape[:-1]))
            Fpt_imag = jnp.conj(Fpt).imag * area(bounds)**2 / jnp.prod(jnp.asarray(t.shape[:-1]))

            # interpolate the omega values
            # adjust frequencies for lengthscale and 2pi
            freqs = [freq * 2 * jnp.pi for i, freq in enumerate(freqs)]
            mu_w_cos = interpn(
                freqs, Fpt_real, w_ls,
                # freqs, Fpt_real, w, 
                method="cubic", bounds_error=False, fill_value=None
            )
            mu_w_cos = jnp.asarray(mu_w_cos)

            mu_w_sin = interpn(
                freqs, Fpt_imag, w_ls,
                method="cubic", bounds_error=False, fill_value=None
            )
            mu_w_sin = jnp.asarray(mu_w_sin)
        
        # modulate with cos and sin waves to account for phase (mean) of measure
        # this assumes the measure is centered
        # midpts = jnp.median(bounds, axis=0)
        # mu_w_cos = mu_w * jnp.cos(jnp.dot(w, midpts))
        # mu_w_sin = mu_w * jnp.sin(jnp.dot(w, midpts))

        # concatenate and reshape features to make final phi mean
        mu_phi = jnp.concatenate([mu_w_cos, mu_w_sin]) / jnp.sqrt(w.shape[0])
        return mu_phi, mu_w_sin

    def mu_phi_unbounded(self, gp, sr, bounds, center=True):
        # bounds should be set sufficiently large to cover the entire region of
        # probability mass
        return self.mu_phi_bounded(gp, sr, bounds, center)

