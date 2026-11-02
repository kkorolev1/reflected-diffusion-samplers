# ---------------------------------------------------------------------------------------- #
#                                EQUINOX GP IMPLEMENTATIONS                                #
# ---------------------------------------------------------------------------------------- #

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import equinox as eqx
from jaxtyping import Array, Float, Bool
from typing import Callable
from .kernels import RFF
from jax import jit, vmap
from functools import partial
from jax.lax import cond
import jax.tree_util as jtu
from tensorflow_probability.substrates.jax import distributions as tfd
from tinygp.helpers import JAXArray

# from steinRF.utils import stabilize



# --------------------------------------- BASIC GP --------------------------------------- #
class GP(eqx.Module):
    mean: eqx.Module
    kernel: eqx.Module
    X: Float[Array, "N d"]
    diag: Float
    
    def __init__(self, kernel, X, diag=None, mean=None):
        
        self.X = X
        self.kernel = kernel

        if diag is None:
            diag = jnp.float32(1e-5)
        self.diag = jnp.log(diag)

        if mean is None:
            mean = ZeroMean()
        self.mean = mean

    @property
    def _diag(self):
        return jnp.exp(self.diag)

    @jit
    def nll(self, y):
        K = self.kernel(self.X, self.X) + jnp.eye(self.X.shape[0]) * self._diag
        n = y.shape[0]
        L = jnp.linalg.cholesky(K)
        y_diff = y - self.mean(self.X)

        alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y_diff))
        term1 = 0.5 * jnp.dot(y_diff.T, alpha)
        term2 = jnp.sum(jnp.log(jnp.diag(L)))
        term3 = 0.5 * n * jnp.log(2 * jnp.pi)
        return term1 + term2 + term3

    @jit
    def condition(self, y, X_test):
        K = self.kernel(self.X, self.X) + jnp.eye(self.X.shape[0]) * self._diag
        K_star = self.kernel(self.X, X_test)
        K_star_star = self.kernel(X_test, X_test) # + jnp.eye(X_test.shape[0]) * self._diag
        
        L = jnp.linalg.cholesky(K)
        y_diff = y - self.mean(self.X)
        alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y_diff))
        
        mu_pred = jnp.dot(K_star.T, alpha) + self.mean(X_test)
        v = jnp.linalg.solve(L, K_star)
        sigma_pred = jnp.sqrt(jnp.diag(K_star_star - jnp.dot(v.T, v)))
        
        return mu_pred, sigma_pred

    @jit
    def __call__(self, y, X_test):
        mu, sigma = self.condition(y, X_test)

        return mu, sigma


# -------------------------------------- LOW RANK GP ------------------------------------- #
class LowRankGP(eqx.Module):
    mean: eqx.Module
    kernel: eqx.Module
    X: Float[Array, "N d"]
    diag: Float

    def __init__(self, kernel, X, diag=None, mean=None):
        self.X = X
        self.kernel = kernel

        if diag is None:
            diag = jnp.float32(1e-5)
        self.diag = jnp.log(diag)

        if mean is None:
            mean = ZeroMean()
        self.mean = mean

    @property
    def _diag(self):
        return jnp.exp(self.diag)

    @partial(jit, static_argnums=(2,))
    def nll(self, y, solver="chol"):
        # return cond(solver == "chol", self.chol_nll, self.chol_nll, y)
        return self.chol_nll(y)

    @eqx.filter_jit
    def chol_nll(self, y: JAXArray) -> JAXArray:
        diag = self._diag
        y_diff = y - self.mean(self.X)
        phiX = self.kernel.phi(self.X)
        n, m = phiX.shape
        A = phiX.T @ phiX + jnp.eye(m) * diag
        R = jnp.linalg.cholesky(A)
        y_loss = jnp.linalg.solve(R, phiX.T @ y_diff)

        lml_1 = -((y_diff.T @ y_diff) - (y_loss.T @ y_loss)) / (2 * diag)
        lml_2 = -0.5 * jnp.sum(jnp.log(jnp.diag(R)**2))
        lml_3 = m * jnp.log(m * diag)
        lml_4 = -0.5 * n * jnp.log(2 * jnp.pi * diag)
        lml = lml_1 + lml_2 + lml_3 + lml_4
        return -lml

    @eqx.filter_jit
    def condition(self, y_train, X_test):
        y_diff = y_train - self.mean(self.X)
        phiXt = self.kernel.phi(X_test)
        phiX = self.kernel.phi(self.X)
        n, m = phiX.shape
        A = phiX.T @ phiX + jnp.eye(m) * self._diag
        R = jnp.linalg.cholesky(A)

        # mean calculation
        R_phiy = jnp.linalg.solve(R, phiX.T @ y_diff)
        y_pred = jnp.linalg.solve(R.T, R_phiy)
        mu = phiXt @ y_pred + self.mean(X_test)

        # variance calculation
        R_phixt = jnp.linalg.solve(R, phiXt.T)
        V = R_phixt.T @ R_phixt * self._diag + self._diag
        # V = R_phixt.T @ R_phixt
        sigma = jnp.sqrt(jnp.diag(V))

        return mu, sigma

    # @partial(eqx.filter_jit, static_argnums=(4,))
    def __call__(self, y_train, X_test, diag=jnp.float32(0.)):
        diag = cond(diag == 0., lambda: self._diag, lambda: diag)
        return self.condition(y_train, X_test, diag)


# -------------------------------------- NYSTROM GP -------------------------------------- #
class SVGP(LowRankGP):
    """
    Sparse-Variational GP adapated from
    http://krasserm.github.io/2020/12/12/gaussian-processes-sparse/#optimization
    """

    mean: eqx.Module
    kernel: eqx.Module
    m: Float[Array, "R d"]  # inducing points
    X: Float[Array, "N d"]
    diag: Float
    R: int = eqx.field(static=True)

    def __init__(self, kernel, X, R, diag=None, mean=None, prior=None, key=jr.PRNGKey(0)):
        self.X = X
        self.kernel = kernel
        self.R = R

        if prior is not None:
            self.m = prior
        else:  # random choice of inducing points
            self.m = self.X[
                jr.choice(key, self.X.shape[0], (R,), replace=False)
            ]

        if diag is None:
            diag = jnp.float32(1e-5)
        self.diag = jnp.log(diag)

        if mean is None:
            mean = ZeroMean()
        self.mean = mean

    @jit
    def phi_opt(self, y):
        X_m = self.m
        diag = self._diag

        K_mm = K_mm = self.kernel(X_m, X_m) + diag * jnp.eye(X_m.shape[0])
        K_mm_inv = jnp.linalg.inv(K_mm)
        K_nm = self.kernel(self.X, self.m)
        K_mn = K_nm.T

        precision = (1.0 / self._diag)
        Sigma = jnp.linalg.inv(K_mm + precision * K_mn @ K_nm)
        
        y_diff = y - self.mean(self.X)
        mu_m = precision * (K_mm @ Sigma @ K_mn).dot(y_diff)
        mu_m += self.mean(X_m)
        A_m = K_mm @ Sigma @ K_mm    
        
        return mu_m, A_m, K_mm_inv

    @jit
    def nll(self, y):  # modification of traditional NLL to instead be ELBO
        """
        Negative lower bound on log marginal likelihood.
        """
        X = self.X
        n = X.shape[0]
        X_m = self.m
        diag = self._diag

        K_mm = self.kernel(X_m, X_m) + diag * jnp.eye(X_m.shape[0])
        K_mn = self.kernel(X_m, X)

        y_diff = y - self.mean(X)
        L = jnp.linalg.cholesky(K_mm)  # m x m
        A = jsp.linalg.solve_triangular(L, K_mn, lower=True) / diag  # m x n        
        AAT = A @ A.T  # m x m
        B = jnp.eye(X_m.shape[0]) + AAT  # m x m
        LB = jnp.linalg.cholesky(B)  # m x m
        c = jsp.linalg.solve_triangular(LB, A.dot(y_diff), lower=True) / diag  # m x 1
        
        lb = - n / 2 * jnp.log(2 * jnp.pi)
        lb -= jnp.sum(jnp.log(jnp.diag(LB)))
        lb -= n / 2 * jnp.log(diag ** 2)
        lb -= 0.5 / diag ** 2 * y_diff.T.dot(y_diff)
        lb += 0.5 * c.T.dot(c)
        lb -= 0.5 / diag ** 2 * n
        lb += 0.5 * jnp.trace(AAT)
        return -lb

    @jit
    def condition(self, y, X_test):
        X_m = self.m

        mu_m, A_m, K_mm_inv = self.phi_opt(y)
        K_ss = self.kernel(X_test, X_test)
        K_sm = self.kernel(X_test, X_m)
        K_ms = K_sm.T

        f_q = (K_sm @ K_mm_inv).dot(mu_m)
        f_q_cov1 = K_ss - K_sm @ K_mm_inv @ K_ms
        f_q_cov = f_q_cov1 + K_sm @ K_mm_inv @ A_m @ K_mm_inv @ K_ms
        f_q_sigma = jnp.sqrt(jnp.diag(f_q_cov))
        
        return f_q, f_q_sigma


# ------------------------------------ MEAN FUNCTIONS ------------------------------------ #
class ZeroMean(eqx.Module):
    @jax.jit
    def __call__(self, X: Float[Array, "N d"]) -> Float[Array, "N"]:
        return jnp.zeros(X.shape[0])


class ConstantMean(eqx.Module):
    mean: Float

    def __init__(self, mean):
        self.mean = mean
    
    @jax.jit
    def __call__(self, X: Float[Array, "N d"]) -> Float[Array, "N"]:
        return jnp.broadcast_to(self.mean, X.shape[0])
    

class LinearMean(eqx.Module):
    weights: Float[Array, "d"]

    def __init__(self, weights):
        self.weights = weights

    @jax.jit
    def __call__(self, X: Float[Array, "N d"]) -> Float[Array, "N"]:
        return jnp.dot(X, self.weights)
