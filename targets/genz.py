from __future__ import annotations

import math
from typing import Callable, Dict, Type

import chex
import jax
import jax.numpy as jnp
from jax.scipy.special import erf, gammaln, betaln

from targets.cube import Cube


Array = chex.Array


def _expand_to_array(value: float | Array, dim: int) -> Array:
    value = jnp.asarray(value, dtype=jnp.float32)
    if value.ndim == 0:
        return jnp.full((dim,), value)
    if value.shape != (dim,):
        raise ValueError(f"array shape {value.shape} doesn't equal d={dim}")
    return value


class GenzContinuous(Cube):
    """Continuous Genz: exp(-sum_k a_k |x_k - u_k|)."""

    "Small a's make it more uniform, large a's - more degenerate."
    """u's controls center of a mode."""

    def __init__(
        self,
        dim: int,
        a: float | Array = 1.3,
        u: float | Array = 0.55,
    ):
        self.a = _expand_to_array(a, dim)
        self.u = _expand_to_array(u, dim)
        assert jnp.all(self.a > 0) and jnp.all((self.u > 0) & (self.u < 1))
        super().__init__(
            dim=dim,
            log_prob_fn=self._log_integrand,
            sample_fn=lambda seed, sample_shape: 0.5 * jnp.ones(sample_shape + (dim,)),
        )

    def _log_integrand(self, x: Array) -> Array:
        return -jnp.sum(self.a * jnp.abs(x - self.u), axis=-1)

    def _compute_true_logZ(self) -> float:
        au = self.a * self.u
        a1u = self.a * (1.0 - self.u)
        terms = (2.0 - jnp.exp(-au) - jnp.exp(-a1u)) / self.a
        return float(jnp.sum(jnp.log(terms)))


class GenzCornerPeak(Cube):
    """Corner peak Genz: (1 + sum_k a_k x_k)^(-(d + 1))."""

    "Small a's make it more uniform, large a's - more degenerate."

    def __init__(self, dim: int, a: float | Array = 5.0):
        self.a = _expand_to_array(a, dim)
        assert jnp.all(self.a > 0)
        super().__init__(
            dim=dim,
            log_prob_fn=self._log_integrand,
            sample_fn=lambda seed, sample_shape: 0.5 * jnp.ones(sample_shape + (dim,)),
        )

    def _log_integrand(self, x: Array) -> Array:
        return -(self.dim + 1.0) * jnp.log1p(jnp.sum(self.a * x, axis=-1))

    def _compute_true_logZ(self) -> float:
        if jnp.allclose(self.a, self.a[0]):
            a = self.a[0]
            return (
                betaln(1.0 / a, self.dim + 1.0)
                - gammaln(self.dim + 1.0)
                - (self.dim + 1.0) * jnp.log(a)
            )
        total = 0.0
        for mask in range(1 << self.dim):
            subset_sum = 0.0
            bits = 0
            for i, ai in enumerate(self.a):
                if mask & (1 << i):
                    subset_sum += ai
                    bits += 1
            total += ((-1.0) ** bits) / (1.0 + subset_sum)
        return jnp.log(total) - gammaln(self.dim + 1.0) - jnp.sum(jnp.log(self.a))


class GenzDiscontinuous(Cube):
    """Discontinuous Genz: exp(sum_k a_k x_k) on x_k <= u_k for all k, else 0."""

    "Small a's make it more uniform, large a's - more degenerate."

    def __init__(
        self,
        dim: int,
        a: float | Array = 5.0,
        u: float | Array = 0.5,
        corrector: float = 0.0,
    ):
        self.a = _expand_to_array(a, dim)
        self.u = _expand_to_array(u, dim)
        assert jnp.all(self.a > 0) and jnp.all((self.u >= 0) & (self.u <= 1))
        super().__init__(
            dim=dim,
            log_prob_fn=self._log_integrand,
            sample_fn=lambda seed, sample_shape: 0.5 * jnp.ones(sample_shape + (dim,)),
            corrector=corrector,
        )

    def _log_integrand(self, x: Array) -> Array:
        return jnp.where(
            jnp.all(x <= self.u, axis=-1),
            jnp.sum(self.a * x, axis=-1),
            -jnp.inf,
        )

    def _compute_true_logZ(self) -> float:
        return float(jnp.sum(jnp.log(jnp.expm1(self.a * self.u) / self.a)))


class GenzGaussianPeak(Cube):
    """Gaussian peak Genz: exp(-sum_k a_k^2 (x_k - u_k)^2)."""

    "Small a's make it more uniform, large a's - more degenerate."

    def __init__(self, dim: int, a: float | Array = 5.0, u: float | Array = 0.5):
        self.a = _expand_to_array(a, dim)
        self.u = _expand_to_array(u, dim)
        assert jnp.all(self.a > 0) and jnp.all((self.u > 0) & (self.u < 1))
        super().__init__(
            dim=dim,
            log_prob_fn=self._log_integrand,
            sample_fn=lambda seed, sample_shape: 0.5 * jnp.ones(sample_shape + (dim,)),
        )

    def _log_integrand(self, x: Array) -> Array:
        return -jnp.sum((self.a**2) * (x - self.u) ** 2, axis=-1)

    def _compute_true_logZ(self) -> float:
        terms = (
            jnp.sqrt(jnp.pi)
            / (2.0 * self.a)
            * (erf(self.a * (1.0 - self.u)) + erf(self.a * self.u))
        )
        return float(jnp.sum(jnp.log(terms)))


class GenzProductPeak(Cube):
    """Product peak Genz: prod_k 1 / (a_k^-2 + (x_k - u_k)^2)."""

    """Optional rescaling by the maximum value to make reward in [0,1] interval."""
    "Small a's make it more uniform, large a's - more degenerate."

    def __init__(
        self,
        dim: int,
        a: float | Array = 5.0,
        u: float | Array = 0.5,
        rescale: bool = True,
    ):
        self.a = _expand_to_array(a, dim)
        self.u = _expand_to_array(u, dim)

        assert jnp.all(self.a > 0)
        assert jnp.all((self.u > 0) & (self.u < 1))

        self.log_scale = 2.0 * jnp.sum(jnp.log(self.a)) if rescale else 0.0

        super().__init__(
            dim=dim,
            log_prob_fn=self._log_integrand,
            sample_fn=lambda seed, sample_shape: 0.5 * jnp.ones(sample_shape + (dim,)),
        )

    def _log_integrand(self, x: Array) -> Array:
        log_reward = -jnp.sum(
            jnp.log(self.a**-2 + (x - self.u) ** 2),
            axis=-1,
        )
        return log_reward - self.log_scale

    def _compute_true_logZ(self) -> float:
        terms = self.a * (
            jnp.arctan(self.a * (1.0 - self.u)) + jnp.arctan(self.a * self.u)
        )
        log_Z = jnp.sum(jnp.log(terms))
        return log_Z - self.log_scale


class GenzOscillatory(Cube):
    """Oscillatory Genz: cos(2*pi*u + sum_k a_k x_k)."""

    def __init__(
        self, dim: int, a: float | Array = 5.0, u: float = 0.5, corrector: float = 0.0
    ):
        self.a = _expand_to_array(a, dim)
        self.u = _expand_to_array(u, 1)
        assert jnp.all(self.a > 0) and jnp.all((self.u > 0) & (self.u < 1))
        super().__init__(
            dim=dim,
            log_prob_fn=self._log_abs_integrand,
            sample_fn=lambda seed, sample_shape: 0.5 * jnp.ones(sample_shape + (dim,)),
            corrector=corrector,
            is_non_negative=False,
        )
        self._log_Z = None  # Z is signed
        self.Z = self._compute_true_Z()

    def _integrand(self, x: Array) -> Array:
        return jnp.cos(2.0 * jnp.pi * self.u[0] + jnp.sum(self.a * x, axis=-1))

    def _log_abs_integrand(self, x: Array) -> Array:
        return jnp.log(jnp.abs(self._integrand(x)))

    def sign(self, x: Array) -> Array:
        return jnp.sign(self._integrand(x))

    def _compute_true_Z(self) -> float:
        phase = 2.0 * jnp.pi * self.u[0] + 0.5 * jnp.sum(self.a)
        factors = 2.0 * jnp.sin(0.5 * self.a) / self.a
        Z = jnp.cos(phase) * jnp.prod(factors)
        return float(Z)


if __name__ == "__main__":

    def check(dist):
        N = 10**6
        key = jax.random.PRNGKey(0)
        samples = jax.random.uniform(key, shape=(N, dist.dim))
        log_probs = dist.log_prob(samples, constrained=True)
        if dist.needs_correction:
            log_probs = jnp.logaddexp(log_probs, jnp.log(dist.corrector))
        print("\t dim", dist.dim)
        print(
            f"\t Min log_prob: {jnp.min(log_probs)}, Max log_prob: {jnp.max(log_probs)}"
        )
        print("\t", dist.Z, dist.log_Z)

    dims = [5, 10, 20, 30]
    # print("GenzContinuous")
    # avals = [8.0, 5.0, 4.0, 3.0]
    # for d, a in zip(dims, avals):
    #     dist = GenzContinuous(d, a=a)
    #     check(dist)
    # dist = GenzProductPeak(2, a=8)
    # dist.visualise(None, show=True)

    # avals = [0.5, 0.25, 0.1, 0.08]
    # print("GenzCornerPeak")
    # for d, a in zip(dims, avals):
    #     dist = GenzCornerPeak(d, a=a)
    #     check(dist)
    # dist = GenzCornerPeak(dims[-1], a=avals[-1])
    # dist.visualise(None, show=True)

    # print("GenzDiscontinuous")
    # avals = [5.0, 4.0, 3.0, 3.0]
    # for d, a in zip(dims, avals):
    #     dist = GenzDiscontinuous(d, a=a)
    #     check(dist)
    # dist = GenzDiscontinuous(dims[3], a=avals[3])
    # dist.visualise(None, show=True)

    # print("GenzGaussianPeak")
    # avals = [8.0, 5.0, 4.0, 3.0]
    # for d, a in zip(dims, avals):
    #     dist = GenzGaussianPeak(d, a=a)
    #     check(dist)
    # dist = GenzGaussianPeak(dims[3], a=avals[3])
    # dist.visualise(None, show=True)

    # print("GenzProductPeak")
    # avals = [8.0, 5.0, 4.0, 4.0]
    # for d, a in zip(dims, avals):
    #     dist = GenzProductPeak(d, a=a)
    #     check(dist)
    # dist = GenzProductPeak(dims[3], a=avals[3])
    # dist.visualise(None, show=True)

    # print("GenzOscillatory")
    # for d in dims:
    #     dist = GenzOscillatory(d)
    #     check(dist)

    # dist = GenzOscillatory(2)
    # dist.visualise(None, show=True)
