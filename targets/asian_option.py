from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from typing import Any

from targets.cube import Cube

PROBLEMS = {
    "easy_1": {
        "dim": 10,
        "start_period": 111,
        "strike": 90,
        "Z_gt": 13.008,
    },
    "easy_2": {
        "dim": 10,
        "start_period": 111,
        "strike": 100,
        "Z_gt": 5.863,
    },
    "easy_3": {
        "dim": 10,
        "start_period": 12,
        "strike": 90,
        "Z_gt": 11.367,
    },
    "easy_4": {
        "dim": 10,
        "start_period": 12,
        "strike": 100,
        "Z_gt": 3.617,
    },
    "hard_1": {
        "dim": 120,
        "start_period": 1,
        "strike": 90,
        "Z_gt": 11.207,
    },
    "hard_2": {
        "dim": 120,
        "start_period": 1,
        "strike": 100,
        "Z_gt": 3.367,
    },
}


class ArithmeticAsianOption(Cube):
    "Implementation of the Example 3 from `Quasi-Monte Carlo methods with applications in finance` paper"

    def __init__(
        self,
        dim: int = 120,
        initial_price: float = 100.0,
        strike: float = 90,
        maturity: float = 120 / 365,
        risk_free_rate: float = math.log(1.09),
        volatility: float = 0.2,
        start_period=1,
        total_period=365,
        Z_gt: float = 0.0,
        corrector: float = 0.0,
    ) -> None:
        self.initial_price = initial_price
        self.strike = strike
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.maturity = maturity
        self.start_period = start_period
        self.total_period = total_period

        self._compute_true_logZ = lambda *args: jnp.log(Z_gt)

        self.times = start_period / total_period + (
            maturity - start_period / total_period
        ) * jnp.arange(0, dim, dtype=float) / (dim - 1)
        self.delta_times = jnp.diff(
            jnp.concatenate([jnp.zeros((1,), dtype=float), self.times])
        )

        super().__init__(
            dim=dim,
            log_prob_fn=self._log_integrand_batch,
            min_coord=0.0,
            max_coord=1.0,
            mask_outside=True,
            sample_fn=lambda seed, sample_shape: 0.5 * jnp.ones(sample_shape + (dim,)),
            corrector=corrector,
            is_non_negative=True,
        )

    def _asset_path(
        self,
        u: jax.Array,
        initial_price: float,
    ) -> jax.Array:

        u = jnp.asarray(u)  # crucial: convert first
        eps = jnp.finfo(u.dtype).eps

        shocks = jax.scipy.special.ndtri(jnp.clip(u, eps, 1.0 - eps))

        brownian_path = jnp.cumsum(
            jnp.sqrt(self.delta_times) * shocks,
            axis=-1,
        )

        log_path = (
            jnp.log(initial_price)
            + (self.risk_free_rate - 0.5 * self.volatility**2) * self.times
            + self.volatility * brownian_path
        )

        return jnp.exp(log_path)

    def _arithmetic_payoff(
        self,
        u: jax.Array,
        initial_price: float,
    ) -> jax.Array:
        path = self._asset_path(u, initial_price)
        average = jnp.mean(path, axis=-1)
        return jnp.maximum(
            average - self.strike,
            0.0,
        )

    def _log_integrand_batch(
        self,
        u: jax.Array,
    ) -> jax.Array:
        payoff = self._arithmetic_payoff(u, self.initial_price)
        safe_payoff = jnp.maximum(payoff, jnp.finfo(payoff.dtype).tiny)
        log_reward = -self.risk_free_rate * self.maturity + jnp.log(safe_payoff)
        return jnp.where(
            payoff > 0.0,
            log_reward,
            -jnp.inf,
        )


if __name__ == "__main__":

    def check(dist):
        N = 10**6
        key = jax.random.PRNGKey(0)
        samples = jax.random.uniform(key, shape=(N, dist.dim))
        log_probs = dist.log_prob(samples, constrained=True)
        if dist.needs_correction:
            log_probs = jnp.logaddexp(log_probs, jnp.log(dist.corrector))
        print(
            f"\t Min log_prob: {jnp.min(log_probs)}, Max log_prob: {jnp.max(log_probs)}"
        )
        print("\t", dist.Z, dist.log_Z)

    # for name, cfg in PROBLEMS.items():
    #     target = ArithmeticAsianOption(**cfg)
    #     print(name, target.Z)
    #     check(target)

    target = ArithmeticAsianOption(**PROBLEMS["hard_1"], corrector=0)

    from scipy.stats import qmc

    def compute_estimates_from_weights(samples, log_is_weights, target):
        if not target.is_non_negative:
            signs = target.sign(samples)
        else:
            signs = jnp.ones_like(log_is_weights)
        log_abs_sum, sign_Z = jax.scipy.special.logsumexp(
            log_is_weights,
            b=signs,
            return_sign=True,
        )
        log_Z = log_abs_sum - jnp.log(log_is_weights.shape[0])
        Z = sign_Z * jnp.exp(log_Z)
        return log_Z, Z

    def rnd_qmc(key, target, num_samples):
        # fmt: off
        np_seed = int(jax.random.randint(key, (), minval=0, maxval=1_000_000, dtype=jnp.uint32))
        sampler = qmc.Sobol(d=target.dim, scramble=True, seed=np_seed)
        samples = sampler.random_base2(m=int(math.ceil(math.log2(num_samples))))
        # currently we test qmc only in a unit cube, so we don't account for volume of the domain in log_is_weights
        log_is_weights = target.log_prob(samples)
        return compute_estimates_from_weights(samples, log_is_weights, target)

    log_Z, Z = rnd_qmc(jax.random.PRNGKey(0), target, 524288)
    print(log_Z, Z)

    # # grad_log_prob: shape (batch, dim)
    # grad_log_prob_fn = jax.vmap(jax.grad(target_log_prob))
    # grad_log_probs = grad_log_prob_fn(grid_samples)

    # # Compute the norms for min/max reporting
    # grad_norms = jnp.linalg.norm(grad_log_probs, axis=-1)

    # min_grad = jnp.min(grad_norms)
    # max_grad = jnp.max(grad_norms)

    # print(f"Min grad norm: {min_grad}, Max grad norm: {max_grad}")
