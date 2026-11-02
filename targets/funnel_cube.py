import chex
import distrax
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from targets.cube import Cube


class FunnelCube(Cube):
    def __init__(
        self,
        dim,
        dominant_loc=0.6,
        dominant_scale=0.22,
        sigma_min=0.02,
        sigma_max=0.30,
        other_loc=0.5,
    ) -> None:
        self.data_ndim = dim
        self.dominant_loc = dominant_loc
        self.dominant_scale = dominant_scale
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.other_loc = other_loc
        self.dist_dominant = distrax.Normal(
            jnp.array([dominant_loc]), jnp.array([dominant_scale])
        )
        self.mean_other = other_loc * jnp.ones(dim - 1, dtype=float)
        self.cov_eye = jnp.eye(dim - 1).reshape((1, dim - 1, dim - 1))

        def log_prob(x: chex.Array):
            batched = x.ndim == 2
            if not batched:
                x = x[None, :]

            dominant_x = x[:, 0]

            log_density_dominant = self.dist_dominant.log_prob(dominant_x)

            # Scale funnel directly inside [0, 1].
            # x0 = 0 -> sigma_min
            # x0 = 1 -> sigma_max
            log_sigma_min = jnp.log(sigma_min)
            log_sigma_max = jnp.log(sigma_max)

            log_sigma = log_sigma_min + dominant_x[:, None] * (
                log_sigma_max - log_sigma_min
            )

            sigma2 = jnp.exp(2.0 * log_sigma)

            centered_other = x[:, 1:] - other_loc

            log_density_other_per_dim = (
                -0.5 * jnp.log(2.0 * jnp.pi)
                - log_sigma
                - 0.5 * centered_other**2 / sigma2
            )

            log_density_other = jnp.sum(log_density_other_per_dim, axis=-1)

            log_prob = log_density_dominant + log_density_other

            if not batched:
                log_prob = jnp.squeeze(log_prob, axis=0)

            return log_prob

        def sample(seed: chex.PRNGKey, sample_shape: chex.Shape = ()) -> chex.Array:
            key1, key2 = jax.random.split(seed)
            dominant_x = self.dist_dominant.sample(
                seed=key1, sample_shape=sample_shape
            )  # (B,1)
            x_others = self._dist_other(dominant_x).sample(seed=key2)  # (B, dim-1)
            return jnp.hstack([dominant_x, x_others])

        super().__init__(
            dim,
            log_prob_fn=log_prob,
            sample_fn=sample,
        )

    def _dist_other(self, dominant_x):
        log_sigma_min = jnp.log(self.sigma_min)
        log_sigma_max = jnp.log(self.sigma_max)
        log_sigma = log_sigma_min + dominant_x[:, None] * (
            log_sigma_max - log_sigma_min
        )
        sigma2 = jnp.exp(2.0 * log_sigma)
        cov_other = sigma2.reshape(-1, 1, 1) * self.cov_eye
        # use covariance matrix, not std
        return distrax.MultivariateNormalFullCovariance(self.mean_other, cov_other)

    def _compute_true_logZ(self) -> float:
        def log_integrand(x0):
            # Marginal density of the dominant coordinate.
            log_p0 = norm.logpdf(
                x0,
                loc=self.dominant_loc,
                scale=self.dominant_scale,
            )

            # sigma(x0)
            log_sigma = np.log(self.sigma_min) + x0 * (
                np.log(self.sigma_max) - np.log(self.sigma_min)
            )
            sigma = np.exp(log_sigma)

            # Probability that one conditional coordinate lies in [0, 1].
            p_inside = norm.cdf((1.0 - self.other_loc) / sigma) - norm.cdf(
                -self.other_loc / sigma
            )

            # Guard only against numerical log(0).
            p_inside = np.clip(
                p_inside,
                np.finfo(float).tiny,
                1.0,
            )

            return log_p0 + (self.dim - 1) * np.log(p_inside)

        # Find the maximum of the log-integrand on [0,1].
        # We subtract it before numerical integration to avoid
        # underflow in high dimension.
        result = minimize_scalar(
            lambda x: -log_integrand(x),
            bounds=(0.0, 1.0),
            method="bounded",
        )

        log_scale = log_integrand(result.x)

        integral, _ = quad(
            lambda x: np.exp(log_integrand(x) - log_scale),
            0.0,
            1.0,
            epsabs=1e-12,
            epsrel=1e-10,
            limit=200,
        )

        return float(log_scale + np.log(integral))

    def log_prob_marginal_pair(
        self, x_2d: chex.Array, i: int, j: int, constrained: bool = True
    ) -> chex.Array:
        return self.log_prob(x_2d, constrained=constrained)


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
            f"\t Min log_prob: {jnp.min(log_probs):.4f}, Max log_prob: {jnp.max(log_probs):.4f}"
        )
        print("\t", dist.Z, dist.log_Z)

    for dim in [2, 20, 50]:
        dist = FunnelCube(dim=dim)
        check(dist)

    dist = FunnelCube(dim=2)
    samples = dist.sample(jax.random.PRNGKey(0), (1000,))
    dist.visualise(show=True)

    # sampling
    # dim 2
    # Min log_prob: -312.1877, Max log_prob: 2.1409
    # 0.95808 -0.04282400384545326
    # dim 20
    # Min log_prob: -3514.1794, Max log_prob: 9.5114
    # 0.906038 -0.09867405146360397
    # dim 50
    # Min log_prob: -7322.2593, Max log_prob: 8.4534
    # 0.864569 -0.14552414417266846
