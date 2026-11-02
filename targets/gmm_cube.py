import chex
import distrax
import jax
import jax.numpy as jnp
from targets.cube import Cube


class GMMCube(Cube):
    def __init__(
        self,
        dim,
        num_components=5,
        std=0.1,
        loc_margin=0.01,
        seed=1,
    ) -> None:
        side_length = 1.0
        margin = loc_margin * side_length

        assert std > 0.0
        assert 0.0 <= loc_margin < 0.5
        assert margin > 0.0

        key = jax.random.PRNGKey(seed)

        # ------------------------------------------------------------------
        # Component means
        #
        # Keeping the means away from the boundary prevents the probability
        # mass inside the cube from collapsing exponentially with dimension.
        # ------------------------------------------------------------------
        key, subkey = jax.random.split(key)

        self.locs = jax.random.uniform(
            subkey,
            shape=(num_components, dim),
            minval=0.0 + margin,
            maxval=1.0 - margin,
        )

        # ------------------------------------------------------------------
        # Covariances
        #
        # Isotropic covariance gives predictable scale in every dimension.
        # In particular, `std` has the same interpretation for all d.
        # ------------------------------------------------------------------
        self.covariances = std**2 * jnp.broadcast_to(
            jnp.eye(dim),
            (num_components, dim, dim),
        )

        component_dist = distrax.MultivariateNormalFullCovariance(
            loc=self.locs,
            covariance_matrix=self.covariances,
        )

        self.mixture_weights = distrax.Categorical(
            probs=jnp.ones(num_components) / num_components
        )

        self.mixture_distribution = distrax.MixtureSameFamily(
            mixture_distribution=self.mixture_weights,
            components_distribution=component_dist,
        )

        super().__init__(
            dim=dim,
            log_prob_fn=self._log_prob,
            sample_fn=self.mixture_distribution.sample,
        )

    def _log_prob(self, x: chex.Array) -> chex.Array:
        return self.mixture_distribution.log_prob(x)

    def _compute_true_logZ(
        self, num_samples: int = 1000000, seed: chex.PRNGKey = None
    ) -> float:
        if seed is None:
            seed = jax.random.PRNGKey(42)
        samples = self.sample_fn(
            seed=seed,
            sample_shape=(num_samples,),
        )
        inside = self.is_inside(samples)
        Z = jnp.mean(inside.astype(jnp.float32))
        return float(jnp.log(Z))

    def log_prob_marginal_pair(
        self,
        x_2d: chex.Array,
        i: int,
        j: int,
        constrained: bool = True,
    ) -> chex.Array:
        locs_ij = self.locs[:, [i, j]]
        covs_ij = self.covariances[:, [i, j]][:, :, [i, j]]

        component_dist = distrax.MultivariateNormalFullCovariance(
            locs_ij,
            covs_ij,
        )

        marginal_distribution = distrax.MixtureSameFamily(
            mixture_distribution=self.mixture_weights,
            components_distribution=component_dist,
        )
        log_prob_fn = lambda x: marginal_distribution.log_prob(x)
        return Cube(2, log_prob_fn).log_prob(
            x_2d,
            constrained=constrained,
        )


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

    dist = GMMCube(dim=2, num_components=7, loc_margin=0.01, seed=0)
    check(dist)
    samples = dist.sample(jax.random.PRNGKey(0), (1000,))
    # dist.visualise(samples, show=True)
    dist.visualise(show=True)

    # dist = GMMCube(dim=20)
    # check(dist)
    # samples = dist.sample(jax.random.PRNGKey(0), (1000,))
    # dist.visualise(samples, show=True)

    # dist = GMMCube(dim=50)
    # check(dist)
    # samples = dist.sample(jax.random.PRNGKey(0), (1000,))
    # dist.visualise(samples, show=True)
