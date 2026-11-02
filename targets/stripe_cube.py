import chex
import jax
import jax.numpy as jnp

from targets.cube import Cube


class StripeCube(Cube):
    """Continuous cube target with high density near every boundary face."""

    def __init__(
        self,
        dim: int,
        boundary_prob: float = 1.0,
        floor_prob: float = 1e-10,
        vanish_power: float = 8.0,
    ) -> None:
        self.data_ndim = dim
        self.boundary_prob = float(boundary_prob)
        self.floor_prob = float(floor_prob)
        self.vanish_power = float(vanish_power)
        side_length = 1.0
        if side_length <= 0:
            raise ValueError("max_coord must be greater than min_coord")
        if self.boundary_prob <= 0:
            raise ValueError("boundary_prob must be positive")
        if self.floor_prob <= 0:
            raise ValueError("floor_prob must be positive")
        if self.vanish_power <= 0:
            raise ValueError("vanish_power must be positive")
        if self.boundary_prob <= self.floor_prob:
            raise ValueError("boundary_prob must be greater than floor_prob")

        self.side_length = float(side_length)
        self.center_distance = 0.5 * self.side_length

        super().__init__(
            dim,
            log_prob_fn=self._log_prob,
            sample_fn=lambda seed, sample_shape: 0.5 * jnp.ones(sample_shape + (dim,)),
        )

    def _edge_score(self, x: chex.Array) -> chex.Array:
        coord_distance_to_boundary = jnp.minimum(
            x - self.min_coord,
            self.max_coord - x,
        )
        distance_to_boundary = jnp.min(coord_distance_to_boundary, axis=-1)
        edge_score = 1.0 - distance_to_boundary / self.center_distance
        return jnp.clip(edge_score, min=0.0, max=1.0)

    def _log_prob(self, x: chex.Array) -> chex.Array:
        edge_score = self._edge_score(x)
        prob = self.floor_prob + (self.boundary_prob - self.floor_prob) * jnp.power(
            edge_score,
            self.vanish_power,
        )
        return jnp.log(prob)

    def log_prob_marginal_pair(self, x_2d, i, j, constrained=True):
        center = 0.5 * (self.min_coord + self.max_coord)
        x = jnp.full((x_2d.shape[0], self.dim), center)
        x = x.at[:, i].set(x_2d[:, 0])
        x = x.at[:, j].set(x_2d[:, 1])
        return self.log_prob(x, constrained=constrained)

    def _compute_true_logZ(self) -> float:
        volume = self.side_length**self.dim
        mean_edge_power = self.dim / (self.dim + self.vanish_power)
        z = volume * (
            self.floor_prob + (self.boundary_prob - self.floor_prob) * mean_edge_power
        )
        return float(jnp.log(z))


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

    # for dim in [10, 50, 100]:
    #     dist = StripeCube(dim=dim)
    #     check(dist)
    dist = StripeCube(dim=2)
    dist.visualise(show=True)
