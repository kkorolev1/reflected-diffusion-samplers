import jax.numpy as jnp
from flax import linen as nn


class RDSCondNet(nn.Module):
    dim: int

    num_layers: int = 2
    num_hid: int = 64
    outer_clip: float = 1e4
    inner_clip: float = 1e2

    weight_init: float = 1e-8
    bias_init: float = 0.1

    use_lp: bool = True

    def setup(self):
        self.timestep_phase = self.param(
            "timestep_phase", nn.initializers.zeros_init(), (1, self.num_hid)
        )
        self.timestep_coeff = jnp.linspace(start=0.1, stop=100, num=self.num_hid)[None]

        self.time_coder_state = nn.Sequential(
            [nn.Dense(self.num_hid), nn.gelu, nn.Dense(self.num_hid)]
        )

        self.condition_coder = nn.Sequential(
            [nn.Dense(self.num_hid), nn.gelu, nn.Dense(self.num_hid)]
        )

        self.time_coder_grad = None
        if self.use_lp:
            self.time_coder_grad = nn.Sequential(
                [nn.Dense(self.num_hid)]
                + [
                    nn.Sequential([nn.gelu, nn.Dense(self.num_hid)])
                    for _ in range(self.num_layers)
                ]
                + [nn.gelu]
                + [
                    nn.Dense(
                        self.dim,
                        kernel_init=nn.initializers.constant(self.weight_init),
                        bias_init=nn.initializers.constant(self.bias_init),
                    )
                ]
            )

        self.state_time_net = nn.Sequential(
            [
                nn.Sequential([nn.Dense(self.num_hid), nn.gelu])
                for _ in range(self.num_layers)
            ]
            + [
                nn.Dense(
                    self.dim,
                    kernel_init=nn.initializers.constant(1e-8),
                    bias_init=nn.initializers.zeros_init(),
                )
            ]
        )

    def get_fourier_features(self, timesteps):
        sin_embed_cond = jnp.sin(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        cos_embed_cond = jnp.cos(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def __call__(self, input_array, time_array, lgv_term, condition_array):
        time_array_emb = self.get_fourier_features(time_array)
        if condition_array.ndim == 0:
            condition_array = condition_array[None, None]
        elif condition_array.ndim == 1:
            condition_array = condition_array[..., None]

        cond_emb = self.condition_coder(condition_array)
        if len(input_array.shape) == 1:
            time_array_emb = time_array_emb[0]
            cond_emb = cond_emb[0]

        t_net1 = self.time_coder_state(time_array_emb)
        extended_input = jnp.concatenate((input_array, cond_emb, t_net1), axis=-1)
        out_state = self.state_time_net(extended_input)
        if self.use_lp:
            t_net2 = self.time_coder_grad(time_array_emb)
            lgv_term = jnp.clip(lgv_term, -self.inner_clip, self.inner_clip)
            out_state = out_state + t_net2 * lgv_term
        out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)
        return out_state
