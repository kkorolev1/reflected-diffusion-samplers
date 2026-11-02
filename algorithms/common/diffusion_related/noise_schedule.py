import jax.numpy as jnp


def get_linear_noise_schedule(
    total_steps, sigma_min=0.01, sigma_max=10.0, reverse=True
):
    if reverse:

        def linear_noise_schedule(step):
            t = step / total_steps
            return 0.5 * ((1 - t) * sigma_min + t * sigma_max)

    else:

        def linear_noise_schedule(step):
            t = (total_steps - step) / total_steps
            return 0.5 * ((1 - t) * sigma_min + t * sigma_max)

    return linear_noise_schedule


def get_cosine_noise_schedule(
    total_steps, sigma_min=0.01, sigma_max=10.0, s=0.008, pow=2, reverse=True
):
    if reverse:

        def cosine_noise_schedule(step):
            t = step / total_steps
            offset = 1 + s
            return (
                0.5
                * (sigma_max - sigma_min)
                * jnp.cos(0.5 * jnp.pi * (offset - t) / offset) ** pow
                + 0.5 * sigma_min
            )

    else:

        def cosine_noise_schedule(step):
            t = (total_steps - step) / total_steps
            offset = 1 + s
            return (
                0.5
                * (sigma_max - sigma_min)
                * jnp.cos(0.5 * jnp.pi * (offset - t) / offset) ** pow
                + 0.5 * sigma_min
            )

    return cosine_noise_schedule


def get_constant_noise_schedule(value, reverse=True):
    def constant_noise_schedule(step):
        return jnp.array(value)

    return constant_noise_schedule


def get_ve_noise_schedule(
    total_steps: int,
    sigma_min: float = 0.01,
    sigma_max: float = 10.0,
):
    """Variance-exploding SDE with SMLD (NCSN) discretization.

    Returns a dict of schedule helpers for VE diffusion.
    Sigmas are log-linearly spaced: sigma_i = sigma_min * (sigma_max / sigma_min)^(i / N).
    """
    discrete_sigmas = jnp.exp(
        jnp.linspace(jnp.log(sigma_min), jnp.log(sigma_max), total_steps)
    )

    def sigma_at(step: int):
        return discrete_sigmas[step]

    def adjacent_sigma(step: int):
        return jnp.where(step == 0, 0.0, discrete_sigmas[step - 1])

    def g_at(step):
        """SMLD forward noise increment g_i = sqrt(sigma_i^2 - sigma_{i-1}^2)."""
        sigma = sigma_at(step)
        sigma_prev = adjacent_sigma(step)
        return jnp.sqrt(sigma**2 - sigma_prev**2)

    return {
        "discrete_sigmas": discrete_sigmas,
        "g_at": g_at,
    }
