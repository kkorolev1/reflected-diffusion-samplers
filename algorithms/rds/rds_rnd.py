from functools import partial

import jax
import jax.numpy as jnp
import numpyro.distributions as npdist

from algorithms.common.types import Array, RandomKey


def sample_kernel(key_gen, mean, scale, target):
    key, key_gen = jax.random.split(key_gen)
    eps = jax.random.normal(key, shape=mean.shape)
    return target.reflect(mean + scale * eps), eps, key_gen


def per_sample_rnd(
    key: RandomKey,
    model_state,
    params,
    input_state: Array,
    aux_tuple,
    initial_dist,
    target,
    num_steps: int,
    use_lp: bool,
    stop_grad: bool = False,
    prior_to_target: bool = True,
):
    (ve_schedule,) = aux_tuple
    g_at = ve_schedule["g_at"]

    if target.needs_correction:
        target_log_prob = lambda x: jnp.logaddexp(
            target.log_prob(x), jnp.log(target.corrector)
        )
    else:
        target_log_prob = target.log_prob

    reflect_sample_kernel = partial(sample_kernel, target=target)

    def simulate_prior_to_target(state, per_step_input):
        x, key_gen = state
        step = per_step_input.astype(jnp.int32)
        t = step.astype(jnp.float32) / (num_steps - 1)

        if stop_grad:
            x = jax.lax.stop_gradient(x)

        g_t = g_at(step)

        if use_lp:
            langevin = jax.lax.stop_gradient(jax.grad(target_log_prob)(x))
        else:
            langevin = jnp.zeros(x.shape[0])

        model_output = model_state.apply_fn(params, x, t * jnp.ones(1), langevin)

        fwd_mean = x + (g_t**2) * model_output
        fwd_scale = g_t
        x_new, noise, key_gen = reflect_sample_kernel(key_gen, fwd_mean, fwd_scale)

        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        running_cost = 0.5 * (g_t**2) * jnp.sum(jnp.square(model_output))
        stochastic_cost = (model_output * noise).sum() * g_t

        next_state = (x_new, key_gen)
        per_step_output = (running_cost, stochastic_cost, x_new)
        return next_state, per_step_output

    if prior_to_target:
        init_x = input_state
        aux = (init_x, key)
        aux, per_step_output = jax.lax.scan(
            simulate_prior_to_target, aux, jnp.arange(num_steps)[::-1]
        )
        terminal_x, _ = aux
    else:
        raise NotImplementedError
    terminal_cost = initial_dist.log_prob(terminal_x) - target_log_prob(terminal_x)
    running_cost, stochastic_cost, trajectory = per_step_output
    return (
        terminal_x,
        trajectory,
        running_cost,
        stochastic_cost,
        terminal_cost,
    )


def rnd(
    key_gen: RandomKey,
    model_state,
    params,
    batch_size: int,
    aux_tuple,
    initial_dist,
    target,
    num_steps: int,
    use_lp: bool,
    stop_grad: bool = False,
    prior_to_target: bool = True,
    terminal_xs: Array | None = None,
    accumulate_costs: bool = True,
):

    if prior_to_target:
        key, key_gen = jax.random.split(key_gen)
        input_states = initial_dist.sample(seed=key, sample_shape=(batch_size,))
    else:
        input_states = terminal_xs

    seeds = jax.random.split(key_gen, num=batch_size)
    (
        terminal_xs,
        trajectories,
        running_costs,
        stochastic_costs,
        terminal_costs,
    ) = jax.vmap(
        per_sample_rnd,
        in_axes=(0, None, None, 0, None, None, None, None, None, None, None),
    )(
        seeds,
        model_state,
        params,
        input_states,
        aux_tuple,
        initial_dist,
        target,
        num_steps,
        use_lp,
        stop_grad,
        prior_to_target,
    )
    # costs are accumulated for evaluation
    if accumulate_costs:
        return (
            terminal_xs,
            running_costs.sum(1),
            stochastic_costs.sum(1),
            terminal_costs,
        )
    return terminal_xs, running_costs, stochastic_costs, terminal_costs


def kl_loss_reparam_fn(
    key: RandomKey,
    model_state,
    params,
    rnd_partial,
):
    """Minimize E_Q[0.5 * int g(t)^2 ||f_theta||^2 dt - log r(Y_1)]."""
    _, running_costs, stochastic_costs, terminal_costs = rnd_partial(
        key, model_state, params
    )
    losses = running_costs.sum(1) + terminal_costs
    loss = jnp.mean(losses)
    return loss, (
        jax.lax.stop_gradient(losses),
        jax.lax.stop_gradient(
            -(running_costs.sum(1) + stochastic_costs.sum(1) + terminal_costs)
        ),
    )


def leave_one_out_baseline(values):
    batch_size = values.shape[0]

    if batch_size <= 1:
        return jnp.zeros_like(values)

    return (jnp.sum(values, axis=0, keepdims=True) - values) / (batch_size - 1)


def cost_to_go(values):
    """
    Input:
        values[b] = [c_0, c_1, ..., c_{K-1}]
    Output:
        [c_1 + ... + c_{K-1},
         c_2 + ... + c_{K-1},
         ...,
         0]
    """
    inclusive_reverse = jnp.flip(
        jnp.cumsum(
            jnp.flip(values, axis=1),
            axis=1,
        ),
        axis=1,
    )
    return inclusive_reverse - values


def kl_loss_reinforce_fn(
    key: RandomKey,
    model_state,
    params,
    rnd_partial,
    use_baseline=True,
):
    (
        _,
        running_costs,
        stochastic_costs,
        terminal_costs,
    ) = rnd_partial(
        key,
        model_state,
        params,
    )

    # after differentiation this thing below is the partial derivative dJ / dtheta with detached trajectories
    # producing gradients of the control
    control_losses = running_costs.sum(1) + jax.lax.stop_gradient(terminal_costs)

    # reinforce component with costs-to-go multiplied by scores
    costs_to_go = (
        cost_to_go(jax.lax.stop_gradient(running_costs))
        + jax.lax.stop_gradient(terminal_costs)[:, None]
    )
    if use_baseline:
        baseline = jax.lax.stop_gradient(leave_one_out_baseline(costs_to_go))
    else:
        baseline = jnp.zeros_like(costs_to_go)

    advantages = jax.lax.stop_gradient(costs_to_go - baseline)
    scores = stochastic_costs - jax.lax.stop_gradient(stochastic_costs)
    reinforce_losses = jnp.sum(advantages * scores, 1)

    losses = control_losses + reinforce_losses

    loss = jnp.mean(losses)

    return loss, (
        jax.lax.stop_gradient(control_losses),
        jax.lax.stop_gradient(
            -(running_costs.sum(1) + stochastic_costs.sum(1) + terminal_costs)
        ),
    )
