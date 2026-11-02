# ---------------------------------------------------------------------------------------- #
#                               GENZ FUNCTION IMPLEMENTATIONS                              #
# ---------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
from scipy.integrate import nquad
from sympy import symbols, exp, cos
from sympy.integrals import integrate
import math
from .utils import area
import torchquad
import numpy as np


# -------------------------------------- QUADRATURE -------------------------------------- #
def genz_quad(genz, N_d):
    torchquad.set_up_backend("numpy", "float64")
    bounds = [bound.tolist() for bound in genz.bounds.T]
    quad = torchquad.Simpson()
    d = len(bounds)
    N = N_d ** d
    f_np = lambda x: np.array(genz(x)).reshape(-1)

    result = quad.integrate(f_np, dim=d, N=N, integration_domain=bounds)

    return result


# -------------------------------------- BASE CLASS -------------------------------------- #
class GenzProblem:
    def __init__(self, d, bounds, params):
        self.d = d
        self.bounds = bounds
        self.params = params
        self.f = None

    def sample(self, N, noise=None, key=jax.random.PRNGKey(2023)):
        X = jax.random.uniform(
            key, (N, self.d), minval=self.bounds[0, :], maxval=self.bounds[1, :]
        )

        y = jax.vmap(self.f)(X)
        if noise is not None:
            y_noisy = y + jax.random.normal(key, (N,)) * noise
        else:
            y_noisy = y
        
        return X, y, y_noisy
    
    def symbolic(self):
        raise NotImplementedError("Symbolic solution not implemented for this problem")

    def quad(self, **kwargs):
        quadf = lambda *x: self.f(jnp.array(x))
        dim_bounds = [self.bounds[:, i] for i in range(self.d)]
        return nquad(quadf, dim_bounds, **dict(kwargs))
    
    def mc(self, **kwargs):
        N = kwargs.pop("N", 1000000)
        _, _, samples = self.sample(N, **kwargs)
        mc_mu = samples.mean()
        mc_var = samples.var() / N
        return mc_mu, mc_var

    def solution(self, solver, **kwargs):
        if solver == "quad":
            quadf = lambda *x: self.f(jnp.array(x))
            dim_bounds = [self.bounds[:, i] for i in range(self.d)]
            return nquad(quadf, dim_bounds, **dict(kwargs))
        elif solver == "symbolic":
            return self.symbolic()
        elif solver == "mc":
            return self.mc(**kwargs)
        else:
            raise ValueError(f"Invalid solver: {solver}")

    def __call__(self, X):
        return jax.vmap(self.f)(X)


# ------------------------------------ GENZ CONTINUOUS ----------------------------------- #
class GenzContinuous(GenzProblem):
    def __init__(self, d, bound=1, random_params=False, key=jax.random.PRNGKey(2023),):
        # init params and f
        if random_params:
            u = jax.random.uniform(key, (d,), minval=0.0, maxval=1.0)
            a = jax.random.uniform(key, (d,), minval=0.0, maxval=10.0)
        else:
            u = jnp.repeat(jnp.array([0.5]), d)
            a = jnp.repeat(jnp.array([5.0]), d)

        # init super object
        params = {"a": a, "u": u}
        bounds = jnp.tile(jnp.array([[0], [bound]]), d)
        super().__init__(d, bounds, params)

        # init f
        self.f = self.init_f()

    def init_f(self):
        a = self.params["a"]
        u = self.params["u"]

        def f(x):
            return jnp.exp(-jnp.sum(a * jnp.abs(x - u)))
    
        return f
    
    def symbolic(self):
        a = self.params["a"]
        u = self.params["u"]
        d = self.bounds.shape[-1]
        
        # create symbolic expression
        symbol_str = " ".join([f"x{i}" for i in range(d)])
        x_sym = symbols(symbol_str)
        expr = exp(-1 * sum([a[i] * abs(x_sym[i] - u[i]) for i in range(d)]))

        # integrate
        ub = self.bounds[1, 1].item()
        sympy_bounds = [(x_sym[i], 0, ub) for i in range(d)]
        sol = float(integrate(expr, *sympy_bounds).evalf())
        return (sol, 0.)


# ---------------------------------- GENZ DISCONTINUOUS ---------------------------------- #
class GenzDiscontinuous(GenzProblem):
    def __init__(self, d, bound=1, random_params=False, key=jax.random.PRNGKey(2023),):
        # init params and f
        if random_params:
            u = jax.random.uniform(key, (d,), minval=0.0, maxval=1.0)
            a = jax.random.uniform(key, (d,), minval=0.0, maxval=10.0)
        else:
            u = jnp.repeat(jnp.array([0.5]), d)
            a = jnp.repeat(jnp.array([5.0]), d)

        # init super object
        params = {"a": a, "u": u }
        bounds = jnp.tile(jnp.array([[0], [bound]]), d)
        super().__init__(d, bounds, params)

        # init f
        self.f = self.init_f()

    def init_f(self):
        a = self.params["a"]
        u = self.params["u"]

        def f(x):
            return jnp.where((x <= u).all(), jnp.exp(jnp.sum(a * x)), 0.0)
    
        return f

    def symbolic(self):
        a = self.params["a"]
        d = self.bounds.shape[-1]

        # create symbolic expression
        symbol_str = " ".join([f"x{i}" for i in range(d)])
        x_sym = symbols(symbol_str)
        expr = exp(sum([a[i] *x_sym[i] for i in range(d)]))

        # integrate
        ub_integration = self.params["u"].tolist()
        sympy_bounds = [(x_sym[i], 0, ub_integration[i]) for i in range(d)]
        sol = float(integrate(expr, *sympy_bounds).evalf())
        return (sol, 0.)


# ----------------------------------- GENZ OSCILLATORY ----------------------------------- #
class GenzOscillatory(GenzProblem):
    def __init__(self, d, bound=1, random_params=False, key=jax.random.PRNGKey(2023),):
        # init params and f
        if random_params:
            u = jax.random.uniform(key, (d,), minval=0.0, maxval=1.0)
            a = jax.random.uniform(key, (d,), minval=0.0, maxval=10.0)
        else:
            u = jnp.repeat(jnp.array([0.5]), d)
            a = jnp.repeat(jnp.array([5.0]), d)

        # init super object
        params = {"a": a, "u": u }
        bounds = jnp.tile(jnp.array([[0], [bound]]), d)
        super().__init__(d, bounds, params)

        # init f
        self.f = self.init_f()

    def init_f(self):
        a = self.params["a"]
        u = self.params["u"]

        def f(x):
            return jnp.cos(2 * jnp.pi * u[0] + jnp.sum(a * x))
    
        return f
    
    def symbolic(self):
        a = self.params["a"]
        u = self.params["u"]
        d = self.bounds.shape[-1]

        # create symbolic expression
        symbol_str = " ".join([f"x{i}" for i in range(d)])
        x_sym = symbols(symbol_str)
        expr = cos(2 * jnp.pi * u[0] + sum([a[i] * x_sym[i] for i in range(d)]))

        # integrate
        ub = self.bounds[1, 1].item()
        sympy_bounds = [(x_sym[i], 0, ub) for i in range(d)]
        sol = float(integrate(expr, *sympy_bounds).evalf())
        return (sol, 0.)
    

# ----------------------------------- GENZ CORNER PEAK ----------------------------------- #
class GenzCorner(GenzProblem):
    def __init__(self, d, bound=1, random_params=False, key=jax.random.PRNGKey(2023),):
        # init params and f
        if random_params:
            a = jax.random.uniform(key, (d,), minval=0.0, maxval=10.0)
        else:
            a = jnp.repeat(jnp.array([5.0]), d)

        # init super object
        params = {"a": a}
        bounds = jnp.tile(jnp.array([[0], [bound]]), d)
        super().__init__(d, bounds, params)

        # init f
        self.f = self.init_f()

    def init_f(self):
        a = self.params["a"]

        def f(x):
            return (1 + jnp.sum(a * x))**(-self.d - 1)
    
        return f
    
    def symbolic(self):
        a = self.params["a"]
        d = self.bounds.shape[-1]

        # create symbolic expression
        symbol_str = " ".join([f"x{i}" for i in range(d)])
        x_sym = symbols(symbol_str)
        expr = (1 + sum([a[i] * x_sym[i] for i in range(d)]))**(-d - 1)

        # integrate
        ub = self.bounds[1, 1].item()
        sympy_bounds = [(x_sym[i], 0, ub) for i in range(d)]
        sol = float(integrate(expr, *sympy_bounds).evalf())
        return (sol, 0.)


# ------------------------------------- GENZ GAUSSIAN ------------------------------------ #
class GenzGaussian(GenzProblem):
    def __init__(self, d, random_params=False, key=jax.random.PRNGKey(2023),):
        # init params and f
        if random_params:
            u = jax.random.uniform(key, (d,), minval=0.0, maxval=1.0)
            a = jax.random.uniform(key, (d,), minval=0.0, maxval=10.0)
        else:
            u = jnp.repeat(jnp.array([0.5]), d)
            a = jnp.repeat(jnp.array([5.0]), d)

        # init super object
        params = {"a": a, "u": u }
        bounds = jnp.tile(jnp.array([[0], [1]]), d)
        super().__init__(d, bounds, params)

        # init f
        self.f = self.init_f()

    def init_f(self):
        a = self.params["a"]
        u = self.params["u"]

        def f(x):
            return jnp.where((x <= u).all(), jnp.exp(jnp.sum(a * x)), 0.0)
    
        return f

    def solution(self, **kwargs):
        quadf = lambda *x: self.f(jnp.array(x))
        dim_bounds = [self.bounds[:, i] for i in range(self.d)]
        return nquad(quadf, dim_bounds, **dict(kwargs))


# ------------------------------------- GENZ PRODUCT ------------------------------------- #
class GenzProduct(GenzProblem):
    def __init__(self, d, bound=1, random_params=False, key=jax.random.PRNGKey(2023),):
        if random_params:
            u = jax.random.uniform(key, (d,), minval=0.0, maxval=1.0)
            a = jax.random.uniform(key, (d,), minval=0.0, maxval=10.0)
        else:
            u = jnp.repeat(jnp.array([0.5]), d)
            a = jnp.repeat(jnp.array([5.0]), d)

        # init super object
        params = {"a": a, "u": u }
        bounds = jnp.tile(jnp.array([[0], [bound]]), d)
        super().__init__(d, bounds, params)

        # init f
        self.f = self.init_f()

    def init_f(self):
        a = self.params["a"]
        u = self.params["u"]
        d = self.bounds.shape[-1]

        def f(x):
            return jnp.prod(1 / (a**(-2) + (x - u)**2))
    
        return f
    
    def symbolic(self):
        a = self.params["a"].tolist()
        u = self.params["u"].tolist()
        d = self.bounds.shape[-1]

        # create symbolic expression
        symbol_str = " ".join([f"x{i}" for i in range(d)])
        x_sym = symbols(symbol_str)
        expr = math.prod([
            (1 / (a[i]**(-2) + (x_sym[i] - u[i])**2)) for i in range(d)
        ])

        # integrate
        ub = self.bounds[1, 1].item()
        sympy_bounds = [(x_sym[i], 0, ub) for i in range(d)]
        return float(integrate(expr, *sympy_bounds).evalf())
