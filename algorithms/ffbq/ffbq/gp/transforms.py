# ---------------------------------------------------------------------------------------- #
#                                      DATA TRANSFORMS                                     #
# ---------------------------------------------------------------------------------------- #
from typing import Any
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float

import tinygp
from tinygp.helpers import JAXArray


class Transform(tinygp.kernels.base.Kernel):
    transform: eqx.Module
    kernel: tinygp.kernels.base.Kernel

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel.evaluate(self.transform(X1), self.transform(X2))
    
    @jax.jit
    def phi(self, X: JAXArray) -> JAXArray:
        X_transformed = jax.vmap(self.transform)(X)
        return self.kernel.phi(X_transformed)


class ARD(eqx.Module):
    scale: Float[Array, "d"]

    def __init__(self, scale):
        self.scale = jnp.log(scale)

    @property
    def _scale(self):
        return jnp.exp(self.scale)
    
    def __call__(self, X: JAXArray) -> JAXArray:
        return X / self._scale

