from __future__ import annotations

import math

import jax.numpy as jnp

from targets.cube import Cube


# Euclidean benchmark point used by pySecDec for f_66^A.
DEFAULT_S = -4.0 / 3.0
DEFAULT_T = -16.0 / 5.0
DEFAULT_P4_SQ = -100.0 / 39.0
DEFAULT_MASS_SQ = 1.0

# Published analytic value for f_66^A at the point above.
F66_A_REFERENCE = 0.247074199140732131068066

# Independent sector-decomposition + scrambled-Sobol references at the same
# Euclidean point.  The uncertainty fields below are deliberately conservative.
F70_A_REFERENCE = 0.822994
F70_A_REFERENCE_STD = 1.5e-5
A111011100_REFERENCE = 0.18890897
A111011100_REFERENCE_STD = 2.1e-7


# Family-A propagator routing.  Each denominator is represented as
# q_i^2 - m_i^2 with q_i = a_i k1 + b_i k2 + P_i and
# P_i represented in the (p1,p2,p3) basis.
_A7_LOOP = (
    (1.0, 0.0),  # d1: k1
    (1.0, 0.0),  # d2: k1+p1+p2
    (0.0, 1.0),  # d3: k2
    (0.0, 1.0),  # d4: k2+p1+p2
    (1.0, 0.0),  # d5: k1+p1
    (1.0, -1.0),  # d6: k1-k2
    (0.0, 1.0),  # d7: k2-p3
)
_A7_SHIFT = (
    (0.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
)
_A7_MASSIVE = (1, 1, 1, 1, 1, 0, 1)


def stick_breaking(z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map ``[0,1]^d`` to the ``d``-simplex and return ``log |J|``.

    For ``d=N-1`` cube coordinates,

        alpha_i = z_i prod_{j<i}(1-z_j),  i=1,...,N-1,
        alpha_N = prod_{j=1}^{N-1}(1-z_j),

    and ``|J| = prod_{j=1}^{N-2}(1-z_j)^(N-1-j)``.
    """

    z = jnp.asarray(z)
    single = z.ndim == 1
    zb = z[None, :] if single else z
    if zb.ndim != 2:
        raise ValueError(f"Expected shape (d,) or (batch,d), got {z.shape}.")

    one_minus = 1.0 - zb
    prefix = jnp.cumprod(
        jnp.concatenate([jnp.ones_like(zb[:, :1]), one_minus], axis=-1),
        axis=-1,
    )
    alpha = jnp.concatenate(
        [zb * prefix[:, :-1], prefix[:, -1:]],
        axis=-1,
    )

    d = zb.shape[-1]
    exponents = jnp.arange(d - 1, -1, -1, dtype=zb.dtype)
    log_terms = jnp.where(
        exponents[None, :] > 0,
        exponents[None, :] * jnp.log1p(-zb),
        jnp.zeros_like(zb),
    )
    log_jac = jnp.sum(log_terms, axis=-1)

    if single:
        return alpha[0], log_jac[0]
    return alpha, log_jac


def _external_gram(s: float, t: float, p4_sq: float, dtype) -> jnp.ndarray:
    u = p4_sq - s - t
    return jnp.asarray(
        [
            [0.0, 0.5 * s, 0.5 * t],
            [0.5 * s, 0.0, 0.5 * u],
            [0.5 * t, 0.5 * u, 0.0],
        ],
        dtype=dtype,
    )


def _symanzik_polynomials(
    alpha: jnp.ndarray,
    loop_coeffs: jnp.ndarray,
    shifts: jnp.ndarray,
    masses_sq: jnp.ndarray,
    gram: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the two-loop Symanzik polynomials U and F."""

    a = loop_coeffs[:, 0]
    b = loop_coeffs[:, 1]

    m11 = jnp.einsum("...i,i->...", alpha, a * a)
    m12 = jnp.einsum("...i,i->...", alpha, a * b)
    m22 = jnp.einsum("...i,i->...", alpha, b * b)
    U = m11 * m22 - m12 * m12

    v1 = jnp.einsum("...i,i,ij->...j", alpha, a, shifts)
    v2 = jnp.einsum("...i,i,ij->...j", alpha, b, shifts)
    v1_sq = jnp.einsum("...i,ij,...j->...", v1, gram, v1)
    v2_sq = jnp.einsum("...i,ij,...j->...", v2, gram, v2)
    v12 = jnp.einsum("...i,ij,...j->...", v1, gram, v2)

    shift_sq = jnp.einsum("ni,ij,nj->n", shifts, gram, shifts)
    C = jnp.einsum("...i,i->...", alpha, shift_sq - masses_sq)

    F = m22 * v1_sq - 2.0 * m12 * v12 + m11 * v2_sq - U * C
    return U, F


def _is_default_kinematics(s: float, t: float, p4_sq: float, mass_sq: float) -> bool:
    values = (s, t, p4_sq, mass_sq)
    defaults = (DEFAULT_S, DEFAULT_T, DEFAULT_P4_SQ, DEFAULT_MASS_SQ)
    return all(
        math.isclose(x, y, rel_tol=0.0, abs_tol=1e-14) for x, y in zip(values, defaults)
    )


def _validate_euclidean_point(s: float, t: float, p4_sq: float, mass_sq: float) -> None:
    if mass_sq <= 0.0:
        raise ValueError("mass_sq must be positive.")
    x1 = s / mass_sq
    x2 = p4_sq / mass_sq
    x3 = t / mass_sq
    if not (x3 < x2 < x1 < 0.0):
        raise ValueError(
            "This benchmark expects the Euclidean ordering "
            "t/m^2 < p4^2/m^2 < s/m^2 < 0."
        )


class _HiggsPlanarCube(Cube):
    active: tuple[int, ...] = ()
    integral_id: str = ""
    canonical_prefactor: str = "scalar"
    default_reference_Z: float | None = None
    default_reference_std: float = 0.0
    default_reference_kind: str = ""
    default_reference_source: str = ""

    def __init__(
        self,
        *,
        s: float = DEFAULT_S,
        t: float = DEFAULT_T,
        p4_sq: float = DEFAULT_P4_SQ,
        mass_sq: float = DEFAULT_MASS_SQ,
        reference_Z: float | None = None,
        corrector: float = 0.0,
    ) -> None:
        _validate_euclidean_point(s, t, p4_sq, mass_sq)

        self.s = float(s)
        self.t = float(t)
        self.p4_sq = float(p4_sq)
        self.mass_sq = float(mass_sq)
        self.u = self.p4_sq - self.s - self.t
        self.num_propagators = len(self.active)

        default_point = _is_default_kinematics(self.s, self.t, self.p4_sq, self.mass_sq)
        if reference_Z is None:
            if not default_point or self.default_reference_Z is None:
                raise ValueError(
                    "A non-default kinematic point requires an explicit reference_Z."
                )
            reference_Z = self.default_reference_Z
            self.reference_std = float(self.default_reference_std)
            self.reference_kind = self.default_reference_kind
            self.reference_source = self.default_reference_source
        else:
            if reference_Z <= 0.0:
                raise ValueError("reference_Z must be positive.")
            self.reference_std = 0.0
            self.reference_kind = "user_supplied"
            self.reference_source = "user supplied"

        self.reference_Z = float(reference_Z)

        active = self.active
        self._loop_coeffs = jnp.asarray(
            [_A7_LOOP[i] for i in active], dtype=jnp.float64
        )
        self._shifts = jnp.asarray([_A7_SHIFT[i] for i in active], dtype=jnp.float64)
        self._masses_sq = jnp.asarray(
            [self.mass_sq if _A7_MASSIVE[i] else 0.0 for i in active],
            dtype=jnp.float64,
        )

        super().__init__(
            dim=self.num_propagators - 1,
            log_prob_fn=self._log_integrand_batch,
            min_coord=0.0,
            max_coord=1.0,
            mask_outside=True,
            sample_fn=None,
            corrector=corrector,
            is_non_negative=True,
        )

    def _compute_true_logZ(self) -> float:
        return math.log(self.reference_Z)

    def _basis_log_prefactor(self) -> float:
        return 0.0

    def _log_integrand_batch(self, z: jnp.ndarray) -> jnp.ndarray:
        z = jnp.atleast_2d(jnp.asarray(z, dtype=jnp.float64))
        alpha, log_jac = stick_breaking(z)
        gram = _external_gram(self.s, self.t, self.p4_sq, z.dtype)
        U, F = _symanzik_polynomials(
            alpha,
            self._loop_coeffs,
            self._shifts,
            self._masses_sq,
            gram,
        )

        n = self.num_propagators
        u_power = n - 6
        f_power = n - 4
        tiny = jnp.finfo(z.dtype).tiny
        log_F = jnp.log(jnp.maximum(F, tiny))

        log_value = (
            math.lgamma(n - 4) + self._basis_log_prefactor() + log_jac - f_power * log_F
        )
        if u_power:
            log_value = log_value + u_power * jnp.log(jnp.maximum(U, tiny))

        # In the Euclidean region U,F are non-negative.  The strict checks below
        # only remove roundoff-invalid points; endpoint zeros are measure-zero.
        valid = (F >= 0.0) & jnp.isfinite(log_jac)
        if u_power:
            valid = valid & (U >= 0.0)
        return jnp.where(valid, log_value, -jnp.inf)


class HiggsA6Elliptic(_HiggsPlanarCube):
    """Five-dimensional cube target for Family-A I_{110111100}."""

    active = (0, 1, 3, 4, 5, 6)
    integral_id = "A:110111100"
    canonical_prefactor = "f66A"
    default_reference_Z = F66_A_REFERENCE
    default_reference_std = 0.0
    default_reference_kind = "published_analytic"
    default_reference_source = "Borowka et al., pySecDec, arXiv:1703.09692, Eq. (3)"

    def _basis_log_prefactor(self) -> float:
        x1 = self.s / self.mass_sq
        return 1.5 * math.log(-x1)


class HiggsA7Elliptic(_HiggsPlanarCube):
    """Six-dimensional cube target for the top Family-A elliptic sector."""

    active = (0, 1, 2, 3, 4, 5, 6)
    integral_id = "A:111111100"
    canonical_prefactor = "positive Euclidean f70A normalization"
    default_reference_Z = F70_A_REFERENCE
    default_reference_std = F70_A_REFERENCE_STD
    default_reference_kind = "sector_decomposition_qmc"
    default_reference_source = (
        "independent sector-decomposition + scrambled-Sobol evaluation at the "
        "pySecDec Euclidean benchmark point"
    )

    def _basis_log_prefactor(self) -> float:
        x1 = self.s / self.mass_sq
        x2 = self.p4_sq / self.mass_sq
        x3 = self.t / self.mass_sq
        radicand = 4.0 * (x2 - x1 - x3) + x1 * x3
        prefactor = abs(x1 * math.sqrt(-x3) * math.sqrt(4.0 - x1) * math.sqrt(radicand))
        return math.log(prefactor)


class HiggsA6PolylogControl(_HiggsPlanarCube):
    """Five-dimensional finite non-elliptic Family-A scalar I_{111011100}."""

    # d1,d2,d3,d5,d6,d7 active; d4 absent.
    active = (0, 1, 2, 4, 5, 6)
    integral_id = "A:111011100"
    canonical_prefactor = "none (scalar integral)"
    default_reference_Z = A111011100_REFERENCE
    default_reference_std = A111011100_REFERENCE_STD
    default_reference_kind = "sector_decomposition_qmc"
    default_reference_source = (
        "independent sector-decomposition + scrambled-Sobol evaluation at the "
        "pySecDec Euclidean benchmark point"
    )


if __name__ == "__main__":
    target = HiggsA6Elliptic()
    print(f"logZ: {target.log_Z}, Z: {target.Z}, dim: {target.dim}")
    import numpy as np

    # Estimate Z using simple Monte Carlo sampling
    def monte_carlo_estimate(target, n_samples=100000, seed=0):
        rng = np.random.default_rng(seed)
        dim = target.num_propagators - 1
        samples = rng.uniform(low=0.0, high=1.0, size=(n_samples, dim))
        vals = jnp.exp(target.log_prob(samples, constrained=True))
        Z_hat = vals.mean() if len(vals) > 0 else np.nan
        std_err = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
        return Z_hat, std_err

    Z_est, Z_err = monte_carlo_estimate(target, n_samples=int(1e6))
    print("Monte Carlo estimate of Z:", Z_est)
    print("Standard error:", Z_err)
    target.visualise(show=True)
