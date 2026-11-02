# ---------------------------------------------------------------------------------------- #
#                                MODEL CONSTRUCTOR UTILTIES                                #
# ---------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from tinygp.kernels import ExpSquared as RBF
from tensorflow_probability.substrates.jax import distributions as tfd

from ffbq.gp.gp import GP, LowRankGP, SVGP
from ffbq.gp.training import fit_lrgp, fitgp
from ffbq.gp.kernels import RFF, M32,Periodic
from ffbq.gp.transforms import ARD, Transform
# from steinRF.gp.training import fitgp, train_with_restarts

# from steinRF.stein.targets import NLLTarget, PriorNLLTarget, TFTarget
# from steinRF.stein.srfr import srfr
# from steinRF.stein.sm_srfr import sm_srfr
# from steinRF.stein.mar_srfr import mar_srfr


__all__ = [
    "build_gp",
    "build_train_gp",
    "build_rff_gp",
    "build_train_rff_gp",
    "build_lrgp",
    "build_train_lrgp",
    "build_svgp",
    "build_train_svgp",
    "build_train_lrrbf"
]


# ------------------------------------- FULL RANK GP ------------------------------------- #
def build_gp(key, X_tr, diag, mean=None, init_ls=True, kernel="rbf"):
    d = X_tr.shape[-1]
    if init_ls:
        dX = X_tr[:, None, :] - X_tr[None, :, :]
        ls_init = jnp.median(dX**2, axis=(0, 1))
    else:
        ls_init = jnp.ones(d)

    # kernel
    if kernel == "rbf":
        k = RBF()
    elif kernel == "m32":
        k = M32()
    else:
        raise ValueError("Invalid kernel type.")
    k = Transform(ARD(ls_init), k)

    # kernel and gp initialization
    gp_pre = GP(k, X_tr, diag=diag, mean=mean)
    return gp_pre


def build_train_gp(key, X_tr, y_tr, diag, epochs, lr, kernel="rbf", **kwargs):
    # extract kwargs
    to_train = kwargs.pop(
        "to_train", lambda t: [t.kernel.transform.scale, t.diag]
    )
    if kernel == "m32":
        to_train = lambda t: [t.kernel.transform.scale]

    mean = kwargs.pop("mean", None)
    init_ls = kwargs.pop("init_ls", True)
    key, subkey = jax.random.split(key)

    def _train(subkey):
        gp_pre = build_gp(
            subkey, X_tr, diag, mean=mean, init_ls=init_ls, kernel=kernel
        )

        # Train the model
        gp, gp_losses = fitgp(
            gp_pre, y_tr, epochs, 
            to_train=to_train, lr=lr, **kwargs
        )

        return gp, gp_losses

    gp, gp_losses = _train(subkey)

    return gp, gp_losses


# ---------------------------------------- RFF GP ---------------------------------------- #
def build_rff_gp(key, X_tr, R, diag, mean=None, init_ls=True, kernel="rff"):
    d = X_tr.shape[-1]
    if init_ls:
        dX = X_tr[:, None, :] - X_tr[None, :, :]
        ls_init = jnp.median(dX**2, axis=(0, 1))
    else:
        ls_init = jnp.ones(d)

    if kernel == "rbf" or kernel =="rff":
        k = RFF(key, d=d, R=R, dist="normal")
    elif kernel == "m32":
        k = RFF(key, d=d, R=R, dist="m32")
    else:
        raise ValueError("Invalid kernel type.")
    k = Transform(ARD(ls_init), k)

    # kernel and gp initialization
    gp_pre = GP(k, X_tr, diag=diag, mean=mean)
    return gp_pre


def build_train_rff_gp(key, X_tr, y_tr, R, diag, epochs, lr, kernel="rff", **kwargs):
    if kernel == "rbf":
        to_train = kwargs.pop(
            "to_train", lambda t: [t.kernel.transform.scale, t.diag]
        )
    elif kernel == "m32":
        to_train = kwargs.pop(
            "to_train", lambda t: [t.kernel.transform.scale, t.diag]
        )
    elif kernel == "rff":
        to_train = kwargs.pop(
            "to_train", lambda t: [t.kernel.kernel.w, t.kernel.transform.scale, t.diag]
        )
    else:
        raise ValueError("Invalid kernel type.")

    # extract kwargs
    mean = kwargs.pop("mean", None)
    init_ls = kwargs.pop("init_ls", True)
    key, subkey = jax.random.split(key)

    def _train(subkey):
        gp_pre = build_rff_gp(subkey, X_tr, R, diag, mean=mean, init_ls=init_ls, kernel=kernel)

        # Train the model
        gp, gp_losses = fit_lrgp(
            gp_pre, y_tr, epochs, 
            to_train=to_train, lr=lr, **kwargs
        )

        return gp, gp_losses

    gp, gp_losses = _train(subkey)
    return gp, gp_losses


# --------------------------------------- LR RFF GP -------------------------------------- #
def build_lrgp(key, X_tr, R, diag, mean=None, w_init=None, from_data=False, init_ls=True):
    d = X_tr.shape[-1]
    if init_ls:
        dX = X_tr[:, None, :] - X_tr[None, :, :]
        ls_init = jnp.median(dX**2, axis=(0, 1))
    else:
        ls_init = jnp.ones(d)

    # Initialize model with current hyperparameters
    k = RFF(key, d=d, R=R)
    if from_data:
        k = k.initialize_from_data(key, R, X_tr)
    k = Transform(ARD(ls_init), k)
    
    # kernel and gp initialization
    if w_init is not None:
        k = eqx.tree_at(lambda t: t.kernel.kernel.w, k, w_init)
    gp_pre = LowRankGP(k, X_tr, diag=diag, mean=mean)
    return gp_pre


def build_train_lrgp(key, X_tr, y_tr, R, diag, epochs, lr, **kwargs):
    # extract kwargs
    to_train = kwargs.pop(
        "to_train", lambda t: [t.kernel.kernel.w, t.kernel.transform.scale, t.diag]
    )
    w_init = kwargs.pop("w_init", None)
    mean = kwargs.pop("mean", None)
    from_data = kwargs.pop("from_data", False)
    init_ls = kwargs.pop("init_ls", True)
    key, subkey = jax.random.split(key)

    def _train(subkey):
        gp_pre = build_lrgp(
            subkey, X_tr, R, diag, mean=mean, w_init=w_init, from_data=from_data, 
            init_ls=init_ls
        )

        # Train the model
        gp, gp_losses = fit_lrgp(
            gp_pre, y_tr, epochs, 
            to_train=to_train, lr=lr, **kwargs
        )

        return gp, gp_losses

    # restarts = kwargs.pop("restarts", 1)
    # best_gp, best_loss = train_with_restarts(key, _train, restarts)
    gp, gp_losses = _train(subkey)

    return gp, gp_losses


# ----------------------------------------- SVGP ----------------------------------------- #
def build_svgp(key, X_tr, R, diag, mean=None, init_ls=True, kernel="rbf"):
    d = X_tr.shape[-1]
    if init_ls:
        dX = X_tr[:, None, :] - X_tr[None, :, :]
        ls_init = jnp.median(dX**2, axis=(0, 1))
    else:
        ls_init = jnp.ones(d)

    if kernel == "rbf":
        k = RBF()
    elif kernel == "periodic":
        k = Periodic(gamma=1.)
    elif kernel == "rff":
        k = RFF(key, d=d, R=R)
    else:
        raise ValueError("Invalid kernel type.")

    # Initialize model with current hyperparameters
    k = Transform(ARD(ls_init), k)
    gp_pre = SVGP(k, X_tr, R, diag=diag, mean=mean, key=key)

    return gp_pre


def build_train_svgp(key, X_tr, y_tr, R, diag, epochs, lr, kernel="rbf", **kwargs):
    # extract kwargs
    to_train = kwargs.pop(
        "to_train", lambda t: []
    )
    mean = kwargs.pop("mean", None)
    init_ls = kwargs.pop("init_ls", True)
    key, subkey = jax.random.split(key)

    # trainable parameters
    if kernel == "rff":
        to_train = kwargs.pop(
            "to_train", lambda t: [t.m, t.kernel.kernel.w, t.kernel.transform.scale, t.diag]
        )
    elif kernel == "periodic":
        to_train = kwargs.pop(
            "to_train", lambda t: [t.m, t.kernel.transform.scale, t.diag, t.kernel.kernel.gamma]
        )
    else:
        to_train = kwargs.pop(
            "to_train", lambda t: [t.m, t.kernel.transform.scale, t.diag]
        )

    def _train(subkey):
        gp_pre = build_svgp(
            subkey, X_tr, R, diag, mean=mean, init_ls=init_ls, kernel=kernel
        )

        # Train the model
        gp, gp_losses = fitgp(
            gp_pre, y_tr, epochs, 
            to_train=to_train, lr=lr, **kwargs
        )

        return gp, gp_losses

    # restarts = kwargs.pop("restarts", 1)
    # best_gp, best_loss = train_with_restarts(key, _train, restarts)
    gp, gp_losses = _train(subkey)

    return gp, gp_losses


# ------------------------------------- LOW RANK RBF ------------------------------------- #
def build_train_lrrbf(key, X_tr, y_tr, R, diag, epochs, lr, **kwargs):
    # extract kwargs
    to_train = kwargs.pop(
        "to_train", lambda t: [t.kernel.transform.scale, t.diag]
    )
    w_init = kwargs.pop("w_init", None)
    mean = kwargs.pop("mean", None)
    from_data = kwargs.pop("from_data", False)
    init_ls = kwargs.pop("init_ls", True)
    key, subkey = jax.random.split(key)

    def _train(subkey):
        gp_pre = build_lrgp(
            subkey, X_tr, R, diag, mean=mean, w_init=w_init, from_data=from_data, 
            init_ls=init_ls
        )

        # Train the model
        gp, gp_losses = fit_lrgp(
            gp_pre, y_tr, epochs, 
            to_train=to_train, lr=lr, **kwargs
        )

        return gp, gp_losses

    # restarts = kwargs.pop("restarts", 1)
    # best_gp, best_loss = train_with_restarts(key, _train, restarts)
    gp, gp_losses = _train(subkey)

    return gp, gp_losses