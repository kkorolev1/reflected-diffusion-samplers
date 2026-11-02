# ---------------------------------------------------------------------------------------- #
#                                     UTILITY FUNCTIONS                                    #
# ---------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
from jax import vmap, jit
from scipy.interpolate import interpn
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle
import math



# ----------------------------------- TRAINING AND EVAL ---------------------------------- #
def mse(y_true, y_pred, scaler=None):
    if scaler is not None:
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(-1)
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
    
    return jnp.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred, scaler=None):
    if scaler is not None:
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(-1)
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
    
    return jnp.mean(jnp.abs(y_true - y_pred))


def rele(y_true, y_pred, scaler=None):
    if scaler is not None:
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(-1)
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
    
    return jnp.mean(jnp.abs(y_true - y_pred) / jnp.abs(y_true))


def calibration(y_true, mu_pred, sd_pred, scaler=None):
    lb, ub = mu_pred - 1.95 * sd_pred, mu_pred + 1.95 * sd_pred

    if scaler is not None:
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(-1)
        mu_pred = scaler.inverse_transform(mu_pred.reshape(-1, 1)).reshape(-1)
        lb = scaler.inverse_transform(lb.reshape(-1, 1)).reshape(-1)
        ub = scaler.inverse_transform(ub.reshape(-1, 1)).reshape(-1)

    return jnp.mean((y_true >= lb) * (y_true <= ub))


def z_score(y_true, mu_pred, sd_pred, scaler=None):
    return jnp.mean((y_true - mu_pred) / sd_pred)


def metric_bq(y_true, mu_pred, var_pred):
    sd_pred = jnp.sqrt(jnp.maximum(var_pred, 1e-10))  # clipping negative variances
    mse_val = float(y_true - mu_pred)**2
    mae_val = float(jnp.abs(y_true - mu_pred))
    rele_val = float(mae_val / jnp.abs(y_true))
    cal_val = float(calibration(y_true, mu_pred, sd_pred))
    z_val = float(z_score(y_true, mu_pred, sd_pred))

    results_dict = {
        'mse': mse_val, 'mae': mae_val, 'rel': rele_val, 
        'cal': cal_val, 'z_score': z_val
    }
    return results_dict


def train_test(key, X, y, test_size=0.2, shuffle=True):
    if shuffle:
        seed = int(key[0])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
    
    X_train, X_test, y_train, y_test = map(jnp.array, [X_train, X_test, y_train, y_test])
    return X_train, X_test, y_train, y_test


def log_results(filename, res):
    # Check if file exists
    if Path(filename).is_file():
        # Load existing data
        with open(filename, 'rb') as file:
            results = pickle.load(file)
    else:
        results = []

    # Update the data (append new results)
    results.append(res)

    # Save back to JSON
    with open(filename, 'wb') as file:
        pickle.dump(results, file)



# ----------------------------------- KERNEL UTILITIES ----------------------------------- #
@jit
def pairwise_distance(X):
    X1X1 = jnp.dot(X.T, X)
    x1_norm = jnp.diag(X1X1)
    dists = jnp.sqrt(jnp.maximum(x1_norm - 2 * X1X1 + x1_norm[:, None], 0))
    return dists


def stabilize(K, alpha=None):
    # Perform eigenvalue decomposition
    eigvals, eigvecs = jnp.linalg.eigh(K)
    
    if alpha is None:
        alpha = eigvals.max() * 0.001

    # Adjust eigenvalues below the threshold alpha
    adjusted_eigvals = jnp.maximum(eigvals, alpha)
    
    # Reconstruct the covariance matrix
    K_adjusted = eigvecs @ jnp.diag(adjusted_eigvals) @ eigvecs.T
    
    return K_adjusted, alpha


# -------------------------------- DATA AND GRID UTILITIES ------------------------------- #
def area(bounds):
    return jnp.prod(jnp.abs(bounds[1, :] - bounds[0, :]))


def shift_axis(ax):
    shift_val = (ax[1:] - ax[:-1]).mean()
    return ax - shift_val / 2


def center_signal(signal, bounds=None):
    if bounds is None:
        return signal - signal.mean()
    else:
        return signal - signal.mean() * area(bounds)


def grid_inds(bounds, sr=None, N=None, center=False):
    """
    Create a grid of points for signal processeing and FFTs.
    """
    d = bounds.shape[-1]
    if N is not None:
        if isinstance(N, int):
            N = [N] * d
        axes = [
            jnp.linspace(bounds[0, i], bounds[1, i], N[i])
            for i in range(d)
        ]
    elif sr is not None:
        dim_N = jnp.abs(bounds[1, :] - bounds[0,:]) * sr
        axes = [
            jnp.linspace(bounds[0, i], bounds[1, i], int(dim_N[i]))
            for i in range(d)
        ]
    else:
        raise ValueError("Either sr or N must be provided.")

    if center:
        dimN = jnp.array([len(ax) for ax in axes])
        axes = [
            shift_axis(axes[i]) if dimN[i] % 2 == 0 else axes[i] 
            for i in range(len(axes))
        ]

    return axes


def grid(bounds, sr=None, N=None, center=False, flatten=True, axes=False):
    """
    Create a grid of points for signal processeing and FFTs.
    """
    _axes = grid_inds(bounds, sr=sr, N=N, center=center)

    _grid = jnp.meshgrid(*_axes)
    _grid = jnp.stack(_grid, axis=-1)
    if flatten:
        _grid = _grid.reshape(-1, _grid.shape[-1])

    if axes:
        return _grid, jnp.array(_axes)
    return _grid


def regrid(data, d):  # assumes that the resulting grid is hypercubic
    assert data.ndim == 2 or data.ndim == 1
    
    N = data.shape[0]
    Nd = round(N**(1/d))
    axes_Nd = tuple([Nd for _ in range(d)])

    if data.ndim == 1:
        data = data.reshape(*axes_Nd)
    else:  # given assert above this means data.ndim == 2
        if d == 1:
            data = data.reshape(*axes_Nd)
        else:
            data = data.reshape(*axes_Nd, d)
    return data


def sr_to_N(sr, bounds):
    axes = grid_inds(bounds, sr=sr)
    N = tuple([len(ax) for ax in axes])
    return N


def smart_grid_inputs(X):  # assumes that the bounds are the same for all dims
    """Process whether a grid is needed or not for multi-d inputs.
    Regrid if necessary.

    Args:
        X (jnp.ndarray): (n * d) grid or flat points

    Returns:
        jnp.ndarray: n1 x n2 x ... x nd x d grid of points
    """

    if X.ndim == 1:
        raise ValueError("X must be at least 2D")

    elif X.ndim > 1:
        # if properly gridded then len(X.shape[:-1]) == X.shape[-1]
        if len(X.shape[:-1]) == X.shape[-1]:
            X_grid = X

        else:
            X_grid = regrid(X, X.shape[-1])

    return X_grid


def bounds_from_X(X):
    # get bounds
    bounds = jnp.stack([
        jnp.amin(X, axis=tuple(range(X.ndim - 1))),
        jnp.amax(X, axis=tuple(range(X.ndim - 1)))
    ], axis=0)

    return bounds


def fftpad(gridY, center=False):
    """Pad a matrix with trailing (N_d - 1, ...) zeros.
    Center instead pads on both sides

    Args:
        Y (jnp.ndarray): Uniformly spaced gridded matrix to pad
        leading (tuple): Leading dimensions to pad
        trailing (tuple): Trailing dimensions to pad

    Returns:
        jnp.ndarray: Padded matrix
    """
    # gridY = smart_grid(Y)
    if center:
        grid_shape = jnp.array(gridY.shape)
        pad_grid_shape = 2 * grid_shape - 1
        next_pow_2 = jnp.ceil(jnp.log2(pad_grid_shape)).astype(int)
        # ps = jnp.asarray(gridY.shape) - 1
        ps = 2 ** next_pow_2 - grid_shape
        pad_vals = [(p // 2, p - p // 2) for p in ps]
        return jnp.pad(gridY, pad_vals)
    else:
        return jnp.pad(gridY, [(0, nd - 1) for nd in gridY.shape])


def unpad(Xpad, newshape):
    currshape = jnp.asarray(Xpad.shape)
    newshape = jnp.asarray(newshape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = tuple(slice(startind[i], endind[i]) for i in range(len(startind)))
    return Xpad[myslice]


# ------------------------------------- FFT UTILITIES ------------------------------------ #
def outer_nd(vecs):
    """Outer product for n-vectors."""
    d = vecs.shape[0]
    einsum_str = ','.join([chr(97 + i) for i in range(d)]) + '->' + \
        ''.join([chr(97 + i) for i in range(d)])

    return jnp.einsum(einsum_str, *vecs)


def fftn_sep(ts):
    """Separable FFTN for multi-dimensional data"""
    # Fts = jax.vmap(jnp.fft.fftn)(ts)
    Fts = jnp.fft.fft(ts)
    Ft = outer_nd(Fts)
    return Ft


def fftfreqn(dims: tuple, delta, shift=True, flatten=True, axes=False):
    # get frequencies for each dim
    freqs = [
        jnp.fft.fftfreq(n_dim, delta[d]) for d, n_dim in enumerate(dims)
    ]
    if shift:
        freqs = [jnp.fft.fftshift(freq) for freq in freqs]

    # get meshgrid
    freq_grid = jnp.stack(jnp.meshgrid(*freqs, indexing='ij'), axis=-1)
    if flatten:
        freq_grid = freq_grid.reshape(-1, freq_grid.shape[-1])

    if axes:
        return freq_grid, freqs
    
    return freq_grid


def X_to_freq(X, sr, shift=True, flatten=True):
    X_grid = smart_grid_inputs(X)
    
    # get frequencies
    grid_dims = X_grid.shape[:-1]
    freqs = fftfreqn(grid_dims, sr, shift=shift, flatten=flatten)
    freqs = freqs
    return freqs


def scale_conv(conv, bounds, inverse=False, bounded=True):
    """Scale convolution to be consistent with the original data

    Args:
        X_orig (jnp.ndarray): Original data
        conv (jnp.ndarray): Convolution grid
        inverse (bool, optional): whether to scale original data to convolultion size. 
            Defaults to False.
        single (bool, optional): Whether conv is a convolution or single FFT. 
            Defaults to False.

    Returns:
        jnp.ndarray: _description_
    """

    area_X = area(bounds)
    dimsN = jnp.array(conv.shape) - 1
    d = dimsN.shape[0]

    # create scaling factor
    scaling_factor = area_X / jnp.prod(dimsN)

    if inverse:
        scaling_factor = 1 / scaling_factor

    # scale convolution
    conv_scaled = conv * scaling_factor

    return conv_scaled


def fft_interp(X, conv, bounds, sr, center=True):
    Xg_ax = grid(bounds, sr, axes_only=True, centered=center)

    X_interp = interpn(
        Xg_ax, conv, X, 
        method="pchip", bounds_error=False, fill_value=None
    )
    return X_interp


def fftn_1d(X, s=None):
    """N-dimensional FFT as a for loop of 1D FFTs"""
    d = X.ndim
    n = jnp.array(X.shape)

    # padding - centered padding
    if s is not None:
        if isinstance(s, int):
            s = jnp.asarray([s] * d)
        elif isinstance(s, tuple):
            s = jnp.asarray(s)

        pad_vals = s - n
        pad_vals = [(p // 2, p - p // 2) for p in pad_vals]
        X = jnp.pad(X, [(p1, p2) for p1, p2 in pad_vals])

    # loop through dimensions
    for i in range(d):
        X = jnp.fft.fft(X, axis=i)
    return X


def ifftn_1d(X):
    """N-dimensional FFT as a for loop of 1D FFTs"""
    d = X.ndim

    for i in range(d):
        X = jnp.fft.ifft(X, axis=i)
    return X


def energy_ratio(signal, edge_pct=0.5):
    """Calculate the amount of energy in a signal on edges vs on
    the interior. This is useful for determining the amount of padding.
    """
    
    # calculate bound_pct to multi_d bound pct
    ds = jnp.asarray(signal.shape)
    inds = ((1 - edge_pct) * signal.size)**(1/len(ds))
    inds = ((ds - inds) / 2).tolist()
    inds = [(int(jnp.floor(ind)), int(jnp.ceil(ind))) for ind in inds]
    interior_slices = tuple(slice(lb, -ub) for lb, ub in inds)
    interior_mask = jnp.zeros(signal.shape, dtype=bool)
    interior_mask = interior_mask.at[interior_slices].set(True)

    # get energy
    ext_energy = (signal[~interior_mask]**2).sum()
    cent_energy = (signal[interior_mask]**2).sum()

    return ext_energy / cent_energy


def smart_conv(h, g, pad=True, pow2=True, energy_bound=0.5):
    """Convolve with auto-determination of padding amounts to ensure good frequency"""
    grid_shape = h.shape
    shape_arr = jnp.asarray(grid_shape)

    # calculate energies
    energy = jnp.maximum(energy_ratio(h, energy_bound), energy_ratio(g, energy_bound))
    
    # calculate padding
    if pad:
        if energy > 0.5:
            n_pad = shape_arr * 4 - 1
        elif 0.1 < energy < 0.5 :
            n_pad = shape_arr * 2 - 1 - shape_arr
        else:
            n_pad = jnp.zeros_like(shape_arr)
        n_total = shape_arr + n_pad

        if pow2:
            next_pow_2 = jnp.ceil(jnp.log2(n_total)).astype(int)
            pow2_pad = 2 ** next_pow_2 - n_total
            n_total = 2 ** next_pow_2
    else:
        n_total = shape_arr
    n_total = tuple(n_total.astype(int).tolist())

    # convolve
    Fh = fftn_1d(h, s=n_total)
    Fg = fftn_1d(g, s=n_total)
    Fconv = Fh * Fg
    conv = jnp.fft.fftshift(ifftn_1d(Fconv))

    # shift and unpad
    if pad:
        conv = unpad(conv, grid_shape)
    
    return conv
