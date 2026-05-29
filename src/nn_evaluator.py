#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os, sys, shutil, joblib

import torch

from surrogate.cnn import CNNRegressor
from utils.grid_interp import Grid

# Add the config path to the directory
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from config import SPECFEM2D_WORKDIR, SPECFEM2D_DATA, SURR_PATH
from config import domain_xmax, domain_xmin, domain_zmax, domain_zmin


# =====================================================================
#            CACHED SURROGATE SINGLETON (model + scalers)
# =====================================================================

_cache = {
    "model": None,
    "scaler_X": None,
    "scaler_y": None,
    "device": None,
    "model_path": None,
    "scaler_x_path": None,
    "scaler_y_path": None,
}


def _get_device():
    """Resolve and cache the torch device once."""
    if _cache["device"] is None:
        _cache["device"] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return _cache["device"]


def _get_surrogate(
    model_path=os.path.join(SURR_PATH, "OUTPUT_FILES/files/best_model.pt"),
    scaler_x_path=os.path.join(SURR_PATH, "OUTPUT_FILES/files/input_scaler.pkl"),
    scaler_y_path=os.path.join(SURR_PATH, "OUTPUT_FILES/files/label_scaler.pkl"),
):
    """
    Return (model, scaler_X, scaler_y, device), loading from disk only
    on the first call or after reload_surrogate() has been invoked.
    """
    device = _get_device()

    # Load model if not cached (or after invalidation)
    if _cache["model"] is None or _cache["model_path"] != model_path:
        model = CNNRegressor().to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        _cache["model"] = model
        _cache["model_path"] = model_path

    # Load scalers if not cached (or after invalidation)
    if _cache["scaler_y"] is None or _cache["scaler_y_path"] != scaler_y_path:
        _cache["scaler_y"] = joblib.load(scaler_y_path)
        _cache["scaler_y_path"] = scaler_y_path

    if _cache["scaler_X"] is None or _cache["scaler_x_path"] != scaler_x_path:
        _cache["scaler_X"] = (
            joblib.load(scaler_x_path) if scaler_x_path is not None else None
        )
        _cache["scaler_x_path"] = scaler_x_path

    return _cache["model"], _cache["scaler_X"], _cache["scaler_y"], device


def reload_surrogate():
    """
    Invalidate the cached model and scalers so that the next call to
    nn_sims() will reload everything from disk.

    Call this AFTER fine-tuning the surrogate (i.e. after saving new
    weights to best_model.pt).
    """
    _cache["model"] = None
    _cache["scaler_X"] = None
    _cache["scaler_y"] = None


# =====================================================================
#                        FORWARD HELPER
# =====================================================================

def forward_obs_local_nn(workdir, n_task=1):
    data_dir = os.path.join(workdir, "DATA")

    par = os.path.join(data_dir, "Par_file")
    par_nn = os.path.join(data_dir, "Par_file_NN")

    if not os.path.isfile(par_nn):
        raise FileNotFoundError(f"Missing Par_file_NN in {data_dir}")

    # Always overwrite Par_file with NN version
    shutil.copy2(par_nn, par)

    # Run SPECFEM
    os.chdir(workdir)

    rc1 = os.system(f"mpirun -n {n_task} ./bin/xmeshfem2D")
    if rc1 != 0:
        raise RuntimeError(f"xmeshfem2D failed (exit code {rc1})")

    rc2 = os.system(f"mpirun -n {n_task} ./bin/xspecfem2D")
    if rc2 != 0:
        raise RuntimeError(f"xspecfem2D failed (exit code {rc2})")

    return None


# =====================================================================
#                         MAIN INFERENCE
# =====================================================================

def _to_original_misfit(y_scaled_np, scaler_y):
    """Invert: standardized → unstandardize → exp → original positive misfit."""
    y_log = scaler_y.inverse_transform(y_scaled_np)
    y_original = np.exp(y_log)
    return y_original


def nn_sims(
    workdir=SPECFEM2D_WORKDIR,
    model_path=os.path.join(SURR_PATH, "OUTPUT_FILES/files/best_model.pt"),
    scaler_x_path=os.path.join(SURR_PATH, "OUTPUT_FILES/files/input_scaler.pkl"),
    scaler_y_path=os.path.join(SURR_PATH, "OUTPUT_FILES/files/label_scaler.pkl"),
    device=None,
    n_task=1,
    run_forward=True,
    mc_samples=0,
    return_samples=False,
):
    """
    Surrogate inference with cached model loading.

    Parameters
    ----------
    run_forward : bool
        If False, skip SPECFEM and reuse existing proc000000_rho_vp_vs.dat.
    mc_samples : int
        If >0, perform MC-Dropout with this many stochastic forward passes.
    return_samples : bool
        If True and mc_samples>0, return (mu, sd, samples).

    Returns
    -------
    - if mc_samples == 0:  float  (deterministic unscaled prediction)
    - if mc_samples > 0 and return_samples==False:  float  (MCD mean)
    - if mc_samples > 0 and return_samples==True:   (mu, sd, samples)
    """

    # Run the forward step configured for NN input generation (optional)
    if run_forward:
        forward_obs_local_nn(workdir, n_task)

    # Read .dat file
    file = np.loadtxt(os.path.join(SPECFEM2D_DATA, "proc000000_rho_vp_vs.dat"))

    # Filter out PMLs (in meters)
    mask = (
        (file[:, 0] >= domain_xmin) & (file[:, 0] <= domain_xmax) &
        (file[:, 1] >= domain_zmin) & (file[:, 1] <= domain_zmax)
    )

    file_filtered = file[mask]

    # Interpolate to structured grid
    vp_2d = Grid(file_filtered[:, 0], file_filtered[:, 1], file_filtered[:, 3])

    # --- Get cached model and scalers (no disk I/O if already loaded) ---
    model, scaler_X, scaler_y, dev = _get_surrogate(
        model_path, scaler_x_path, scaler_y_path
    )

    # Override device if caller explicitly requested one
    if device is not None:
        dev = torch.device(device)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available.")

    # Validate / format input
    vp_2d = np.asarray(vp_2d, dtype=np.float32)
    if vp_2d.shape != (256, 256):
        raise ValueError(f"Expected vp_2d shape (256,256), got {vp_2d.shape}")

    # Apply input scaling (same as training)
    if scaler_X is not None:
        vp_scaled = scaler_X.transform(vp_2d.reshape(1, -1)).reshape(256, 256).astype(np.float32)
    else:
        vp_scaled = vp_2d

    # Input tensor
    x = torch.from_numpy(vp_scaled).to(dev).unsqueeze(0).unsqueeze(0)  # (1,1,256,256)

    # -----------------------------
    # Deterministic inference
    # -----------------------------
    if int(mc_samples) <= 0:
        model.eval()
        with torch.inference_mode():
            y_scaled = model(x)
            y_scaled_np = y_scaled.detach().cpu().numpy().reshape(1, -1)

        y_unscaled = _to_original_misfit(y_scaled_np, scaler_y)
        return float(y_unscaled.squeeze())

    # -----------------------------
    # MC Dropout inference
    # -----------------------------
    # MC Dropout inference (SOL)
    model.eval()
    model.enable_mc_dropout()

    samples = []

    with torch.inference_mode():
        for _ in range(int(mc_samples)):
            y_scaled = model(x)
            y_scaled_np = y_scaled.detach().cpu().numpy().reshape(1, -1)
            y_original = _to_original_misfit(y_scaled_np, scaler_y)
            samples.append(float(y_original.squeeze()))

    model.disable_mc_dropout()

    samples = np.asarray(samples, dtype=np.float64)
    mu = float(samples.mean())
    sd = float(samples.std(ddof=1)) if samples.size > 1 else 0.0

    if return_samples:
        return mu, sd, samples

    return mu