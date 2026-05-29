#!/usr/bin/env python
# coding: utf-8

# import libraries
import os, sys

# Separating Cubit path from PyTorch
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
sys.path = [p for p in sys.path if "Coreform-Cubit" not in p and "python3.10/site-packages" not in p]


import time, shutil
import numpy as np
from pathlib import Path

import torch
from surrogate.cnn import CNNRegressor
import joblib

from src.nn_evaluator import nn_sims, reload_surrogate
from utils.nn_cubit_mesher import MCMC_automatic_mesh
from utils.grid_interp import Grid

from config import SPECFEM2D_WORKDIR, SPECFEM2D_DATA, SURR_PATH, TARGET_DATA
from config import stdvs, parameter_min, parameter_max, domain_xmax, domain_xmin, domain_zmax, domain_zmin, n_shots, machine_type

from src.seismic.specfem2d import forward

# REGIME & CONTROLS
from config import regime, gamma, beta, K, R

import logging
logging.basicConfig(
    filename=os.path.join(SPECFEM2D_WORKDIR, "nn_mcmc_run.log"),
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M"
)



# Model function with parameters (NN surrogate)
def model_function(x_center, z_center, radius, alpha=0, proposed_vp=2000):

    # MESH GENERATION & P VELOCITY CHANGE
    major_axis = radius
    minor_axis = radius
    MCMC_automatic_mesh(x_center, z_center, major_axis, minor_axis, alpha, proposed_vp)

    # NN MISFIT PREDICTION CALL (deterministic)
    phi = nn_sims()
    return phi

# Model function with parameters
def set_par_file_exact(workdir):
    data_dir = os.path.join(workdir, "DATA")
    par_exact = os.path.join(data_dir, "Par_file_EXACT")
    par = os.path.join(data_dir, "Par_file")

    if not os.path.isfile(par_exact):
        raise FileNotFoundError("Missing Par_file_EXACT")

    shutil.copy2(par_exact, par)

def model_function_exact(x_center, z_center, radius, alpha=0, proposed_vp=2000):

    # MESH GENERATION & P VELOCITY CHANGE
    major_axis = radius
    minor_axis = radius
    MCMC_automatic_mesh(major_axis, minor_axis, x_center, z_center, alpha, proposed_vp)
    # FORWARD SIMULATIONS
    forward.sims(machine_type)
    return None

# misfit function
def misfit_exact(baseline_data, target, target_dir=TARGET_DATA):
    """
    Noise-free misfit using the exact forward solver.

    Convention matches sobol_data_gen.py exactly:
        chi = 0.5 * ||r||^2     (sigma_n = 1, positive)

    The noise scaling (sigma_n from data.noise.npz) is NOT applied here.
    Instead, it enters the MCMC acceptance ratio as T = sigma_n^2.
    """
    modelled = forward.data_1d(n_shots, SPECFEM2D_WORKDIR)
    pred_diff = modelled - baseline_data
    r = target - pred_diff

    chi = 0.5 * np.dot(r, r)        # noise-free positive misfit
    return chi

# Prior function: NORMAL
def prior(cov_matrix):
    return np.random.multivariate_normal(np.zeros(cov_matrix.shape[0]), cov_matrix)

# Fine-tune surrogate model
def finetune_surrogate(model_path=os.path.join(SURR_PATH, "OUTPUT_FILES/files/best_model.pt"),
                       scaler_x_path=os.path.join(SURR_PATH, "OUTPUT_FILES/files/input_scaler.pkl"),
                       scaler_y_path=os.path.join(SURR_PATH, "OUTPUT_FILES/files/label_scaler.pkl"),
                       lr=1e-5, epochs=5, batch_size=32, freeze_features=True):

    # --- FIX: prevent importing Cubit python packages (sympy) ---
    import sys, os
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["TORCHINDUCTOR_DISABLE"] = "1"
    sys.path = [p for p in sys.path
                if "Coreform-Cubit" not in p
                and "python3.10/site-packages" not in p]
    if "sympy" in sys.modules:
        del sys.modules["sympy"]
    # ------------------------------------------------------------

    logging.info("\t\t[FINETUNE] Starting fine-tuning")

    # Load dataset (same convention as train.py)
    inputs = np.load(os.path.join(SURR_PATH, "dataset/nn_surr/sobol_input_array.npz"))["input_array"][:, 1, :, :]
    labels = np.load(os.path.join(SURR_PATH, "dataset/nn_surr/sobol_label_misfit.npz"))["label_misfit"] * -1

    inputs = inputs.astype(np.float32)
    labels = labels.astype(np.float32).reshape(-1, 1)

    # Load scalers
    scaler_X = joblib.load(scaler_x_path) if scaler_x_path is not None else None
    scaler_y = joblib.load(scaler_y_path)
    
    # Scale X
    if scaler_X is not None:
        N = inputs.shape[0]
        inputs = scaler_X.transform(inputs.reshape(N, -1)).reshape(inputs.shape)

    # Scale y (IMPORTANT: do NOT refit scalers during fine-tuning)
    labels = scaler_y.transform(labels)

    # Torch tensors
    X = torch.from_numpy(inputs[:, None, :, :]).float()  # (N,1,H,W)
    y = torch.from_numpy(labels).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model weights
    model = CNNRegressor().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    # Freeze conv feature extractor (optional but recommended)
    if freeze_features:
        for p in model.features.parameters():
            p.requires_grad = False
        logging.info("\t\t[FINETUNE] Frozen feature extracting layers")

    # Optimizer on trainable params only
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = torch.nn.MSELoss()

    # DataLoader
    ds = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    # Fine-tune loop
    model.train()
    for ep in range(epochs):
        running = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            running += loss.item() * xb.size(0)

        ep_loss = running / len(loader.dataset)
        logging.info(f"\t\t[FINETUNE] Epoch {ep+1}/{epochs} loss = {ep_loss:.6e}")

    # Save updated weights back to same path
    torch.save(model.state_dict(), model_path)
    logging.info("\t\t[FINETUNE] Saved updated weights")
    return None


def append_training_example(vp_2d, exact_misfit_value):
    """
    Append new training point to dataset/nn_surr/*.npz

    IMPORTANT: train.py expects:
      - input_array with at least channel index 1 used as vp-field
      - label_misfit stored with opposite sign, because train.py multiplies by -1
    """

    in_path = os.path.join(SURR_PATH, "dataset/nn_surr/sobol_input_array.npz")
    y_path = os.path.join(SURR_PATH, "dataset/nn_surr/sobol_label_misfit.npz")

    inputs_npz = np.load(in_path)
    labels_npz = np.load(y_path)

    X = inputs_npz["input_array"]          # shape (N, C, 256, 256) (assumed)
    y = labels_npz["label_misfit"]         # shape (N,1) or (N,)

    # Create a new sample with same shape as X[0]
    x_new = np.zeros_like(X[0])
    # Put vp_2d in channel 1 (consistent with your training loader)
    x_new[1, :, :] = vp_2d.astype(np.float32)

    # Labels stored negative (because train.py multiplies by -1)
    y_new_val = -1.0 * float(exact_misfit_value)

    X_new = np.concatenate([X, x_new[None, ...]], axis=0)

    if y.ndim == 1:
        y_new = np.concatenate([y, np.array([y_new_val], dtype=y.dtype)], axis=0)
    else:
        y_new = np.concatenate([y, np.array([[y_new_val]], dtype=y.dtype)], axis=0)

    np.savez(in_path, input_array=X_new)
    np.savez(y_path, label_misfit=y_new)

    logging.info(f"[REFINE] Added training point. New dataset size: {X_new.shape[0]}")
    return None


# =====================================================================
#          SOBOL POINTER (geometry-aware, for adaptive regime)
# =====================================================================

class FreshSobolSequence:
    """
    Generates a fresh geometry-aware Sobol low-discrepancy sequence.
    Points are [x_center, z_center, radius] with x/z constrained so
    the circle stays inside the domain with the given margin.
    """
    def __init__(self, n_points, parameter_min, parameter_max,
                 domain_xmin, domain_xmax, domain_zmin, domain_zmax,
                 margin=100, seed=42):
        from scipy.stats import qmc

        self._pmin = np.asarray(parameter_min, dtype=float)
        self._pmax = np.asarray(parameter_max, dtype=float)
        self._dxmin = float(domain_xmin)
        self._dxmax = float(domain_xmax)
        self._dzmin = float(domain_zmin)
        self._dzmax = float(domain_zmax)
        self._margin = float(margin)

        n_param = len(self._pmin)
        sampler = qmc.Sobol(d=n_param, scramble=True, seed=seed)
        raw = sampler.random(n=n_points)  # (n_points, n_param) in [0,1]

        # Parameter-bound scaling + geometry rejection
        points = []
        for i in range(n_points):
            u = raw[i]

            # Scale ALL parameters from [0,1] to [parameter_min, parameter_max]
            X = self._pmin[0] + u[0] * (self._pmax[0] - self._pmin[0])
            Z = self._pmin[1] + u[1] * (self._pmax[1] - self._pmin[1])
            R = self._pmin[2] + u[2] * (self._pmax[2] - self._pmin[2])

            m = self._margin

            # Geometry check: circle must fit inside domain with margin
            if (X - R - m < self._dxmin) or \
               (X + R + m > self._dxmax) or \
               (Z - R - m < self._dzmin) or \
               (Z + R + m > self._dzmax):
                continue  # skip points that fail geometry

            points.append(np.round(np.array([X, Z, R]), 1))

        self._points = np.array(points)
        self._idx = 0
        self._n_points = n_points

        logging.info(
            f"[FreshSobolSequence] Generated {n_points} geometry-aware Sobol points"
        )

    def next(self):
        """Return the next Sobol point. Wraps around if exhausted."""
        if self._idx >= self._n_points:
            logging.info("[FreshSobolSequence] All points consumed — wrapping around")
            self._idx = 0
        point = self._points[self._idx]
        self._idx += 1
        return point

    @property
    def remaining(self):
        return self._n_points - self._idx


# Metropolis-Hastings algorithm
# ----------------------- NORMAL
def MH(baseline_data, target_data,
       initial_model, iterations,
       parameter_min=parameter_min, parameter_max=parameter_max,
       stdvs=np.array(stdvs), save=True, n=10, T=None):

    import numpy as np
    import os, time, logging

    initial_model = np.array(initial_model)
    init_cov_matrix = np.diag(stdvs**2)

    start = time.time()

    objective_function = []
    accepted_samples = []
    prop_params = [[] for _ in range(len(parameter_min))]

    k = 0
    q = 1

    parameter_min = np.array(parameter_min)
    parameter_max = np.array(parameter_max)

    for i, (lo, hi) in enumerate(zip(parameter_min, parameter_max)):
        assert lo <= hi, f"Bounds swapped at index {i}"

    current_params = np.array(initial_model)

    MaxLit = 5000000000

    # max refines per MCMC step (q)
    max_refines = R

    # ── Temperature = sigma_n^2 from the noise model ──────────────
    if T is None:
        meta_path = Path(TARGET_DATA) / "data.noise.npz"
        if not meta_path.exists():
            raise FileNotFoundError(f"Noise metadata not found: {meta_path}")
        _meta = np.load(meta_path)
        _sigma_n = float(_meta["sigma_n"])
        _sigma_n = _sigma_n if _sigma_n > 0.0 else 1.0
        T = _sigma_n ** 2
        logging.info(f"Temperature loaded from {meta_path}")
        logging.info(f"sigma_n: {_sigma_n:.6e}")
    else:
        logging.info(f"Temperature set manually")

    logging.info(f"REGIME: {regime}")
    logging.info(f"ITERATIONS: {iterations}")
    logging.info(f"TEMPERATURE T: {T:.6e}")

    # This is to make sure the circle is ALWAYS inside the domain
    margin = 100

    # ==========================================================
    # ======================== ADAPTIVE =========================
    # ==========================================================
    if regime == "adaptive":

        # refinement counters
        n_refine_random = 0
        n_refine_gamma = 0
        n_refine_total = 0

        # ----------------------------------------------------------
        # Tracking: variable-length lists (can grow as large as needed)
        # We store a record EVERY time we evaluate triggers (each while-pass),
        # including re-evaluations after fine-tuning.
        # ----------------------------------------------------------
        random_indicator = []        # stores u_ref draws
        occasional_indicator = []    # stores eps_ind values
        trigger_q = []               # stores MH iteration q for each record
        trigger_refines = []         # stores current "refines" count at time of record
        trigger_rand = []            # stores refine_random (0/1)
        trigger_occ = []             # stores refine_indicator (0/1)

        # Optional: per-q summary (fixed length, easy plotting)
        rand_trigger_q = np.zeros(iterations, dtype=np.int8)   # any random trigger happened in q
        occ_trigger_q  = np.zeros(iterations, dtype=np.int8)   # any indicator trigger happened in q
        n_refines_q    = np.zeros(iterations, dtype=np.int16)  # number of exact refines executed in q
        max_eps_q      = np.full(iterations, np.nan, dtype=float)

        # load Sobol dataset ONCE
        data_dir = os.path.join(SURR_PATH, "dataset/nn_surr")
        in_path = os.path.join(data_dir, "sobol_input_array.npz")
        y_path = os.path.join(data_dir, "sobol_label_misfit.npz")

        X_data = np.load(in_path)["input_array"]
        y_data = np.load(y_path)["label_misfit"]

        # ── Generate fresh Sobol sequence for random refinement (Eq. 19) ──
        sobol_ptr = FreshSobolSequence(
            n_points=2 * iterations,
            parameter_min=parameter_min,
            parameter_max=parameter_max,
            domain_xmin=domain_xmin,
            domain_xmax=domain_xmax,
            domain_zmin=domain_zmin,
            domain_zmax=domain_zmax,
            margin=margin,
            seed=42,
        )
        logging.info(
            f"[ADAPTIVE] Fresh Sobol sequence: {2 * iterations} points "
            f"for random refinement"
        )

        # initial NN evaluation (same as non-adaptive)
        current_likelihood = model_function(*current_params)
        _, _, current_phi_samples = nn_sims(mc_samples=K, return_samples=True)
        objective_function.append(current_likelihood)

        for lit in range(MaxLit):

            # ---------------- proposal ----------------
            if q < int(iterations * 0.05):
                proposal = current_params + prior(init_cov_matrix)
            elif int(iterations * 0.05) <= q < int(iterations * 0.5):
                eps = 1e-10
                n_unknown = init_cov_matrix.shape[0]
                sd_cov = (2.4**2) / n_unknown
                C_matrix = sd_cov * np.cov(np.array(accepted_samples).T) \
                           + sd_cov * eps * np.identity(n_unknown)
                proposal = current_params + prior(C_matrix)
            else:
                proposal = current_params + prior(C_matrix)

            # Check if proposal is within bounds (ALL parameters)
            if np.any(proposal < parameter_min) or np.any(proposal > parameter_max):
                continue

            # Check circle stays inside domain (with margin)
            if (proposal[0] - proposal[2] < domain_xmin + margin) or \
               (proposal[0] + proposal[2] > domain_xmax - margin) or \
               (proposal[1] - proposal[2] < domain_zmin + margin) or \
               (proposal[1] + proposal[2] > domain_zmax - margin):
                continue

            proposal = np.round(proposal, decimals=1)
            for i in range(len(init_cov_matrix)):
                prop_params[i].append(proposal[i])

            logging.getLogger().handlers[0].stream.write("\n")
            logging.info(f"ITERATION {q}")
            logging.info(f"X-LOC {proposal[0]}, Z-LOC {proposal[1]}, RADIUS {proposal[2]}")

            # ==================================================
            # Refinement loop (bounded) for THIS proposal and THIS q
            # ==================================================
            refines = 0
            phi_exact = None
            idx = q - 1  # 0-based index for per-q arrays

            # ──────────────────────────────────────────────────
            # CRASH-SAFETY: snapshot state BEFORE refinement
            # so we can roll back if exact_misfit crashes
            # ──────────────────────────────────────────────────
            exact_crashed = False

            # Back up Par_file so we can restore it on crash
            par_file_path = os.path.join(SPECFEM2D_WORKDIR, "DATA", "Par_file")
            par_file_backup = par_file_path + "._mh_backup"
            if os.path.isfile(par_file_path):
                shutil.copy2(par_file_path, par_file_backup)

            # Snapshot dataset sizes so we can roll back appended rows
            dataset_size_before = X_data.shape[0]

            # Snapshot tracking list lengths so we can trim on crash
            tracking_len_before = len(random_indicator)

            # Snapshot per-q summary values
            rand_trigger_q_before  = rand_trigger_q[idx]
            occ_trigger_q_before   = occ_trigger_q[idx]
            n_refines_q_before     = n_refines_q[idx]
            max_eps_q_before       = max_eps_q[idx]

            # Snapshot refinement counters
            n_refine_total_before  = n_refine_total
            n_refine_random_before = n_refine_random
            n_refine_gamma_before  = n_refine_gamma

            while True:

                # ---------------- surrogate evaluation ----------------
                try:
                    prop_likelihood = model_function(*proposal)
                    prop_mu, prop_sd, prop_phi_samples = nn_sims(
                        run_forward=False, mc_samples=K, return_samples=True
                    )
                except Exception as e:
                    logging.info(f"Skipping samples at iteration {q} due to NN pipeline crash:\n{str(e)}")
                    prop_phi_samples = None
                    break

                # ---- Compute K acceptance probabilities (Eq. 14-15) ----
                r_k = (-0.5 * prop_phi_samples + 0.5 * current_phi_samples) / T
                alphas = np.minimum(1.0, np.exp(r_k))
                eps_ind = float(alphas.max() - alphas.min())

                logging.info(f"CURRENT MISFIT: {current_likelihood}")
                logging.info(f"NEW MISFIT: {prop_mu}")
                logging.info(f"INDICATOR: {eps_ind}")

                # ---- refinement triggers (random + indicator) ----
                u_ref = np.random.rand()
                refine_random = (u_ref < beta)
                refine_indicator = (eps_ind >= gamma)

                logging.info(f"RANDOM REFINEMENT TRIGGERED: {refine_random}")
                logging.info(f"OCCASIONAL REFINEMENT TRIGGERED: {refine_indicator}")

                # ----------------------------------------------------------
                # Record values EVERY time triggers are evaluated (variable length)
                # ----------------------------------------------------------
                random_indicator.append(float(u_ref))
                occasional_indicator.append(float(eps_ind))
                trigger_q.append(int(q))
                trigger_refines.append(int(refines))
                trigger_rand.append(int(refine_random))
                trigger_occ.append(int(refine_indicator))

                # Optional per-q summaries
                if refine_random:
                    rand_trigger_q[idx] = 1
                if refine_indicator:
                    occ_trigger_q[idx] = 1
                if np.isnan(max_eps_q[idx]) or eps_ind > max_eps_q[idx]:
                    max_eps_q[idx] = eps_ind

                # If no refinement trigger -> exit refinement loop and do MH step
                if (refine_random == False) and (refine_indicator == False):
                    break

                # If we already refined too many times -> stop refining and proceed
                if refines >= max_refines:
                    logging.info(f"\t\t[REFINE] Max refines reached ({max_refines}) -> proceed with surrogate MH")
                    break

                # ==========================
                # Refinement step (exact)
                # ==========================
                refines += 1
                n_refines_q[idx] += 1

                n_refine_total += 1
                if refine_random:
                    n_refine_random += 1
                if refine_indicator:
                    n_refine_gamma += 1

                logging.info(f"\t\t[REFINE] Random={refine_random}, Indicator={refine_indicator}, Count={refines}")

                # Determine WHAT to evaluate: proposal (uncertainty) or Sobol point (random)
                if refine_indicator:
                    eval_point = proposal
                    logging.info("\t\t[REFINE] Uncertainty-driven: evaluating proposal")
                elif refine_random and sobol_ptr is not None:
                    eval_point = sobol_ptr.next()
                    logging.info(f"\t\t[REFINE] Random (Sobol): evaluating {eval_point}")
                else:
                    # Random triggered but no Sobol sampler -> fall back to proposal
                    eval_point = proposal
                    logging.info("\t\t[REFINE] Random (no Sobol sampler): evaluating proposal")

                try:
                    # exact solver (SPECFEM2D) evaluation
                    set_par_file_exact(SPECFEM2D_WORKDIR)
                    model_function_exact(*eval_point)

                    # exact misfit computation
                    phi_exact = misfit_exact(baseline_data, target_data)
                    logging.info(f"\t\t[REFINE] Exact misfit: {phi_exact}")
                except Exception as e:
                    # ──────────────────────────────────────────
                    # EXACT SOLVER CRASHED — full rollback
                    # ──────────────────────────────────────────
                    exact_crashed = True
                    logging.info(f"\t\t[REFINE] Exact solve CRASHED — rolling back entire proposal q={q}:\n{str(e)}")

                    # 1) Restore Par_file to its pre-proposal state
                    if os.path.isfile(par_file_backup):
                        shutil.copy2(par_file_backup, par_file_path)
                        logging.info("\t\t[REFINE] Restored Par_file from backup")

                    # 2) Roll back the Sobol dataset to its pre-refinement state
                    #    (undo any rows appended during earlier successful refines of this same q)
                    if X_data.shape[0] > dataset_size_before:
                        X_data = X_data[:dataset_size_before]
                        y_data = y_data[:dataset_size_before]
                        np.savez(in_path, input_array=X_data)
                        np.savez(y_path, label_misfit=y_data)
                        logging.info(f"\t\t[REFINE] Rolled back dataset to size {dataset_size_before}")

                    # 3) Roll back variable-length tracking lists
                    del random_indicator[tracking_len_before:]
                    del occasional_indicator[tracking_len_before:]
                    del trigger_q[tracking_len_before:]
                    del trigger_refines[tracking_len_before:]
                    del trigger_rand[tracking_len_before:]
                    del trigger_occ[tracking_len_before:]

                    # 4) Roll back per-q summary arrays
                    rand_trigger_q[idx] = rand_trigger_q_before
                    occ_trigger_q[idx]  = occ_trigger_q_before
                    n_refines_q[idx]    = n_refines_q_before
                    max_eps_q[idx]      = max_eps_q_before

                    # 5) Roll back refinement counters
                    n_refine_total  = n_refine_total_before
                    n_refine_random = n_refine_random_before
                    n_refine_gamma  = n_refine_gamma_before

                    break   # exit refinement while-loop

                # -------- append to Sobol dataset --------
                file = np.loadtxt(os.path.join(SPECFEM2D_DATA, "proc000000_rho_vp_vs.dat"))
                mask = (
                    (file[:, 0] >= domain_xmin) & (file[:, 0] <= domain_xmax) &
                    (file[:, 1] >= domain_zmin) & (file[:, 1] <= domain_zmax)
                )
                file = file[mask]

                rho = np.asarray(Grid(file[:, 0], file[:, 1], file[:, 2]), dtype=np.float32)
                vp  = np.asarray(Grid(file[:, 0], file[:, 1], file[:, 3]), dtype=np.float32)
                vs  = np.asarray(Grid(file[:, 0], file[:, 1], file[:, 4]), dtype=np.float32)

                x_new = np.zeros_like(X_data[0])
                x_new[0] = rho
                x_new[1] = vp
                x_new[2] = vs

                X_data = np.concatenate([X_data, x_new[None, ...]], axis=0)
                y_data = np.concatenate([y_data, np.array([[-phi_exact]])], axis=0)  # stored negative (sobol_data_gen.py convention)

                np.savez(in_path, input_array=X_data)
                np.savez(y_path, label_misfit=y_data)

                finetune_surrogate()
                reload_surrogate()  # invalidate cached model so next nn_sims() picks up new weights

                logging.info("\t\t[REFINE] Completed fine-tuning -> re-evaluating surrogate for same proposal")

                # loop continues: surrogate re-evaluation happens at top of while True
                continue

            # ──────────────────────────────────────────────────
            # POST-REFINEMENT: skip entire proposal if crashed
            # Do NOT count as iteration q, do NOT log, do NOT
            # accept/reject.  Also undo the prop_params append.
            # ──────────────────────────────────────────────────
            if exact_crashed:
                # Undo the prop_params entries we appended for this proposal
                for i in range(len(init_cov_matrix)):
                    prop_params[i].pop()
                logging.info(f"\t\t[SKIP] Proposal at q={q} fully skipped due to exact solver crash")
                continue   # back to `for lit` — q is NOT incremented

            # if surrogate evaluation crashed
            if prop_phi_samples is None:
                continue

            # ──────────────────────────────────────────────────
            # Clean up Par_file backup (successful iteration)
            # ──────────────────────────────────────────────────
            if os.path.isfile(par_file_backup):
                os.remove(par_file_backup)

            # ---------------- accept / reject (Eq. 20) ----------------
            r_bar = float(np.mean(r_k))
            alpha_nom = min(1.0, np.exp(r_bar))
            u = np.random.rand()

            logging.info(f"ACCEPTANCE RATIO (alpha): {alpha_nom}")
            logging.info(f"RANDOM NUMBER: {u}")
            logging.info(f"ACCEPTANCE RATE (%): {round(100 * int(k) / max(q, 1), 2)}")

            if u < alpha_nom:
                current_params = proposal.copy()
                accepted_samples.append(current_params.copy())
                current_likelihood = prop_mu
                current_phi_samples = prop_phi_samples.copy()
                objective_function.append(current_likelihood)
                k += 1
                logging.info("ACCEPTED")
            else:
                accepted_samples.append(current_params.copy())
                logging.info("REJECTED")

            q += 1
            if q > iterations:
                break

        logging.info(f"REFINEMENTS TOTAL: {n_refine_total}")
        logging.info(f"REFINEMENTS RANDOM: {n_refine_random}")
        logging.info(f"REFINEMENTS OCCASIONAL: {n_refine_gamma}")

    # ==========================================================
    # ======================== OFFLINE =========================
    # ==========================================================
    elif regime == "offline":

        current_likelihood = model_function(*current_params)
        objective_function.append(current_likelihood)

        C_matrix = None

        for lit in range(MaxLit):

            if q < int(iterations * 0.05):
                proposal = current_params + prior(init_cov_matrix)
            elif int(iterations * 0.05) <= q < int(iterations * 0.5):
                eps = 1e-10
                n_unknown = init_cov_matrix.shape[0]
                sd_cov = (2.4**2) / n_unknown
                C_matrix = sd_cov * np.cov(np.array(accepted_samples).T) \
                           + sd_cov * eps * np.identity(n_unknown)
                proposal = current_params + prior(C_matrix)
            else:
                proposal = current_params + prior(C_matrix)

            # Check if proposal is within bounds (ALL parameters)
            if np.any(proposal < parameter_min) or np.any(proposal > parameter_max):
                continue

            # Check circle stays inside domain (with margin)
            if (proposal[0] - proposal[2] < domain_xmin + margin) or \
               (proposal[0] + proposal[2] > domain_xmax - margin) or \
               (proposal[1] - proposal[2] < domain_zmin + margin) or \
               (proposal[1] + proposal[2] > domain_zmax - margin):
                continue

            proposal = np.round(proposal, decimals=1)
            for i in range(len(init_cov_matrix)):
                prop_params[i].append(proposal[i])

            try:
                new_likelihood = model_function(*proposal)
            except Exception as e:
                logging.info(f"Skipping iteration {q}:\n{str(e)}")
                continue

            # Acceptance ratio (Eq. 10)
            r_hat = (-0.5 * new_likelihood + 0.5 * current_likelihood) / T
            alpha = min(1.0, np.exp(r_hat))
            u = np.random.rand()

            logging.info(f"ITERATION {q}")
            logging.info(f"CURRENT MISFIT: {current_likelihood}")
            logging.info(f"NEW MISFIT: {new_likelihood}")
            logging.info(f"ACCEPTANCE RATIO: {alpha}")

            if u < alpha:
                current_params = proposal.copy()
                accepted_samples.append(current_params.copy())
                current_likelihood = new_likelihood
                objective_function.append(current_likelihood)
                k += 1
                logging.info("ACCEPTED")
            else:
                accepted_samples.append(current_params.copy())
                logging.info("REJECTED")

            q += 1
            if q > iterations:
                break

    # ==========================================================
    # ====================== TWO-STAGE =========================
    # ==========================================================
    elif regime == "two_stage":

        logging.info("TWO-STAGE: Evaluating initial state with exact solver")

        # Surrogate at initial state (MCD-averaged)
        current_surr = model_function(*current_params)
        _, _, current_surr_samples = nn_sims(run_forward=False, mc_samples=K, return_samples=True)
        current_chi_hat = float(np.mean(current_surr_samples))

        # Exact solver at initial state
        set_par_file_exact(SPECFEM2D_WORKDIR)
        model_function_exact(*current_params)
        current_chi_exact = misfit_exact(baseline_data, target_data)

        objective_function.append(current_chi_exact)

        n_stage1_reject = 0
        n_stage2_reject = 0
        n_stage2_total = 0

        C_matrix = None

        for lit in range(MaxLit):

            if q < int(iterations * 0.05):
                proposal = current_params + prior(init_cov_matrix)
            elif int(iterations * 0.05) <= q < int(iterations * 0.5):
                eps = 1e-10
                n_unknown = init_cov_matrix.shape[0]
                sd_cov = (2.4**2) / n_unknown
                C_matrix = sd_cov * np.cov(np.array(accepted_samples).T) \
                           + sd_cov * eps * np.identity(n_unknown)
                proposal = current_params + prior(C_matrix)
            else:
                proposal = current_params + prior(C_matrix)

            # Check if proposal is within bounds (ALL parameters)
            if np.any(proposal < parameter_min) or np.any(proposal > parameter_max):
                continue

            # Check circle stays inside domain (with margin)
            if (proposal[0] - proposal[2] < domain_xmin + margin) or \
               (proposal[0] + proposal[2] > domain_xmax - margin) or \
               (proposal[1] - proposal[2] < domain_zmin + margin) or \
               (proposal[1] + proposal[2] > domain_zmax - margin):
                continue

            proposal = np.round(proposal, decimals=1)
            for i in range(len(init_cov_matrix)):
                prop_params[i].append(proposal[i])

            logging.getLogger().handlers[0].stream.write("\n")
            logging.info(f"ITERATION {q}")
            logging.info(f"X-LOC {proposal[0]}, Z-LOC {proposal[1]}, RADIUS {proposal[2]}")

            # ============ STAGE 1: surrogate screening (Eq. 22-23) ============
            try:
                prop_surr = model_function(*proposal)
                prop_surr_mu, prop_surr_sd, prop_surr_samples = nn_sims(
                    run_forward=False, mc_samples=K, return_samples=True
                )
            except Exception as e:
                logging.info(f"Skipping iteration {q} due to NN pipeline crash:\n{str(e)}")
                continue

            prop_chi_hat = float(np.mean(prop_surr_samples))

            # Stage-1 log ratio (Eq. 22)
            r_hat = (-0.5 * prop_chi_hat + 0.5 * current_chi_hat) / T
            alpha_1 = min(1.0, np.exp(r_hat))

            logging.info(f"STAGE 1 - Surrogate chi_hat(curr): {current_chi_hat:.6e}")
            logging.info(f"STAGE 1 - Surrogate chi_hat(prop): {prop_chi_hat:.6e}")
            logging.info(f"STAGE 1 - alpha_1: {alpha_1:.6f}")

            u1 = np.random.rand()
            logging.info(f"STAGE 1 - u1: {u1:.6f}")

            if u1 >= alpha_1:
                accepted_samples.append(current_params.copy())
                n_stage1_reject += 1
                logging.info("STAGE 1 REJECTED (no exact solve needed)")
                q += 1
                if q > iterations:
                    break
                continue

            # ============ STAGE 2: exact correction (Eq. 24-25) ============
            logging.info("STAGE 1 PASSED -> running exact solver")
            n_stage2_total += 1

            try:
                set_par_file_exact(SPECFEM2D_WORKDIR)
                model_function_exact(*proposal)
                prop_chi_exact = misfit_exact(baseline_data, target_data)
            except Exception as e:
                logging.info(f"Exact solve failed at iteration {q}:\n{str(e)}")
                accepted_samples.append(current_params.copy())
                q += 1
                if q > iterations:
                    break
                continue

            r_exact = (-0.5 * prop_chi_exact + 0.5 * current_chi_exact) / T

            # Stage-2 correction (Eq. 25):
            # alpha_2 = min(1, exp(r/T) / exp(r_hat/T))
            alpha_2 = min(1.0, np.exp(r_exact - r_hat))

            logging.info(f"STAGE 2 - Exact chi(curr): {current_chi_exact:.6e}")
            logging.info(f"STAGE 2 - Exact chi(prop): {prop_chi_exact:.6e}")
            logging.info(f"STAGE 2 - alpha_2: {alpha_2:.6f}")

            u2 = np.random.rand()
            logging.info(f"STAGE 2 - u2: {u2:.6f}")

            if u2 < alpha_2:
                current_params = proposal.copy()
                accepted_samples.append(current_params.copy())
                current_chi_hat = prop_chi_hat
                current_chi_exact = prop_chi_exact
                objective_function.append(current_chi_exact)
                k += 1
                logging.info("STAGE 2 ACCEPTED")
            else:
                accepted_samples.append(current_params.copy())
                n_stage2_reject += 1
                logging.info("STAGE 2 REJECTED")

            logging.info(f"ACCEPTANCE RATE (%): {round(100 * int(k) / max(q, 1), 2)}")

            q += 1
            if q > iterations:
                break

        logging.info(f"STAGE 1 REJECTIONS: {n_stage1_reject}")
        logging.info(f"STAGE 2 EVALUATIONS: {n_stage2_total}")
        logging.info(f"STAGE 2 REJECTIONS: {n_stage2_reject}")

    else:
        raise ValueError(f"Unknown regime: '{regime}'. Choose 'adaptive', 'offline', or 'two_stage'.")

    # ---------------- save results ----------------
    if save:
        np.savez(os.path.join(SPECFEM2D_WORKDIR, "accepteds.npz"),
                 accepted_samples=np.array(accepted_samples),
                 misfits=np.array(objective_function))
        np.savez(os.path.join(SPECFEM2D_WORKDIR, "proposed.npz"),
                 proposals=np.array(prop_params))

        # Save refinement trigger data ONLY if adaptive ran
        if regime == "adaptive":
            np.savez(
                os.path.join(SPECFEM2D_WORKDIR, "refinement_triggers.npz"),
                random_indicator=np.array(random_indicator, dtype=float),
                occasional_indicator=np.array(occasional_indicator, dtype=float),
                trigger_q=np.array(trigger_q, dtype=np.int32),
                trigger_refines=np.array(trigger_refines, dtype=np.int32),
                trigger_rand=np.array(trigger_rand, dtype=np.int8),
                trigger_occ=np.array(trigger_occ, dtype=np.int8),
                # per-q summaries (handy for quick plots)
                rand_trigger_q=rand_trigger_q,
                occ_trigger_q=occ_trigger_q,
                n_refines_q=n_refines_q,
                max_eps_q=max_eps_q,
                beta=float(beta),
                gamma=float(gamma),
                max_refines=int(max_refines),
            )

    end = time.time()
    logging.info(f"TOTAL SAMPLING TIME: {(end - start)/3600:.2f} HOURS")
    logging.info(f"NUMBER OF ACCEPTED SAMPLES: {k}")
    logging.info(f"ACCEPTANCE RATIO: {100*k/max(iterations, 1):.2f}%")

    return None