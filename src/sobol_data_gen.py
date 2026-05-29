#!/usr/bin/env python
# coding: utf-8

# import libraries
import os, time, pickle
import numpy as np
from pathlib import Path

from scipy.stats import qmc

from src.seismic.specfem2d import forward
from utils.cubit_mesher import MCMC_automatic_mesh
from utils.grid_interp import Grid

from config import SPECFEM2D_WORKDIR, SPECFEM2D_DATA, TARGET_DATA
from config import n_shots, n_recievers, nstep, n_sample, n_recievers ,stdvs, sigma, parameter_min, parameter_max, resX, resY
from config import domain_xmax, domain_xmin, domain_zmax, domain_zmin

from config import machine_type

import logging
logging.basicConfig(
    filename= os.path.join(SPECFEM2D_WORKDIR,"sobol_data_run.log"),
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M"
)

# Model function with parameters
def model_function(x_center, z_center, radius, alpha=0, proposed_vp=2000):

    # MESH GENERATION & P VELOCITY CHANGE
    minor_axis = radius
    major_axis = radius
    MCMC_automatic_mesh(x_center, z_center, major_axis, minor_axis, alpha, proposed_vp)
    # FORWARD SIMULATIONS
    forward.sims(machine_type)
    return None


# misfit function
def misfit(baseline_data, target):
    """
    Gaussian log-likelihood misfit consistent with
    generate_target_from_baseline_and_monitor().

    baseline_data : 1-D float32 array (reference baseline)
    target        : 1-D float32 array (observed noisy difference)
    target_dir    : directory containing data.noise.npz

    Returns: log-likelihood (up to an additive constant)
    """

    sigma_n = 1.0  # unit variance: misfit = -0.5 * ||r||^2 (no artificial Cinv scaling)

    # --- Inverse covariance (C = sigma_n^2 I) ---
    sigma_n = sigma_n
    Cinv = 1.0 / (sigma_n ** 2)

    # --- Forward model ---
    modelled = forward.data_1d(n_shots, SPECFEM2D_WORKDIR)

    # --- Predicted difference ---
    pred_diff = modelled - baseline_data

    # --- Residual (observed - predicted) ---
    r = target - pred_diff

    # --- Gaussian log-likelihood (up to constant) ---
    phi = -0.5 * Cinv * np.dot(r, r)
    return phi



# Training dataset generation
# ----------------------- SOBOL SEQUENCE BASED TRAINING DATASET GENERATION FUNCTION ----------------------- #
def sobol_gen(
    baseline_data,
    target_data,
    iterations,
    parameter_min=parameter_min,
    parameter_max=parameter_max,
    save=True,
    n=10,
    resX=resX,
    resY=resY,
    domain_xmin=domain_xmin,
    domain_xmax=domain_xmax,
    domain_zmin=domain_zmin,
    domain_zmax=domain_zmax,
):

    logging.info("INITIALIZATION")
    logging.getLogger().handlers[0].stream.write("\n")

    start = time.time()

    N = len(target_data)
    n_param = len(parameter_min)
    q = 0                       # successful-sample counter (0-based indexing)

    prop_params = [[] for _ in range(n_param)]

    # Sobol setup — draw EXACTLY iterations samples
    # Every sample is guaranteed valid by construction (geometry-aware scaling),
    # so the only reason to skip is a crash. Retry logic handles that.
    sobol_sampler = qmc.Sobol(d=n_param, scramble=True)

    logging.info("SOBOL SEQUENCE SAMPLING")

    sobol_samples = sobol_sampler.random(n=iterations)

    # Storage arrays
    input_values = np.zeros((iterations, n_param), dtype=np.float32)
    input_array  = np.zeros((iterations, 3, resX, resY), dtype=np.float32)

    samples_per_trace = nstep // n_sample
    values_per_component = n_recievers * samples_per_trace
    values_per_shot = 2 * values_per_component

    label_data   = np.zeros((iterations, int(values_per_shot * n_shots)), dtype=np.float32)
    label_misfit = np.zeros((iterations, 1), dtype=np.float32)

    param_labels = ["X-LOC", "Z-LOC", "RADIUS"]

    margin = 100

    # Local retry parameters — if a proposal crashes, try nearby samples
    max_local_retries = 5
    local_perturb_scale = 0.05  # fraction of parameter range to perturb by (5%)

    logging.info("LOOP STARTED")

    for lit in range(iterations):

        u = sobol_samples[lit]  # in [0,1]^3

        # ---- 1) Scale ALL parameters from [0,1] to [parameter_min, parameter_max] ----
        X = parameter_min[0] + u[0] * (parameter_max[0] - parameter_min[0])
        Z = parameter_min[1] + u[1] * (parameter_max[1] - parameter_min[1])
        R = parameter_min[2] + u[2] * (parameter_max[2] - parameter_min[2])

        # ---- 2) Geometry check: circle must fit inside domain with margin ----
        if (X - R - margin < domain_xmin) or \
           (X + R + margin > domain_xmax) or \
           (Z - R - margin < domain_zmin) or \
           (Z + R + margin > domain_zmax):
            logging.info(f"Sobol index {lit} failed geometry check. Skipping.")
            continue

        base_proposal = np.round(np.array([X, Z, R]), 1)

        # Precompute parameter ranges for perturbations
        param_range = np.array(parameter_max, dtype=float) - np.array(parameter_min, dtype=float)

        success = False
        proposal = None

        for attempt in range(max_local_retries + 1):
            if attempt == 0:
                candidate = base_proposal.copy()
            else:
                # Perturb and re-enforce geometry constraints
                noise = np.random.uniform(
                    low=-local_perturb_scale,
                    high=local_perturb_scale,
                    size=base_proposal.shape
                ) * param_range

                candidate = base_proposal + noise

                # Clip ALL parameters to their bounds
                for pi in range(len(parameter_min)):
                    candidate[pi] = np.clip(candidate[pi], parameter_min[pi], parameter_max[pi])
                candidate = np.round(candidate, 1)

                # Geometry check: circle must fit inside domain with margin
                r_c = candidate[2]
                if (candidate[0] - r_c - margin < domain_xmin) or \
                   (candidate[0] + r_c + margin > domain_xmax) or \
                   (candidate[1] - r_c - margin < domain_zmin) or \
                   (candidate[1] + r_c + margin > domain_zmax):
                    logging.info(
                        f"Attempt {attempt}/{max_local_retries} "
                        f"for Sobol index {lit} failed geometry. Skipping retry."
                    )
                    continue

                logging.info(
                    f"Attempt {attempt}/{max_local_retries} "
                    f"with nearby proposal for Sobol index {lit}"
                )

            # === Clean TEMP files ===
            src_files = [
                'absorbing_surface_file',
                'absorbing_cpml_file',
                'free_surface_file',
                'materials_file',
                'mesh_file',
                'nodes_coords_file'
            ]

            for file_name in src_files:
                file_path = os.path.join(
                    os.path.join(SPECFEM2D_WORKDIR, "DATA/TEMP"),
                    file_name
                )
                if os.path.exists(file_path):
                    os.remove(file_path)

            # === Attempt model run ===
            try:
                model_function(*candidate)
                modelled = forward.data_1d(n_shots, SPECFEM2D_WORKDIR)
                sampled_misift = misfit(baseline_data, target_data)

                proposal = candidate
                success = True
                break

            except Exception as e:
                logging.info(
                    f"Crash for candidate (attempt {attempt}) at Sobol index {lit}, "
                    f"q={q}:\n{str(e)}"
                )
                continue

        if not success:
            logging.info(
                f"All {max_local_retries + 1} attempts failed for Sobol index {lit}. "
                f"Skipping this sample."
            )
            continue

        # ---- Store results ----
        logging.getLogger().handlers[0].stream.write("\n")
        logging.info(f"SAMPLE POINT {q+1}")
        for i in range(n_param):
            logging.info(f"{param_labels[i]:<15} {proposal[i]}")
        logging.info(f"MISFIT: {-1 * sampled_misift}")

        # Store parameters
        for i in range(n_param):
            prop_params[i].append(proposal[i])

        input_values[q, :] = proposal[:]

        # Read model
        file = np.loadtxt(
            os.path.join(SPECFEM2D_DATA, "proc000000_rho_vp_vs.dat")
        )

        mask = (
            (file[:, 0] >= domain_xmin) &
            (file[:, 0] <= domain_xmax) &
            (file[:, 1] >= domain_zmin) &
            (file[:, 1] <= domain_zmax)
        )

        file_filtered = file[mask]

        rho = Grid(file_filtered[:, 0], file_filtered[:, 1], file_filtered[:, 2])
        vp  = Grid(file_filtered[:, 0], file_filtered[:, 1], file_filtered[:, 3])
        vs  = Grid(file_filtered[:, 0], file_filtered[:, 1], file_filtered[:, 4])

        input_array[q, 0, :, :] = rho
        input_array[q, 1, :, :] = vp
        input_array[q, 2, :, :] = vs

        label_data[q, :] = modelled
        label_misfit[q, 0] = sampled_misift

        q += 1

        # Save every n iterations
        if q % n == 0:
            np.savez(
                os.path.join(SPECFEM2D_WORKDIR, 'updated_input_values.npz'),
                input_values=input_values[:q]
            )
            np.savez(
                os.path.join(SPECFEM2D_WORKDIR, 'updated_input_array.npz'),
                input_array=input_array[:q]
            )
            np.savez(
                os.path.join(SPECFEM2D_WORKDIR, 'updated_label_data.npz'),
                label_data=label_data[:q]
            )
            np.savez(
                os.path.join(SPECFEM2D_WORKDIR, 'updated_label_misfit.npz'),
                label_misfit=label_misfit[:q]
            )
            logging.info(f"SAVED INTERMEDIATE DATASETS AT ITERATION {q}")

    logging.info("----------------------------------------  SOBOL TRAINING DATASET GENERATION TERMINATED SUCCESSFULLY  ----------------------------------------")
    logging.info(f"SUCCESSFUL SAMPLES: {q} / {iterations}")

    end = time.time()

    if save:
        os.makedirs(
            os.path.join(SPECFEM2D_WORKDIR, "TRAINING_DATASET"),
            exist_ok=True
        )

        np.savez('TRAINING_DATASET/sobol_input_values.npz', input_values=input_values[:q])
        np.savez('TRAINING_DATASET/sobol_input_array.npz', input_array=input_array[:q])
        np.savez('TRAINING_DATASET/sobol_label_data.npz', label_data=label_data[:q])
        np.savez('TRAINING_DATASET/sobol_label_misfit.npz', label_misfit=label_misfit[:q])

    logging.getLogger().handlers[0].stream.write("\n")
    logging.info(f"TOTAL TRAINING DATA GEN TIME: {round((end - start)/3600, 2)} HOURS.")

    # -----------------------------------------------------------------------
    # Save the Sobol sampler state so the adaptive regime can continue
    # drawing from the same low-discrepancy sequence.  The sampler has
    # already been advanced past the training points, so .random(1) will
    # yield the next unused point in [0,1]^d.
    # -----------------------------------------------------------------------
    sampler_path = os.path.join(SPECFEM2D_WORKDIR, "sobol_sampler.pkl")
    with open(sampler_path, "wb") as f:
        pickle.dump(sobol_sampler, f)
    logging.info(f"Saved Sobol sampler state to {sampler_path}")

    return None