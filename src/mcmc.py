#!/usr/bin/env python
# coding: utf-8

# import libraries
import os, time
import numpy as np
from pathlib import Path

from src.seismic.specfem2d import forward
from utils.cubit_mesher import MCMC_automatic_mesh

from config import SPECFEM2D_WORKDIR, TARGET_DATA
from config import n_shots, n_recievers, nstep, n_sample ,stdvs, sigma, parameter_min, parameter_max, domain_xmin, domain_xmax, domain_zmin, domain_zmax

from config import machine_type

import logging
logging.basicConfig(
    filename= os.path.join(SPECFEM2D_WORKDIR,"mcmc_run.log"),
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
def misfit(baseline_data, target, target_dir=TARGET_DATA):
    """
    Gaussian log-likelihood misfit consistent with
    generate_target_from_baseline_and_monitor().

    baseline_data : 1-D float32 array (reference baseline)
    target        : 1-D float32 array (observed noisy difference)
    target_dir    : directory containing data.noise.npz

    Returns: log-likelihood (up to an additive constant)
    """

    # --- Load noise metadata ---
    meta_path = Path(target_dir) / "data.noise.npz"
    if not meta_path.exists():
        raise FileNotFoundError(f"Noise metadata not found: {meta_path}")

    meta = np.load(meta_path)
    sigma_n = float(meta["sigma_n"])

    # --- Inverse covariance (C = sigma_n^2 I) ---
    # If sigma_n is zero (noiseless case), use unit variance
    sigma_n = sigma_n if sigma_n > 0.0 else 1.0
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


def exact_misfit(baseline_data, target):
    """
    Returns POSITIVE misfit (same convention as nn_mcmc.py)
    """
    phi = misfit(baseline_data, target)
    return -1.0 * float(phi)


# Prior function: NORMAL
def prior(cov_matrix):
    return np.random.multivariate_normal(np.zeros(cov_matrix.shape[0]), cov_matrix)


# Metropolis-Hastings algorithm
# ----------------------- NORMAL
def MH(baseline_data, target_data, initial_model, iterations, parameter_min=parameter_min, parameter_max=parameter_max, stdvs=np.array(stdvs), save=True, n=10):

    initial_model = np.array(initial_model)
    init_cov_matrix = np.diag(stdvs**2)
    
    # tracking time
    start = time.time()
    
    # initializing some params/variables
    T = 1E4            # Arbitrarily choosen value as the temperature (T>1 Warming, T<1 Cooling)
    N = len(target_data)
    objective_function = []
    accepted_samples = []
    k = 0
    q = 1
    prop_params = [[] for _ in range(len(parameter_min))]
    
    
    # ACCEPTED TRIAL BOUNDARY
    parameter_min = np.array(parameter_min)
    parameter_max = np.array(parameter_max)

    for i, (lo, hi) in enumerate(zip(parameter_min, parameter_max)):
        assert lo <= hi, f"Bounds swapped at index {i}: min={lo}, max={hi}"

    
    current_params = np.array(initial_model)
    
    model_function(*current_params)
    current_likelihood = misfit(baseline_data, target_data)
    objective_function.append(-1 * current_likelihood)
    
    MaxLit = 5000000000

    # This is for making sure the cricle is ALWAYS inside the domain
    margin = 100
    
    for lit in range(MaxLit):
        if q < int(iterations * 0.05):
            proposal = current_params + prior(init_cov_matrix)
        
        elif int(iterations * 0.05) <= q < int(iterations * 0.5):
            eps = 1.0e-10
            n_unknown = init_cov_matrix.shape[0]
            sd = (2.4 ** 2) / n_unknown
            C_matrix = sd * np.cov(np.array(accepted_samples).T) + sd * eps * np.identity(n_unknown)
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
                
        
        if q == int(iterations/10):
            logging.getLogger().handlers[0].stream.write("\n")
            logging.info("ADAPTATIVE M-H STARTED")


        # Iterate over the files and remove them
        src_files = ['absorbing_surface_file','absorbing_cpml_file', 'free_surface_file', 'materials_file', 'mesh_file', 'nodes_coords_file']

        for file_name in src_files:
            file_path = os.path.join(os.path.join(SPECFEM2D_WORKDIR, "DATA/TEMP"), file_name)
            if os.path.exists(file_path):
                os.remove(file_path)

        
        # === Attempt model run (Cubit + Specfem2D) ===
        try:
            model_function(*proposal)  # Cubit + Specfem2D meshing and run
            new_likelihood = misfit(baseline_data, target_data)
        except Exception as e:
            logging.info(f"Skipping iteration {q} due to Cubit or Specfem2D crash:\n{str(e)}")
            continue  # Skip this iteration
        
        # Store proposals
        for i in range(len(init_cov_matrix)):
            prop_params[i].append(proposal[i])

        # Print values
        param_labels = [
            "X-LOC", "Z-LOC", "RADIUS",
        ]
        
        logging.getLogger().handlers[0].stream.write("\n")
        logging.info(f"ITERATION {q}")
        for i in range(len(init_cov_matrix)):
            logging.info(f"{param_labels[i]:<15} {proposal[i]}")

        logging.info(f"CURRENT MISFIT: {-1 * current_likelihood}")
        logging.info(f"NEW MISFIT: {-1 * new_likelihood}")
        
        # ACCEPTANCE RATIO
        acceptance_ratio = np.exp((1/T)*(new_likelihood - current_likelihood))
        logging.info(f"ACCEPTANCE RATIO: {acceptance_ratio}")

        u = np.random.rand()
        logging.info(f"RANDOM NUMBER: {u}")
        logging.info(f"TEMPERATURE: {T}")
        logging.info(f"ACCEPTANCE RATE (%): {round(100 * int(k) / iterations, 2)}")
        
        # Modified sample selection (Tarantola's SIAM book - Chapter 2)
        if acceptance_ratio >= 1 or u < acceptance_ratio:
            # print("\tYES")
            current_params = proposal.copy()
            accepted_samples.append(current_params.copy())
            current_likelihood = new_likelihood
            objective_function.append(-1*current_likelihood)

            # save objective function values on disk at each acceptance
            np.savetxt("objective_function.txt", np.array(objective_function))


            if k % n == 0:
                if os.path.exists("ongoing_sampling.npy"):
                    os.remove("ongoing_sampling.npy")

                # Save the numpy array to an HDF5 file
                np.save('ongoing_sampling.npy', np.array(accepted_samples))


            # Keep track of the accepted models
            k = k + 1
            # likelihoods.append(new_posterior)
            logging.info("ACCEPTED")
        
        else:
            # print("\tNO")
            accepted_samples.append(current_params.copy())
            #objective_function.append(current_likelihood)

        
        # path_to_log = os.path.join(SPECFEM2D_WORKDIR, "temp_output.log")
        # clean_file(path_to_log)
        
        
        # Keep track of the iterations
        q = q + 1
        # time.sleep(0.05)

        if q == iterations + 1:
            logging.getLogger().handlers[0].stream.write("\n")
            logging.getLogger().handlers[0].stream.write("\n")
            logging.getLogger().handlers[0].stream.write("\n")
            logging.info("----------------------------------------  MCMC TERMINATED SUCCESSFULLY  ----------------------------------------")
            logging.info(f"MAX ITERATION: {q} REACHED!. LOOP ITERATION IS @ {lit}")
            break

        if lit == MaxLit:
            logging.info(f"LOOP ITERATION IS OVER!! ITERATION IS: {lit}")


    logging.info(f"LOOP ITERATION IS {lit}")

    # tracking time
    end = time.time()
    # track_time.append(round(time_b - time_a, 2))
    
    # save samples in a .npz file
    if save == True:
        # Save the numpy array to an NPY file
        np.savez(os.path.join(SPECFEM2D_WORKDIR,'accepteds.npz'), accepted_samples=np.array(accepted_samples), misfits=np.array(objective_function))

        proposals = np.array(prop_params)
        np.savez(os.path.join(SPECFEM2D_WORKDIR,'proposed.npz'), proposals)

    # print("\nAVERAGE TIME FOR EACH ITERATION IS ", round(np.mean(np.array(track_time)), 2), " SECONDS.")
    logging.info(f"TOTAL SAMPLING TIME: {round(np.array(end - start) / 3600, 2)} HOURS.")
    # print("SHAPE OF FINAL SAMPLES: ", (np.array(accepted_samples)).shape)
    logging.info(f"NUMBER OF ACCEPTED SAMPLES: {int(k)}")
    logging.info(f"ACCEPTANCE RATIO: {round(100 * int(k)/iterations, 2)} %")
    
    # return np.array(accepted_samples), np.array(objective_function)
    return None
