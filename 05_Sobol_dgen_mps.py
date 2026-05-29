#!/usr/bin/env python
# coding: utf-8
"""
05_Sobol_dgen_mps.py
====================
MPS-aware Sobol training-dataset generation for the ellipse (DoF=3) example.

DoF=3: x_center, z_center, radius (circle location + size).
Fixed defaults: alpha=0, proposed_vp=2000.
Note: radius is used for both major_axis and minor_axis (circle).

Usage:
    python 05_Sobol_dgen_mps.py --batch-size 8
    (must be run inside an active MPS daemon — see launch_mps_sobol.sh)
"""

import os, sys, shutil, subprocess, argparse, time, math, pickle, logging
from pathlib import Path
from multiprocessing import Process, Queue, current_process

import numpy as np
from scipy.stats import qmc

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from config import (
    SPECFEM2D_WORKDIR, SPECFEM2D_DATA, TARGET_DATA, WORKDIR,
    n_shots, n_recievers, nstep, n_sample,
    parameter_min, parameter_max, resX, resY,
    domain_xmax, domain_xmin, domain_zmax, domain_zmin,
    machine_type, iterations as default_iterations,
)

from utils.load_data import load_data
from config import BASE_DATA

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_LOCAL_RETRIES   = 5
LOCAL_PERTURB_SCALE = 0.05
PARAM_LABELS = ["X-LOC", "Z-LOC", "RADIUS"]


# ===========================================================================
# Worker directory management
# ===========================================================================
def clone_workdir(src: str, dst: str, repo_root: str):
    """Clone SPECFEM2D_WORKDIR for a worker with symlinks for shared assets."""
    src = Path(src)
    dst = Path(dst)
    repo = Path(repo_root)

    if dst.exists():
        shutil.rmtree(dst)

    shutil.copytree(str(src), str(dst), symlinks=True)

    bin_dir = dst / "bin"
    if bin_dir.exists():
        for f in bin_dir.iterdir():
            if f.is_file():
                os.chmod(f, 0o755)

    # Symlink CPML executables from repo's utils/CPML/
    cpml_dir = dst / "CPML"
    cpml_dir.mkdir(exist_ok=True)
    cpml_src = repo / "utils" / "CPML"
    for exe_name in [
        "xadd_CPML_layers_to_an_existing_mesh",
        "xconvert_external_layers_of_a_given_mesh_to_CPML_layers",
    ]:
        exe_src = cpml_src / exe_name
        exe_dst = cpml_dir / exe_name
        if exe_src.exists() and not exe_dst.exists():
            os.symlink(str(exe_src), str(exe_dst))

    for f in cpml_dir.iterdir():
        if f.is_file() and f.name.startswith("x"):
            os.chmod(f, 0o755)

    # Symlink repo-root assets
    for name in ["Frame", "utils", "src", "config.py", "__pycache__"]:
        link = dst / name
        target = repo / name
        if target.exists() and not link.exists():
            os.symlink(str(target), str(link))

    return str(dst)


# ===========================================================================
# Single-sample processing
# ===========================================================================
def process_one_sample(
    sobol_index, sobol_unit_sample, worker_workdir,
    baseline_data, target_data,
    p_min, p_max, param_range, n_param, worker_id,
):
    import config
    from src.seismic.specfem2d import forward
    from utils.cubit_mesher import MCMC_automatic_mesh
    from utils.grid_interp import Grid
    import utils.cubit_mesher as cubit_mesher_mod

    # --- Monkey-patch paths for this worker ---
    config.SPECFEM2D_WORKDIR = worker_workdir
    config.SPECFEM2D_DATA    = os.path.join(worker_workdir, "DATA")
    config.SPECFEM2D_BIN     = os.path.join(worker_workdir, "bin")
    config.SPECFEM2D_OUTPUT  = os.path.join(worker_workdir, "OUTPUT_FILES")

    cubit_mesher_mod.SPECFEM2D_WORKDIR = worker_workdir
    cubit_mesher_mod.SPECFEM2D_DATA    = config.SPECFEM2D_DATA
    cubit_mesher_mod.SPECFEM2D_BIN     = config.SPECFEM2D_BIN
    cubit_mesher_mod.SPECFEM2D_OUTPUT  = config.SPECFEM2D_OUTPUT

    forward.SPECFEM2D_WORKDIR = worker_workdir
    forward.SPECFEM2D_DATA    = config.SPECFEM2D_DATA

    wlog = logging.getLogger(f"worker-{worker_id}")

    u = sobol_unit_sample
    base_proposal = np.round(p_min + u * param_range, 1)

    for attempt in range(MAX_LOCAL_RETRIES + 1):
        if attempt == 0:
            candidate = base_proposal.copy()
        else:
            noise = np.random.uniform(
                low=-LOCAL_PERTURB_SCALE, high=LOCAL_PERTURB_SCALE,
                size=base_proposal.shape,
            ) * param_range
            candidate = np.clip(base_proposal + noise, p_min, p_max)
            candidate = np.round(candidate, 1)
            wlog.info(f"  Retry {attempt}/{MAX_LOCAL_RETRIES} for Sobol index {sobol_index}")

        if np.any(candidate < p_min) or np.any(candidate > p_max):
            continue

        # Clean TEMP files
        temp_dir = os.path.join(worker_workdir, "DATA", "TEMP")
        for fname in ['absorbing_surface_file', 'absorbing_cpml_file',
                      'free_surface_file', 'materials_file', 'mesh_file',
                      'nodes_coords_file']:
            fp = os.path.join(temp_dir, fname)
            if os.path.exists(fp):
                os.remove(fp)

        try:
            # DoF=3: x_center, z_center, radius
            # radius is used for both major_axis and minor_axis (circle)
            # Fixed: alpha=0, proposed_vp=2000
            x_center = candidate[0]
            z_center = candidate[1]
            radius   = candidate[2]
            MCMC_automatic_mesh(x_center, z_center, radius, radius, 0, 2000)
            forward.sims(machine_type)
            modelled = forward.data_1d(n_shots, worker_workdir)

            # Misfit
            sigma_n = 1.0
            Cinv = 1.0 / (sigma_n ** 2)
            pred_diff = modelled - baseline_data
            r = target_data - pred_diff
            phi = -0.5 * Cinv * np.dot(r, r)

            # Read model and interpolate
            dat_path = os.path.join(worker_workdir, "DATA", "proc000000_rho_vp_vs.dat")
            file = np.loadtxt(dat_path)
            mask = (
                (file[:, 0] >= domain_xmin) & (file[:, 0] <= domain_xmax) &
                (file[:, 1] >= domain_zmin) & (file[:, 1] <= domain_zmax)
            )
            file_filtered = file[mask]
            rho = Grid(file_filtered[:, 0], file_filtered[:, 1], file_filtered[:, 2])
            vp  = Grid(file_filtered[:, 0], file_filtered[:, 1], file_filtered[:, 3])
            vs  = Grid(file_filtered[:, 0], file_filtered[:, 1], file_filtered[:, 4])
            model_channels = np.stack([rho, vp, vs], axis=0)

            return {
                "sobol_index": sobol_index,
                "proposal": candidate.copy(),
                "model_channels": model_channels,
                "modelled": modelled.copy(),
                "misfit": phi,
            }

        except Exception as e:
            wlog.info(f"  Crash at Sobol index {sobol_index} attempt {attempt}: {e}")
            continue

    wlog.info(f"  All attempts failed for Sobol index {sobol_index}")
    return None


# ===========================================================================
# Worker process main loop
# ===========================================================================
def worker_main(
    worker_id, task_queue, result_queue,
    worker_workdir, baseline_data, target_data,
):
    os.chdir(worker_workdir)
    sys.path.insert(0, worker_workdir)

    log_path = os.path.join(worker_workdir, f"worker_{worker_id}.log")
    logging.basicConfig(
        filename=log_path, filemode="w", level=logging.INFO,
        format=f"%(asctime)s [W{worker_id}] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    )
    wlog = logging.getLogger(f"worker-{worker_id}")
    wlog.info(f"Worker {worker_id} started, workdir={worker_workdir}, PID={os.getpid()}")

    p_min = np.array(parameter_min, dtype=float)
    p_max = np.array(parameter_max, dtype=float)
    param_range = p_max - p_min
    n_param = len(parameter_min)

    while True:
        item = task_queue.get()
        if item is None:
            wlog.info(f"Worker {worker_id} received shutdown signal")
            break

        sobol_index, unit_sample = item
        wlog.info(f"Processing Sobol index {sobol_index}")

        result = process_one_sample(
            sobol_index, unit_sample, worker_workdir,
            baseline_data, target_data,
            p_min, p_max, param_range, n_param, worker_id,
        )
        result_queue.put((sobol_index, result))

    wlog.info(f"Worker {worker_id} exiting")


# ===========================================================================
# Main orchestrator
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="MPS-parallel Sobol data generation")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of concurrent workers (MPS clients)")
    parser.add_argument("--iterations", type=int, default=default_iterations,
                        help="Total Sobol samples to generate")
    args = parser.parse_args()

    batch_size = args.batch_size
    iterations = args.iterations

    # --- Pre-flight: copy CPML tools and Frame into base SPECFEM2D_WORKDIR ---
    CPML_SRC  = REPO_ROOT / "utils" / "CPML"
    FRAME_SRC = REPO_ROOT / "utils" / "Frame"
    CPML_DST  = Path(SPECFEM2D_WORKDIR) / "CPML"
    FRAME_DST = Path(SPECFEM2D_WORKDIR) / "Frame"

    if not CPML_SRC.exists():
        raise FileNotFoundError(f"CPML source not found at {CPML_SRC}")
    if not FRAME_SRC.exists():
        raise FileNotFoundError(f"Frame source not found at {FRAME_SRC}")

    shutil.copytree(CPML_SRC, CPML_DST, dirs_exist_ok=True)
    shutil.copytree(FRAME_SRC, FRAME_DST, dirs_exist_ok=True)

    add_bin  = CPML_DST / "xadd_CPML_layers_to_an_existing_mesh"
    conv_bin = CPML_DST / "xconvert_external_layers_of_a_given_mesh_to_CPML_layers"

    if not (add_bin.exists() and conv_bin.exists()):
        mk = CPML_DST / "Makefile"
        if mk.exists():
            subprocess.run(["make"], cwd=str(CPML_DST), check=True)
        else:
            add_src  = CPML_DST / "xadd_CPML_layers_to_an_existing_mesh.f90"
            conv_src = CPML_DST / "xconvert_external_layers_of_a_given_mesh_to_CPML_layers.f90"
            if add_src.exists() and conv_src.exists():
                subprocess.run(["gfortran", "-O3", str(add_src),  "-o", str(add_bin)],  check=True)
                subprocess.run(["gfortran", "-O3", str(conv_src), "-o", str(conv_bin)], check=True)
            else:
                raise FileNotFoundError("CPML binaries and sources not found; cannot build.")

    for p in (add_bin, conv_bin):
        if p.exists():
            os.chmod(p, 0o755)

    # --- Setup logging ---
    logging.basicConfig(
        filename=os.path.join(SPECFEM2D_WORKDIR, "sobol_mps_run.log"),
        filemode="w", level=logging.INFO,
        format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M",
    )
    log = logging.getLogger("main")

    log.info(f"MPS Sobol generation: batch_size={batch_size}, iterations={iterations}")
    log.info(f"Base workdir: {SPECFEM2D_WORKDIR}")

    start_time = time.time()

    # --- Load data ---
    n_values = n_shots * (2 * n_recievers * (nstep // n_sample))
    baseline_data = load_data(os.path.join(BASE_DATA, "data.bin"), n_values)
    target_data   = load_data(os.path.join(TARGET_DATA, "noiseless_data.bin"), n_values)

    # --- Generate Sobol sequence ---
    n_param = len(parameter_min)
    sobol_sampler = qmc.Sobol(d=n_param, scramble=True)
    sobol_samples = sobol_sampler.random(n=iterations)

    # --- Create per-worker clones ---
    worker_dirs = []
    base_workdir = Path(SPECFEM2D_WORKDIR)
    for w in range(batch_size):
        wdir = str(base_workdir.parent / f"specfem2d_workdir_mps_{w}")
        log.info(f"Cloning workdir for worker {w} → {wdir}")
        clone_workdir(SPECFEM2D_WORKDIR, wdir, str(REPO_ROOT))
        worker_dirs.append(wdir)

    # --- Launch workers ---
    task_queue   = Queue()
    result_queue = Queue()

    workers = []
    for w in range(batch_size):
        p = Process(
            target=worker_main,
            args=(w, task_queue, result_queue, worker_dirs[w],
                  baseline_data, target_data),
            daemon=True,
        )
        p.start()
        workers.append(p)
        log.info(f"Worker {w} launched (PID={p.pid})")

    # --- Enqueue samples ---
    for i in range(iterations):
        task_queue.put((i, sobol_samples[i]))
    for _ in range(batch_size):
        task_queue.put(None)

    # --- Collect results ---
    n_param = len(parameter_min)
    samples_per_trace = int(nstep) // int(n_sample)
    values_per_component = int(n_recievers) * samples_per_trace
    values_per_shot = 2 * values_per_component

    input_values = np.zeros((iterations, n_param), dtype=np.float32)
    input_array  = np.zeros((iterations, 3, resX, resY), dtype=np.float32)
    label_data   = np.zeros((iterations, values_per_shot * int(n_shots)), dtype=np.float32)
    label_misfit = np.zeros((iterations, 1), dtype=np.float32)

    q = 0
    received = 0
    save_every = 10

    while received < iterations:
        sobol_index, result = result_queue.get()
        received += 1

        if result is None:
            log.info(f"Sample {sobol_index}: skipped or failed")
            continue

        input_values[q, :]    = result["proposal"]
        input_array[q, :,:,:] = result["model_channels"]
        label_data[q, :]      = result["modelled"]
        label_misfit[q, 0]    = result["misfit"]

        log.info(f"Sample {sobol_index} → stored as q={q}, misfit={-1*result['misfit']:.4e}")
        for i, lbl in enumerate(PARAM_LABELS):
            log.info(f"  {lbl:<15} {result['proposal'][i]}")

        q += 1

        if q % save_every == 0:
            _save_intermediate(SPECFEM2D_WORKDIR, q, input_values, input_array, label_data, label_misfit)
            log.info(f"Saved intermediate datasets at q={q}")

        elapsed = time.time() - start_time
        rate = q / (elapsed / 3600) if elapsed > 0 else 0
        log.info(f"Progress: {received}/{iterations} dispatched, {q} successful, "
                 f"{rate:.1f} samples/hr")

    # --- Finalize ---
    for p in workers:
        p.join(timeout=30)

    end_time = time.time()
    total_hours = (end_time - start_time) / 3600

    log.info(f"SOBOL MPS GENERATION COMPLETE: {q}/{iterations} successful samples")
    log.info(f"Total time: {total_hours:.2f} hours")

    os.makedirs(os.path.join(SPECFEM2D_WORKDIR, "TRAINING_DATASET"), exist_ok=True)
    np.savez(os.path.join(SPECFEM2D_WORKDIR, "TRAINING_DATASET/sobol_input_values.npz"),
             input_values=input_values[:q])
    np.savez(os.path.join(SPECFEM2D_WORKDIR, "TRAINING_DATASET/sobol_input_array.npz"),
             input_array=input_array[:q])
    np.savez(os.path.join(SPECFEM2D_WORKDIR, "TRAINING_DATASET/sobol_label_data.npz"),
             label_data=label_data[:q])
    np.savez(os.path.join(SPECFEM2D_WORKDIR, "TRAINING_DATASET/sobol_label_misfit.npz"),
             label_misfit=label_misfit[:q])

    sampler_path = os.path.join(SPECFEM2D_WORKDIR, "sobol_sampler.pkl")
    with open(sampler_path, "wb") as f:
        pickle.dump(sobol_sampler, f)
    log.info(f"Saved final datasets and sampler state")

    for wdir in worker_dirs:
        if os.path.exists(wdir):
            shutil.rmtree(wdir, ignore_errors=True)
            log.info(f"Cleaned up {wdir}")


def _save_intermediate(workdir, q, input_values, input_array, label_data, label_misfit):
    np.savez(os.path.join(workdir, 'updated_input_values.npz'),  input_values=input_values[:q])
    np.savez(os.path.join(workdir, 'updated_input_array.npz'),   input_array=input_array[:q])
    np.savez(os.path.join(workdir, 'updated_label_data.npz'),    label_data=label_data[:q])
    np.savez(os.path.join(workdir, 'updated_label_misfit.npz'),  label_misfit=label_misfit[:q])


if __name__ == "__main__":
    main()