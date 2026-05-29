#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os, sys, shutil, subprocess
from pathlib import Path

#import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# Add the config path to the directory
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from config import SPECFEM2D_WORKDIR, SPECFEM2D_DATA
from config import n_task, n_recievers, n_sample, nstep


# Helper functions for running on DRAC clusters
def _run_step(n_tasks, *argv):
    """
    Launch a SLURM job step inside the current allocation.
    Ensures 1 GPU per rank so CUDA_VISIBLE_DEVICES is set per rank.
    """
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Not running inside a SLURM allocation.")

    # sanity: make sure the allocation can satisfy this step
    alloc_ntasks = int(os.environ.get("SLURM_NTASKS", n_tasks))
    if n_tasks > alloc_ntasks:
        raise RuntimeError(f"Requested {n_tasks} tasks but allocation has only {alloc_ntasks}.")

    cpus = os.environ.get("SLURM_CPUS_PER_TASK", "1")

    # Request 1 GPU per task; with ntasks-per-gpu=1 this maps ranks→GPUs
    cmd = [
        "srun",
        "-n", str(n_tasks),
        "--cpus-per-task", cpus,
        *argv
    ]
    subprocess.run(cmd, check=True)

def forward_obs_cluster(workdir, n_task):
    # Work dir
    os.chdir(workdir)

    # Threading consistent with --cpus-per-task=1
    os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")

    # SPECFEM2D MPI stages (each rank will see exactly one GPU):
    _run_step(int(os.environ["SLURM_NTASKS"]), "./bin/xmeshfem2D")
    _run_step(int(os.environ["SLURM_NTASKS"]), "./bin/xspecfem2D")

    # Copy results (serial)
    Path("run0001/OUTPUT_FILES").mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-r", "OUTPUT_FILES/.", "run0001/OUTPUT_FILES/"], check=True)


# This is when you run this code on a local machine with single GPU
def forward_obs_local(workdir, n_task):
    # directories: directory to working directory and par_file(s)
    # Run the mesher and solver to generate our initial model

    # First we will set the correct SOURCE and STATION files.
    # This is the same task as shown in ./run_this_example.sh
    os.chdir(SPECFEM2D_DATA)

    # changing directory to the workdir
    os.chdir(workdir)

    # Run the mesher and solver to generate our initial model
    bash_I = 'mpirun -n ' +str(n_task)+ ' ./bin/xmeshfem2D'
    bash_II = 'mpirun -n ' +str(n_task)+ ' ./bin/xspecfem2D'
    bash_III = 'cp -r OUTPUT_FILES/. run0001/OUTPUT_FILES/'
    
    
    # memory before  SPECFEM2D - MESHER
    os.system(bash_I)    
    os.system(bash_II)
    os.system(bash_III)
    
    return None


def _read_float32_exact(path, n_values):
    """Read exactly n_values float32s; raise if file size doesn't match."""
    expected_bytes = n_values * 4  # float32
    try:
        size_bytes = os.path.getsize(path)
    except OSError as e:
        raise FileNotFoundError(f"Cannot stat {path}: {e}")

    if size_bytes != expected_bytes:
        raise ValueError(
            f"{os.path.basename(path)} has {size_bytes} bytes, "
            f"expected {expected_bytes} (n_values={n_values})."
        )

    arr = np.fromfile(path, dtype=np.float32, count=n_values)
    if arr.size != n_values:
        raise ValueError(f"Read {arr.size} values from {path}, expected {n_values}.")
    return arr


def data_1d(
    n_shots,
    SPECFEM2D_WORKDIR,
    *,
    n_receivers=n_recievers,
    nstep=nstep,
    n_sample=n_sample,
    flatten_order="C",
):
    """
    Load Ux_file_single_d.bin and Uz_file_single_d.bin for each run (run0001..),
    stack as [Ux; Uz] per run into shape (2*values_per_component, n_shots),
    then flatten to 1-D.

    Definitions
    -----------
    samples_per_trace = nstep // n_sample  (must divide evenly)
    values_per_component = n_receivers * samples_per_trace
    values_per_shot = 2 * values_per_component  (Ux + Uz)

    Returns
    -------
    vector : np.ndarray, shape (n_shots * values_per_shot,), dtype float32
    """
    
    n_shots     = int(n_shots)
    n_receivers = int(n_receivers)
    nstep       = int(nstep)
    n_sample    = int(n_sample)
    
    # Validate and derive sizes
    if nstep % n_sample != 0:
        raise ValueError(f"nstep ({nstep}) must be divisible by n_sample ({n_sample}).")

    samples_per_trace = nstep // n_sample
    values_per_component = n_receivers * samples_per_trace
    values_per_shot = 2 * values_per_component

    # Preallocate (rows: Ux then Uz; cols: shots)
    data_array = np.empty((values_per_shot, n_shots), dtype=np.float32)

    for i in range(n_shots):
        run_id = f"run{str(i + 1).zfill(4)}"
        data_dir = os.path.join(SPECFEM2D_WORKDIR, run_id, "OUTPUT_FILES")

        ux_path = os.path.join(data_dir, "Ux_file_single_d.bin")
        uz_path = os.path.join(data_dir, "Uz_file_single_d.bin")

        # Each component file must have exactly values_per_component float32s
        data_array[0:values_per_component, i] = _read_float32_exact(
            ux_path, values_per_component
        )
        data_array[values_per_component:, i] = _read_float32_exact(
            uz_path, values_per_component
        )

    # Flatten to match your save/load convention
    return data_array.flatten(order=flatten_order)

def mq_vectorial(target, vector_data_1d):
    # Ensure 1-D and float32 to avoid upcasting
    target = np.asarray(target, dtype=np.float32)
    vector_data_1d = np.asarray(vector_data_1d, dtype=np.float32)

    if target.ndim != 1 or vector_data_1d.ndim != 1:
        raise ValueError("mq_vectorial expects 1-D arrays.")
    if target.shape[0] != vector_data_1d.shape[0]:
        raise ValueError(f"Length mismatch: target={target.shape[0]}, data={vector_data_1d.shape[0]}")

    vec = target - vector_data_1d
    return vec.reshape((vec.shape[0], 1))
   

def sims(machine_type): 

    if machine_type == False:
        forward_obs_local(SPECFEM2D_WORKDIR, n_task)
    if machine_type == True:
        forward_obs_cluster(SPECFEM2D_WORKDIR, n_task)

    return None
