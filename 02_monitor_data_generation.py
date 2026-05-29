#!/usr/bin/env python
# coding: utf-8

import os
from src.seismic.specfem2d import forward

from utils.editlog import clean_file
from utils.cubit_mesher import mesh_and_PML
from utils.resource_monitor import ResourceMonitor
from utils.build_data import copy_and_save_data

from config import MONITOR_DATA, SPECFEM2D_WORKDIR, BASE_DATA, n_shots, sigma, n_sample, nstep, n_recievers, machine_type

# Start monitoring resources
mon = ResourceMonitor(interval=0.5)
mon.start()


# Ellipse parameters
x_center, z_center = 507.4, 511
major_axis, minor_axis = 191.25, 191.25
alpha = 0
proposed_vp = 2000

mon.mark("mesh_and_PML")
mesh_and_PML(x_center, z_center, major_axis, minor_axis, alpha, proposed_vp)

mon.mark("forward_sims")
forward.sims(machine_type) 

# mon.mark("1d_data_assimilation")
# target = forward.data_1d(n_shots, SPECFEM2D_WORKDIR)


mon.mark("1d_data_add_noise_save")

target_path, length = copy_and_save_data(
    n_shots=n_shots,
    SPECFEM2D_WORKDIR=SPECFEM2D_WORKDIR,
    REF_DATA=MONITOR_DATA,
    n_receivers=n_recievers,
    nstep=nstep,
    n_sample=n_sample,
    sigma_percent=0,         # 0 noise here
    seed=123,                # optional, for reproducibility, this can be None
    flatten_order="C"        # matches data_1d default
)

mon.mark("finalize")
os.chdir("../../../../..")

# Stop monitor and print the summary to the log file
mon.stop()
mon.report()


