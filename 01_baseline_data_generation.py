#!/usr/bin/env python
# coding: utf-8

import os
from src.seismic.specfem2d import forward

from utils.editlog import clean_file
from utils.cubit_mesher import mesh_and_PML
from utils.resource_monitor import ResourceMonitor
from utils.build_data import copy_and_save_data

from config import BASE_DIR, SPECFEM2D_WORKDIR, BASE_DATA, n_shots, sigma, n_sample, nstep, n_recievers, machine_type

# Start monitoring resources
mon = ResourceMonitor(interval=0.5)
mon.start()

mon.mark("setup_specfem")
os.system('python utils/setup_specfem2d.py')


# Ellipse parameters
x_center, z_center = 500, 500
major_axis, minor_axis = 200, 200
alpha = 0
proposed_vp = 2400

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
    REF_DATA=BASE_DATA,
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


