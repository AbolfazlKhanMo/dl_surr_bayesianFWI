#!/usr/bin/env python
# coding: utf-8

import os
# --------------------------------------------------------------------------------------------------------------------------
#                                        CONFIG FILE FOR STOCHASTIC SIMULATIONS - McMC
# --------------------------------------------------------------------------------------------------------------------------


#                                          --------------------------------                                #
#                                                    USER CHANGE THIS                                      #
#                                          --------------------------------                                #

# DIRECTORIES TO FILES AND PARAMETERS FOR (CIRCLE IN A HOMOGENEOUS MEDIA)
SPECFEM2D = "/home/abolfazl/specfem2d"
BASE_DIR  = "/home/abolfazl/bayesian-FWI/__.scratch.__/02-ddsurr/00_ellipse_dof_3_training_dataset_I(500)"

# Uncomment for DRAC clusters
# BASE_DIR = "/home/akhanmo/projects/def-amalcolm/akhanmo/mcmc/00-reduced-param/04_half_ellipse"

# where WORKDIR: points to your own working directory
# and SPECFEM2D: points to an existing specfem2D repository if available (if not set as '')


XwellGeo = False         # True: CROSS-WELL GEOMETRY / False: SURFACE SEISMIC
sim_type = 2             # Simulation type: (1) ACOUSTIC (2) ELASTIC
shot_number_to_plot = 1  # Shot number to plot shot and trace for data difference
n_shots = 1              # Number of simultaneous shots
n_cores = 1              # Number of dedicated cores to EACH SHOT!

machine_type = False      # True: remote (cluster) / False: local machine
device = 'gpu'

n_recievers = 16         # Number of recievers
nt = 2.5E-5              # Simulation time step
nstep = 2.4E4            # Number of time steps
n_sample = 160           # NTSTEP_BETWEEN_OUTPUT_SAMPLE from the parfile


                  ################# SAMPLING HYPER-PARAMS ##############
sigma           = 5.0                 # Noise percentage to each datapoint
stdvs           = [50, 50, 50]        # Standard deviation for each model parameter
initial_model   = [300, 300, 100]     # Initial Model - ONLY LOCATION
parameter_min = [100, 100, 100]       # Minimum bound for proposals for each parameter (BUT ONLY RADIUS MATTERS FOR THIS EXAMPLE)
parameter_max = [900, 900, 300]       # Maximum bound for proposals for each parameter (BUT ONLY RADIUS MATTERS FOR THIS EXAMPLE)

# REGIME SELECTION
# Options: "adaptive", "offline", "two_stage"
regime = "adaptive"

# SURROGATE-ASSISTED McMC CONTROLS
gamma = 2.5e-1                     # Value for triggering UQ-aware refinement (adaptive only)
beta = 5e-2                        # Value for triggering random refinement (adaptive only)
K = 5                              # MC Dropout samples/forward passes
R = 2                              # Max refinement iterations per MCMC step (adaptive only)

iterations      = 500              # Number of iterations/samples for MCMC


            ################# INTERPOLATION & VISUALIZATION PARAMS ##############
domain_xmin = 0               # Minimum domain value for X
domain_xmax = 1000            # Maximum domain value for X
domain_zmin = 0               # Minimum domain value for Z
domain_zmax = 1000            # Maximum domain value for Z

resX = 256                    # X-resoultion for interpolation
resY = 256                    # y-resoultion for interpolation




#                                          --------------------------------                                #
#                                              USER DO NOT CHANGE THIS                                     #
#                                          --------------------------------                                #


n_task = n_shots*n_cores # Number of processors for MPI: Number of Scotch (mesh decomposition)

# Distribute the necessary file structure of the SPECFEM2D repository that we will downloaded/reference
WORKDIR = os.path.join(BASE_DIR, "OUTPUT_FILES/seismic/scratch/SPECFEM2D")
SPECFEM2D_ORIGINAL = os.path.join(WORKDIR, "specfem2d")
SPECFEM2D_BIN_ORIGINAL = os.path.join(SPECFEM2D_ORIGINAL, "bin")
SPECFEM2D_DATA_ORIGINAL = os.path.join(SPECFEM2D_ORIGINAL, "DATA")
SPECFEM2D_OUTPUT_ORIGINAL = os.path.join(SPECFEM2D_ORIGINAL, "OUTPUT_FILES")

# The SPECFEM2D working directory that we will create separate from the downloaded repo
SPECFEM2D_WORKDIR = os.path.join(WORKDIR, "specfem2d_workdir")
SPECFEM2D_BIN = os.path.join(SPECFEM2D_WORKDIR, "bin")
SPECFEM2D_DATA = os.path.join(SPECFEM2D_WORKDIR, "DATA")
SPECFEM2D_OUTPUT = os.path.join(SPECFEM2D_WORKDIR, "OUTPUT_FILES")
SPECFEM2D_SOLVER = os.path.join(SPECFEM2D_WORKDIR, "SOLVER")

BASE_DATA = os.path.join(SPECFEM2D_WORKDIR, "BASE_DATA")       # No noise data for baseline model
MONITOR_DATA = os.path.join(SPECFEM2D_WORKDIR, "MONITOR_DATA") # No noise data for monitor model

TARGET_DATA = os.path.join(SPECFEM2D_WORKDIR, "TARGET_DATA")   # Add noise on data difference, this is target data for McMC/SVGD

SURR_PATH = os.path.join(SPECFEM2D_WORKDIR, "surrogate")       # Path to save/load surrogates
