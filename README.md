# Bayesian Full-Waveform Inversion Using Deep Learning Surrogate Models

Source code for Bayesian full waveform inversion using deep learning surrogate models. The code couples a DL-based surrogate model with the SPECFEM2D forward solver to accelerate Bayesian inference of subsurface velocity structures.

## Overview

The workflow estimates posterior distributions of sought after parameters from seismic waveform data. Three McMC regimes are supported:

- **Adaptive**: the surrogate is refined on-the-fly during sampling using UQ-aware and random refinement triggers.
- **Offline**: the surrogate is pre-trained on a fixed dataset and used without further updates.
- **Two-stage**: an initial surrogate-only chain is followed by refinement with the exact solver.

A convolutional neural network (CNN) regressor with MC Dropout provides both misfit predictions and epistemic uncertainty estimates, which guide when to call the expensive SPECFEM2D solver for surrogate retraining.

## Requirements

- Python 3.8+
- PyTorch
- NumPy, SciPy, scikit-learn, joblib, matplotlib
- SPECFEM2D (compiled separately; set the path in `config.py`)
- Coreform Cubit (for mesh generation)
- A Fortran compiler (for building CPML utilities)

## Repository Structure

```
config.py                       -- Central configuration (paths, geometry, hyperparameters)
01_baseline_data_generation.py  -- Generate synthetic baseline seismic data
02_monitor_data_generation.py   -- Generate monitor survey data
03_noiseless_target_data_generation.py -- Noise-free target data from baseline/monitor difference for training data generation
03_noisy_target_data_generation.py     -- Noisy target data with controlled noise level
04_McMC.py                      -- Standard McMC inversion using the exact SPECFEM2D solver
05_Sobol_dgen.py                -- Sobol-sequence-based training dataset generation
05_Sobol_dgen_mps.py            -- Sobol data generation with NVIDIA MPS support
launch_mps_sobol.sh             -- Shell script for launching MPS-based Sobol generation
06_NN_McMC.py                   -- Surrogate-assisted McMC inversion
generate_STATIONS.py            -- Generate receiver station files

src/
    McMC.py                     -- Metropolis-Hastings sampler (exact solver)
    nn_McMC.py                  -- Metropolis-Hastings sampler (surrogate-assisted)
    nn_evaluator.py             -- CNN surrogate inference with MC Dropout uncertainty
    sobol_data_gen.py           -- Sobol quasi-random sampling for training data
    seismic/specfem2d/forward.py -- SPECFEM2D forward simulation wrapper

surrogate/
    cnn.py                      -- CNN regressor architecture (ResBlocks, MC Dropout)
    train.py                    -- Training with k-fold cross-validation
    inference.py                -- Surrogate inference and evaluation
    OUTPUT_FILES/files/         -- Pre-trained model weights and scalers

utils/
    cubit_mesher.py             -- Cubit mesh generation for McMC proposals
    nn_cubit_mesher.py          -- Cubit mesh generation for surrogate pipeline
    build_data.py               -- Seismic data assembly and storage
    grid_interp.py              -- Grid interpolation for velocity models
    setup_specfem2d.py          -- SPECFEM2D directory setup
    CPML/                       -- Convolutional PML layer tools (Fortran)
```

## Usage

1. Edit `config.py` to set paths to your SPECFEM2D installation and working directory, and adjust simulation and hyperparameters as needed.

2. Run the data generation scripts in order:

```
python 01_baseline_data_generation.py
python 02_monitor_data_generation.py
python 03_noisy_target_data_generation.py
```

3. For standard McMC with the exact solver:

```
python 04_McMC.py
```

4. For surrogate-assisted McMC:

   a. Generate the training dataset using the Sobol sampler:

   ```
   python 05_Sobol_dgen.py
   ```

   b. Copy the generated `.npz` files into the surrogate dataset directory so that they match the names expected by the training script:

   ```
   surrogate/dataset/nn_surr/sobol_input_array.npz
   surrogate/dataset/nn_surr/sobol_label_misfit.npz
   ```

   c. Train the surrogate model. Before training, set the `adaptive` flag in `surrogate/train.py` to match the inference regime you intend to use. When `adaptive = True`, all data is used for training and validation (no held-out test set). When `adaptive = False`, a 70/20/10 train/validation/test split is used. Then run:

   ```
   cd surrogate
   python train.py
   ```

   d. Return to the root directory and run the surrogate-assisted McMC:

   ```
   cd ..
   python 06_NN_McMC.py
   ```

The McMC regime (adaptive, offline, or two-stage) and surrogate hyperparameters (refinement thresholds, MC Dropout samples) are set in `config.py`.

`05_Sobol_dgen_mps.py` and `launch_mps_sobol.sh` are provided as utilities for running the Sobol data generation with NVIDIA MPS (Multi-Process Service) on GPU clusters.

## Dataset

The dataset used in the accompanying paper is available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20381558.svg)](https://doi.org/10.5281/zenodo.20381558)

<!-- ## Citation

If you use this code, please cite:

```
@article{CITATION_KEY,
  title   = {TITLE},
  author  = {AUTHORS},
  journal = {JOURNAL},
  year    = {YEAR},
  doi     = {DOI}
}
``` -->

## License

surrogate-assisted Bayesian FWI
Copyright (C) 2026 Abolfazl Khan Mohammadi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.