# import libraries
import os, shutil, subprocess, sys
from pathlib import Path

#import matplotlib
#matplotlib.use('Qt5Agg')

os.environ["XDG_SESSION_TYPE"] = "xcb"

# os.system('python data_generation.py')

from config import SPECFEM2D_WORKDIR, SURR_PATH
from config import iterations, initial_model, sigma

# Repo root = directory containing this MCMC.py
REPO_ROOT = Path(__file__).resolve().parent

# Source locations inside the repo
CPML_SRC  = REPO_ROOT / "utils" / "CPML"
FRAME_SRC = REPO_ROOT / "utils" / "Frame"

# Destinations inside your working SPECFEM tree
CPML_DST  = Path(SPECFEM2D_WORKDIR) / "CPML"
FRAME_DST = Path(SPECFEM2D_WORKDIR) / "Frame"

# Sanity: ensure sources exist
if not CPML_SRC.exists():
    raise FileNotFoundError(f"CPML source not found at {CPML_SRC}")
if not FRAME_SRC.exists():
    raise FileNotFoundError(f"Frame source not found at {FRAME_SRC}")

# Copy (idempotent)
shutil.copytree(CPML_SRC, CPML_DST, dirs_exist_ok=True)
shutil.copytree(FRAME_SRC, FRAME_DST, dirs_exist_ok=True)

# Build CPML tools if needed
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

# Ensure executables
for p in (add_bin, conv_bin):
    if p.exists():
        os.chmod(p, 0o755)

# 1) Remove Cubit from PYTHONPATH (if present)
pp = os.environ.get("PYTHONPATH", "")
if pp:
    parts = [p for p in pp.split(":") if "Coreform-Cubit" not in p]
    os.environ["PYTHONPATH"] = ":".join(parts)

# 2) Remove Cubit from sys.path (this is what controls imports)
sys.path[:] = [p for p in sys.path if "Coreform-Cubit" not in (p or "")]

# 3) If sklearn was already imported from the wrong place, drop it
for k in list(sys.modules.keys()):
    if k.startswith("sklearn"):
        del sys.modules[k]

# Optional: sanity check (keeps you from silently using Cubit sklearn)
import sklearn
print("sklearn from:", sklearn.__file__)
if "Coreform-Cubit" in sklearn.__file__:
    raise RuntimeError("Still importing sklearn from Cubit. sys.path cleanup failed.")

######################### Copy surrogate directory #########################

# Source: surrogate directory in current working directory
SURR_SRC = Path.cwd() / "surrogate"
# Destination: SURR_PATH from config
SURR_DST = Path(SURR_PATH)

# Sanity check
if not SURR_SRC.exists():
    raise FileNotFoundError(f"Surrogate source not found at {SURR_SRC}")

# Ensure destination parent exists
SURR_DST.parent.mkdir(parents=True, exist_ok=True)
# Copy (idempotent: overwrites existing files)
shutil.copytree(SURR_SRC, SURR_DST, dirs_exist_ok=True)


######################### Run Metropolis-Hastings (NN) - NORMAL #########################
from config import BASE_DATA, TARGET_DATA, n_shots, n_recievers, nstep, n_sample
from utils.load_data import load_data

# Load baseline and target data once
n_values = n_shots*(2*n_recievers*(nstep//n_sample))

baseline_data = load_data(os.path.join(BASE_DATA, "data.bin"), n_values)
target_data = load_data(os.path.join(TARGET_DATA, "data.bin"), n_values)    # Here we pass the noise target data

from src.nn_mcmc import MH
MH(baseline_data, target_data, initial_model, iterations)
