
import shutil
import os, sys

# Add the config path to the directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import WORKDIR, SPECFEM2D, SPECFEM2D_BIN_ORIGINAL, SPECFEM2D_DATA_ORIGINAL
from config import SPECFEM2D_WORKDIR, SPECFEM2D_OUTPUT
from config import n_shots


def pre_check():    
    # Download SPECFEM2D from GitHub, devel branch for latest codebase OR symlink from existing repo
    if not os.path.exists(WORKDIR):
        os.makedirs(WORKDIR)
    os.chdir(WORKDIR)

    if os.path.exists("specfem2d"):
        print("SPECFEM2D repository already found, you may skip this subsection.")
        pass
    elif os.path.exists(SPECFEM2D):
        print("Existing SPECMFE2D respository found, symlinking to working directory")
        if os.access(os.path.dirname(os.path.join(WORKDIR, "specfem2d")), os.W_OK):
            os.symlink(SPECFEM2D, os.path.join(WORKDIR, "specfem2d"))
        else:
            print("You don't have permission to create a symbolic link in the destination directory.")
    else:
        print("Cloning respository from GitHub")
        bash_1 = "git clone --recursive --branch devel https://github.com/geodynamics/specfem2d.git"
        os.system(bash_1)
            
    return None

def pre_sim_reqs(n_sources):
    # Incase we've run this docs page before, delete the working directory before remaking
    if os.path.exists(SPECFEM2D_WORKDIR):
        shutil.rmtree(SPECFEM2D_WORKDIR)

    
    os.mkdir(SPECFEM2D_WORKDIR)
    os.chdir(SPECFEM2D_WORKDIR)
    
    # os.mkdir("NEURAL_NETWORK")
    # os.mkdir("NEURAL_NETWORK/TRAIN")

    # os.mkdir("NEURAL_NETWORK/TEST")
    # os.mkdir("NEURAL_NETWORK/TEST")
    # os.mkdir("NEURAL_NETWORK/TEST")
    
    # os.mkdir("DATA_PLOT")
    # os.mkdir("DATA_PLOT/SHOT")
    # os.mkdir("DATA_PLOT/TRACE")
    # os.mkdir("DATA_PLOT/TRACE/X_COMPONENT")
    # os.mkdir("DATA_PLOT/TRACE/Z_COMPONENT")
    # os.mkdir("DATA_PLOT/MODEL")


    # Copy the binary files incase we update the source code. These can also be symlinked.
    shutil.copytree(SPECFEM2D_BIN_ORIGINAL, "bin")

    # Copy the DATA/ directory because we will be making edits here frequently and it's useful to
    # retain the original files for reference. We will be running one of the example problems: Tape2007
    shutil.copytree(SPECFEM2D_DATA_ORIGINAL, "DATA")

    # # SPECFEM requires that we create the OUTPUT_FILES directory before running
    os.chdir(SPECFEM2D_WORKDIR)
    
    # if os.path.exists(SPECFEM2D_OUTPUT):
    #     shutil.rmtree(SPECFEM2D_OUTPUT)

    os.mkdir(SPECFEM2D_OUTPUT)
    
    for i in range(n_sources):
        run_name = f"run{int(i+1):04d}"  # zero-pads to 4 digits
        shutil.copytree(os.path.join(SPECFEM2D, run_name), run_name)

    return None

pre_check()
pre_sim_reqs(n_shots)
