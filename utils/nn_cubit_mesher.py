# from __future__ import print_function
#!/usr/bin/env python
# coding: utf-8

import os
import sys
import shutil
import subprocess
from contextlib import contextmanager

#                            BEFORE RUNNING THE CODE RUN THE FOLLOWING INSIDE THE VENV:

# export PYTHONPATH=/opt/Coreform-Cubit-2023.11/bin:cubit.py$PYTHONPATH


from utils import change_params


@contextmanager
def prepend_sys_path(path):
    old = list(sys.path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = old


#                               ############### CREATING MESH IN CUBIT ###############

from config import WORKDIR, SPECFEM2D_ORIGINAL, SPECFEM2D_BIN_ORIGINAL, SPECFEM2D_DATA_ORIGINAL, SPECFEM2D_OUTPUT_ORIGINAL
from config import SPECFEM2D_WORKDIR, SPECFEM2D_BIN, SPECFEM2D_DATA, SPECFEM2D_OUTPUT, SPECFEM2D_SOLVER
from config import n_shots, device

#                                            ##### HYPER-PARAMETERS #####
# paths and directories
main_path = "../OUTPUT_FILES/seismic"
pathToSpecfem_PML = "../../../cubit2specfem2d_new_version.py"

global cpml_path
cpml_path = os.path.join(WORKDIR, "../../../../../utils/CPML")

# "/home/khan/Desktop/MCMC/GlobalOptimizationMethod/MCMC_py_files/utils/CPML"


# Meshing hyper-parameters
cell_size = 10             # ideal value is 12.5


#####################   THESE FUNCTIONS MUST BE USED   #####################

# The following function must be used just for the "data_generation.py" function which is used for modeling during injection:


def mesh_and_PML(x_center, z_center, major_axis, minor_axis, alpha, proposed_P):


    # PHYSICAL PARAMS
    proposed_P = round(proposed_P, 1)
    reservoir_RHO = 2100
    # keep a consistent numeric format (always one decimal place)
    new_content = (
        f"1 1 {reservoir_RHO:.1f} {proposed_P:.1f} 1474 0 0 9999 9999 0 0 0 0 0 0 # <-- anomaly"
    )

    # modify main Par_file
    change_params.modify_par_file(
        os.path.join(SPECFEM2D_DATA, "Par_file_NN"), 294, new_content
    )
    
    # CHANGE ALL PARALLEL RUNS AS WELL
    for i in range(n_shots):
        run_name = f"run{(i+1):04d}"
        par_path = os.path.join(SPECFEM2D_WORKDIR, run_name, "DATA", "Par_file_NN")
        change_params.modify_par_file(par_path, 294, new_content)

    # MESH GENERATION
    ellipse(major_axis, minor_axis, x_center, z_center, alpha)

    
    if os.path.exists("cubit01.jou"):
        os.remove("cubit01.jou")

    def get_CPML(want_pml):

        if want_pml == True:

            src_files = ['free_surface_file', 'materials_file', 'mesh_file', 'nodes_coords_file']
            for source_file in src_files:
                # Extract the filename from the source file path
                filename = os.path.basename(source_file)

                # Construct the destination file path
                destination_file = os.path.join("utils/CPML", filename)

                # Copy the file
                shutil.move(source_file, destination_file)

            os.chdir("utils/CPML")

            # RIGHT NOW CODE IS COMPLIED FOR X-WELL SCENARIO
            # THE .F90 CODE NEEDS TO BE CHANGED AND COMPILED AGAIN FOR SEURFACE CASE
            command = "./xadd_CPML_layers_to_an_existing_mesh > /dev/null"
            os.system(command)
            command = "./xconvert_external_layers_of_a_given_mesh_to_CPML_layers > /dev/null"
            os.system(command)

            print("\n\n\n")
            print(f"\tPML has been added to model and it is in {os.getcwd()}")
            print("\n\n\n")

            os.chdir("../..")


        else:
            src_files = ['utils/CPML/free_surface_file', 'utils/CPML/materials_file', 'utils/CPML/mesh_file', 'utils/CPML/nodes_coords_file', 'utils/CPML/absorbing_surface_file', 'utils/CPML/absorbing_cpml_file']
            dst_dir =  os.path.join(SPECFEM2D_DATA, "TEMP")
            # Iterate over each source file and copy it to the destination directory
            for source_file in src_files:
                # Extract the filename from the source file path
                filename = os.path.basename(source_file)

                # Construct the destination file path
                destination_file = os.path.join(dst_dir, filename)

                # Copy the file
                shutil.move(source_file, destination_file)
        return None


    get_CPML(want_pml=True)
    get_CPML(want_pml=False)

    # Iterate over the files and remove them
    src_files = ['free_surface_file', 'materials_file', 'mesh_file', 'nodes_coords_file']

    for file_name in src_files:
        file_path = os.path.join("/home/khan/Desktop/MCMC/GlobalOptimizationMethod/MCMC_py_files/utils/", file_name)      
        if os.path.exists(file_path):
            os.remove(file_path)

    return None

def MCMC_automatic_mesh(x_center, z_center, major_axis, minor_axis, alpha, proposed_P):

    # PHYSICAL PARAMS
    proposed_P = round(proposed_P, 1)
    reservoir_RHO = 2100
    # keep a consistent numeric format (always one decimal place)
    new_content = (
        f"1 1 {reservoir_RHO:.1f} {proposed_P:.1f} 1474 0 0 9999 9999 0 0 0 0 0 0 # <-- anomaly"
    )

    # modify main Par_file
    change_params.modify_par_file(
        os.path.join(SPECFEM2D_DATA, "Par_file_NN"), 294, new_content
    )
    
    # CHANGE ALL PARALLEL RUNS AS WELL
    for i in range(n_shots):
        run_name = f"run{(i+1):04d}"
        par_path = os.path.join(SPECFEM2D_WORKDIR, run_name, "DATA", "Par_file")
        change_params.modify_par_file(par_path, 294, new_content)
    
    # MESH GENERATION
    ellipse(major_axis, minor_axis, x_center, z_center, alpha)

    if os.path.exists("cubit01.jou"):
        os.remove("cubit01.jou")

    def get_CPML(want_pml: bool):
        base_src = ['free_surface_file', 'materials_file', 'mesh_file', 'nodes_coords_file']
        CPML_DIR  = os.path.join(SPECFEM2D_WORKDIR, "CPML")
        DATA_TEMP = os.path.join(SPECFEM2D_WORKDIR, "DATA", "TEMP")
        os.makedirs(CPML_DIR, exist_ok=True)
        os.makedirs(DATA_TEMP, exist_ok=True)

        if want_pml:
            missing = [f for f in base_src if not os.path.exists(f)]
            if missing:
                raise FileNotFoundError(f"Missing pre-CPML files: {missing} (cwd={os.getcwd()})")
            for f in base_src:
                shutil.move(f, os.path.join(CPML_DIR, f))

            add_bin  = os.path.join(CPML_DIR, "xadd_CPML_layers_to_an_existing_mesh")
            conv_bin = os.path.join(CPML_DIR, "xconvert_external_layers_of_a_given_mesh_to_CPML_layers")
            if not (os.path.exists(add_bin) and os.path.exists(conv_bin)):
                raise FileNotFoundError(f"CPML tools not found at {add_bin} / {conv_bin}")
            subprocess.run([add_bin],  cwd=CPML_DIR, check=True)
            subprocess.run([conv_bin], cwd=CPML_DIR, check=True)

        else:
            expected = base_src + ['absorbing_surface_file', 'absorbing_cpml_file']
            missing = [f for f in expected if not os.path.exists(os.path.join(CPML_DIR, f))]
            if missing:
                raise FileNotFoundError(f"After CPML, missing: {missing}")
            for f in expected:
                shutil.move(os.path.join(CPML_DIR, f), os.path.join(DATA_TEMP, f))


    get_CPML(want_pml=True)
    get_CPML(want_pml=False)
    
    # Iterate over the files and remove them
    src_files = ['free_surface_file', 'materials_file', 'mesh_file', 'nodes_coords_file']

    for file_name in src_files:
        file_path = file_name
        # os.path.join("/home/khan/Desktop/MCMC_FILES_&_CODES/GlobalOptimizationMethod/MCMC_py_files/utils/", file_name)  
        if os.path.exists(file_path):
            os.remove(file_path)

    return None


# MESHING SUB-ROUTINES HERE:
def ellipse(major_axis, minor_axis, x_center, z_center, alpha):
    """Build geometry, mesh, and export to SPECfem2D files (current working dir)."""

    with prepend_sys_path("/opt/Coreform-Cubit-2024.8/bin"):
        import cubit
        # OPTIONAL: defensive check
        import os
        print("Cubit loaded from:", getattr(cubit, "__file__", "unknown"))

    # Creating geometry
    cubit.cmd('create surface rectangle width 1000 height 1000 yplane')
    cubit.cmd('move volume all x 500 z 500')
    cubit.cmd(f'create surface ellipse major radius {major_axis} minor radius {minor_axis} yplane')
    cubit.cmd(f'rotate surface 2 about 0 1 0 angle {alpha}')
    cubit.cmd(f'volume 2 move x {x_center} z {z_center}')

    cubit.cmd('subtract surface 2 from surface 1 imprint keep')
    cubit.cmd('delete vol 1')
    cubit.cmd('compress')
    cubit.cmd('merge volume all')
    cubit.cmd('imprint volume all')
    cubit.cmd(f'surface all size {cell_size}')
    cubit.cmd('surface all scheme pave')
    cubit.cmd('mesh surface all')

    cubit.cmd('block 1 face in surface 1')
    cubit.cmd('block 1 name "anomaly"')
    cubit.cmd('block 1 attribute count 1')
    cubit.cmd('block 1 attribute index 1 1') 
    cubit.cmd('block 1 element type QUAD4')


    cubit.cmd('block 2 face in surface 2')
    cubit.cmd('block 2 name "background"')
    cubit.cmd('block 2 attribute count 1')
    cubit.cmd('block 2 attribute index 1 2') 
    cubit.cmd('block 2 element type QUAD4')

    cubit.cmd('merge all')

    # cubit.cmd(f'save cub5 "{main_path}/test.cub5" overwrite journal')
    cubit.cmd('set journal off')
    cubit.cmd('set echo off')



    #                               ############### EXPORTING TO SPECFEM2D FORMAT ###############

    # Module to convert the mesh to specfem2d format:
    from utils.cubit2specfem2d_new_version import mtools,block_tools,mesh_tools,mesh

    # Export the mesh under specfem2d format:
    profile=mesh() # Store the mesh from Cubit
    profile.write() # Write it into files (in specfem2d format)


    cubit.cmd('reset')
    return None  

