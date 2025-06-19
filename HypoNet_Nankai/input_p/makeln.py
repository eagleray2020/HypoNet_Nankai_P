import os
import numpy as np


import os

def create_symlinks_to_directories(source_dir, nx, ny):
    # Loop through the range of ix and iy to create symbolic links
    for ix in range(1, nx+1):
        for iy in range(1, ny+1):
            dir_name = f"dir_x{ix}_y{iy}"
            try:
                # Create symbolic link in the current directory
                os.symlink(os.path.join(source_dir, dir_name), dir_name)
                print(f"Symbolic link created for {dir_name}")
            except OSError as e:
                print(f"Failed to create symbolic link for {dir_name}: {e}")

# Specify the directory containing the target directories
source_directory = "/S/data01/G3506/p0791/NT-T2Net/PINN/train/MFF_DDP_surfsource_dividedPINN/dividedPINN_nx7_ny5_20_512_20_384_sigma0.1_0.0/"
# Set the range of ix and iy
div_setting = np.genfromtxt("divide_setting", comments="#")
nx, ny = div_setting[1,0].astype(np.int32), div_setting[1,1].astype(np.int32)

# Create symbolic links to directories
create_symlinks_to_directories(source_directory, nx, ny)
