# Wheelhouse

This directory is used to store pre-built Python wheels for offline installation on DRAC compute nodes.

DRAC compute nodes do not have internet access, so any packages that are not available in the central DRAC `scipy-stack` or standard python modules must be downloaded and built into wheels on a login node (which has internet) first.

## ViennaRNA
ViennaRNA is required by `AUGMENT` but is not always available as a pre-built wheel for the DRAC environment. 

To build it, run the provided script from a login node:
```bash
bash hpc/build_viennarna_wheel.sh
```

This script will download the source tarball from the official website and compile it into a `.whl` file here. Once the `.whl` file is in this directory, `setup_env.sh` will automatically find it and install it offline on the compute nodes.
