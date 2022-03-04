# Installation
Follow the steps given below to install the `liionpack` Python package. The package must be installed to run the included examples. It is recommended to create a virtual environment for the installation, in order not to alter any distribution python files.

## Create a virtual environment

### Using virtualenv
To create a virtual environment `env` within your current directory type:

```bash
# Create a virtual env
virtualenv env

# Activate the environment
source env/bin/activate
```

Now all the calls to pip described below will install `liionpack` and its dependencies into the environment `env`. When you are ready to exit the environment and go back to your original system, just type:

```bash
deactivate
```

### Using conda
Alternatively, use Conda to create a virtual environment then install the `liionpack` package.

```bash
# Create a Conda virtual environment
conda create -n liionpack python=3.8

# Activate the conda environment
conda activate liionpack
```

Now all the calls to pip described below will install `liionpack` and its dependencies into the environment `env`. When you are ready to exit the environment and go back to your original system, just type:

```bash
conda deactivate
```

## Using pip
Execute the following command to install `liionpack` with pip:

```bash
pip install liionpack
```

## Install from source (developer install)
This section describes the build and installation of `liionpack` from the source code, available on GitHub. Note that this is not the recommended approach for most users and should be reserved to people wanting to participate in the development of `liionpack`, or people who really need to use bleeding-edge feature(s) not yet available in the latest released version. If you do not fall in the two previous categories, you would be better off installing `liionpack` using pip.

Run the following command to install the newest version from the Github repository:
To obtain the `liionpack` source code, clone the GitHub repository.

```bash
git clone https://github.com/pybamm-team/liionpack.git
```
From the `liionpack/` directory, you can install `liionpack` using -
```bash
# Install the liionpack package from within the repository
$ pip install -e .
```
