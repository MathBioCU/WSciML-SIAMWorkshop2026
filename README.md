# WSciML SIAM Workshop 2026

Welcome to the repository accompanying the 2026 SIAM workshop on Weak-form Scientific Machine Learning (WSciML)!

## Initialization
Start by running the following to allow git to properly initialize submodules:
git clone --recurse-submodules https://github.com/MathBioCU/WSciML-SIAMWorkshop2026.git

## Running example scripts
1. Scripts to run WSINDy and WENDy algorithms on test sets are found in the python\_scripts and matlab\_scripts folders. 
2. Corresponding source files are found in the libs folder. 
3. To run the python scripts, it is recommended to create a conda env using 
`conda env create -f environment.yml`
4. If you would prefer to use pip, starting from a python>=3.14 environment run
`pip install -r requirements.txt`

## Additional data
1. Several PDE datasets are located in libs/wsindy\_obj\_data/pde\_data
2. A collection of other PDE datasets is found at https://zenodo.org/records/20787783
3. It is recommended that workshop attendees download the data from Zenodo, can change script paths accordingly
4. This is necessary to utilize larger datasets, such as the Navier-Stokes flow passed a cylinder data (Nav\_Stokes.mat)

